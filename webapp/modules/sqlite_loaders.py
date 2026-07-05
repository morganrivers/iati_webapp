"""
SQLite-backed replacements for forecast_with_few_shot data loaders.

_LazyMockForecasts   - dict-like; only queries SQLite when a key is accessed.
                       Keys() returns all IDs (lightweight, no content loaded).
load_activity_info_sqlite - loads all activity_info rows from SQLite.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd


class _LazyMockForecasts:
    """
    Dict-like wrapper around the mock_forecasts SQLite table.
    Content is fetched on demand so 1798 full text entries never all sit in RAM.
    Keys (just IDs) are loaded once lazily on first access.
    """

    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)
        self._keys: Optional[frozenset] = None  # loaded on first keys() call

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db)

    # ── Key set (IDs only, no content) ────────────────────────────────────────

    def _load_keys(self) -> frozenset:
        if self._keys is None:
            with self._conn() as c:
                rows = c.execute("SELECT activity_id FROM mock_forecasts").fetchall()
            self._keys = frozenset(r[0] for r in rows)
        return self._keys

    def keys(self) -> frozenset:
        return self._load_keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self._load_keys())

    def __len__(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM mock_forecasts").fetchone()[0]

    # ── Content access ─────────────────────────────────────────────────────────

    def _fetch(self, key: str) -> Optional[str]:
        with self._conn() as c:
            row = c.execute(
                "SELECT content FROM mock_forecasts WHERE activity_id = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def get(self, key: str, default: str = "") -> str:
        v = self._fetch(key)
        return v if v is not None else default

    def __contains__(self, key: object) -> bool:
        return str(key) in self._load_keys()

    def __getitem__(self, key: str) -> str:
        v = self._fetch(key)
        if v is None:
            raise KeyError(key)
        return v


def load_activity_info_sqlite(db_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load all activity_info rows from SQLite.
    Replaces the CSV + 2x JSONL scan in forecast_with_few_shot._load_activity_info.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM activity_info").fetchall()
    conn.close()
    return {
        r["activity_id"]: dict(r)
        for r in rows
        if r["activity_id"]
    }


def load_activity_dataframe_sqlite(db_path: Path) -> pd.DataFrame:
    """
    Load the full activity_info table as a raw DataFrame.
    Replaces reading info_for_activity_forecasting_old_transaction_types.csv;
    callers apply get_similar_activities.add_derived_columns to finish preparation.
    """
    with sqlite3.connect(str(db_path)) as conn:
        return pd.read_sql_query("SELECT * FROM activity_info", conn)


def make_lazy_mock_forecasts(db_path: Path) -> _LazyMockForecasts:
    return _LazyMockForecasts(db_path)


def load_bm25_corpus_sqlite(db_path: Path) -> Dict[str, str]:
    """
    Build activity_id -> text corpus for BM25 from activity_info in webapp.db.
    Uses activity_title + chatgpt_description (same text the Gemini embeddings were built from).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT activity_id, activity_title, chatgpt_description FROM activity_info"
    ).fetchall()
    conn.close()

    corpus: Dict[str, str] = {}
    for r in rows:
        aid = r["activity_id"]
        if not aid:
            continue
        title = r["activity_title"] or ""
        desc  = r["chatgpt_description"] or ""
        text  = (title + "\n\n" + desc).strip() if title else desc.strip()
        if text:
            corpus[str(aid)] = text
    return corpus
