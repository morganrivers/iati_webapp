#!/usr/bin/env python3
"""
Add an 'embeddings' table to webapp.db from activity_text_embeddings_gemini.jsonl.

Embeddings are stored as raw float32 BLOBs (~27 MB) instead of JSON text (~99 MB),
making the database small enough to commit to git.

Usage:
    python build_embeddings_db.py
    python build_embeddings_db.py --jsonl path/to/embeddings.jsonl --db path/to/webapp.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_JSONL = Path(__file__).resolve().parent.parent / "data" / "activity_text_embeddings_gemini.jsonl"
DEFAULT_DB    = Path(__file__).resolve().parent.parent / "data" / "webapp.db"


def build(jsonl_path: Path, db_path: Path) -> None:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            activity_id TEXT PRIMARY KEY,
            embedding   BLOB NOT NULL
        )
    """)
    conn.execute("DELETE FROM embeddings")  # full rebuild each run

    inserted = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            aid = obj.get("activity_id")
            vec = obj.get("embedding")
            if not aid or vec is None:
                continue
            arr = np.array(vec, dtype="float32")
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (activity_id, embedding) VALUES (?, ?)",
                (str(aid), arr.tobytes()),
            )
            inserted += 1
            if inserted % 200 == 0:
                print(f"  {inserted} embeddings inserted…")

    conn.commit()
    conn.execute("VACUUM")  # reclaim space from deleted 3072-dim rows
    conn.close()

    size_mb = db_path.stat().st_size / 1_048_576
    print(f"Done. {inserted} embeddings written to {db_path}  ({size_mb:.1f} MB total db size)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--db",   type=Path, default=DEFAULT_DB)
    args = ap.parse_args()
    build(args.jsonl, args.db)


if __name__ == "__main__":
    main()
