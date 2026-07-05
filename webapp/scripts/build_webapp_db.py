#!/usr/bin/env python3
"""
Build data/webapp.db from existing CSV and JSONL data files.

Run once (or after data updates) from the repo root or webapp directory:
    python webapp/scripts/build_webapp_db.py

Creates two tables:
  activity_info  - all CSV columns + chatgpt_description + risks_summary
  mock_forecasts - activity_id, content (retrospective forecast text)
"""
import csv
import json
import sqlite3
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = SCRIPT_DIR.parent
DATA_DIR = WEBAPP_DIR.parent / "data"

ACTIVITY_INFO_CSV = DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"
SUMMARIES_JSONL = DATA_DIR / "outputs_summaries.jsonl"
RISKS_JSONL = DATA_DIR / "outputs_risks.jsonl"
RETRO_JSONL = DATA_DIR / "outputs_retrospective_forecast.jsonl"
DB_PATH = DATA_DIR / "webapp.db"


def _load_jsonl_field(path: Path) -> dict[str, str]:
    """
    Extract activity_id -> text from a JSONL file.
    Tries response.content/text first, then response_text directly.
    For response_text that is JSON, tries to extract 'risks_summary' key.
    """
    out: dict[str, str] = {}
    if not path.exists():
        print(f"WARNING: {path} not found, skipping.")
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            aid = (data.get("activity_id") or "").strip()
            if not aid:
                continue

            text = ""
            resp = data.get("response")
            if isinstance(resp, dict):
                text = resp.get("content") or resp.get("text") or ""

            if not text:
                raw = (data.get("response_text") or "").strip()
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            text = parsed.get("risks_summary") or json.dumps(parsed)
                        else:
                            text = str(parsed)
                    except (json.JSONDecodeError, TypeError):
                        text = raw

            text = str(text).strip()
            if text:
                out[aid] = text

    return out


def build(db_path: Path = DB_PATH) -> None:
    print(f"Building {db_path} ...")

    # ── Load CSV ───────────────────────────────────────────────────────────────
    print(f"  Reading {ACTIVITY_INFO_CSV.name} ...", end=" ", flush=True)
    csv_rows: dict[str, dict] = {}
    csv_columns: list[str] = []
    with ACTIVITY_INFO_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        csv_columns = list(reader.fieldnames or [])
        for row in reader:
            aid = (row.get("activity_id") or "").strip()
            if aid:
                csv_rows[aid] = dict(row)
    print(f"{len(csv_rows):,} rows, {len(csv_columns)} columns")

    # ── Load JSONL supplements ─────────────────────────────────────────────────
    print(f"  Reading {SUMMARIES_JSONL.name} ...", end=" ", flush=True)
    chatgpt_descs = _load_jsonl_field(SUMMARIES_JSONL)
    print(f"{len(chatgpt_descs):,} entries")

    print(f"  Reading {RISKS_JSONL.name} ...", end=" ", flush=True)
    risks_summaries = _load_jsonl_field(RISKS_JSONL)
    print(f"{len(risks_summaries):,} entries")

    print(f"  Reading {RETRO_JSONL.name} ...", end=" ", flush=True)
    mock_forecasts = _load_jsonl_field(RETRO_JSONL)
    print(f"{len(mock_forecasts):,} entries")

    # ── Build DB ───────────────────────────────────────────────────────────────
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # activity_info table
    extra_cols = ["chatgpt_description", "risks_summary"]
    all_cols = csv_columns + [c for c in extra_cols if c not in csv_columns]
    col_defs = ", ".join(
        f'"{c}" TEXT' + (" PRIMARY KEY" if c == "activity_id" else "")
        for c in all_cols
    )
    cur.execute(f"DROP TABLE IF EXISTS activity_info")
    cur.execute(f"CREATE TABLE activity_info ({col_defs})")

    placeholders = ", ".join("?" for _ in all_cols)
    insert_sql = f'INSERT OR REPLACE INTO activity_info VALUES ({placeholders})'

    print(f"  Inserting activity_info ...", end=" ", flush=True)
    n_inserted = 0
    for aid, row in csv_rows.items():
        row["chatgpt_description"] = chatgpt_descs.get(aid, "")
        row["risks_summary"] = risks_summaries.get(aid, "")
        values = [row.get(c, "") or "" for c in all_cols]
        cur.execute(insert_sql, values)
        n_inserted += 1
    print(f"{n_inserted:,} rows")

    # mock_forecasts table
    cur.execute("DROP TABLE IF EXISTS mock_forecasts")
    cur.execute("""
        CREATE TABLE mock_forecasts (
            activity_id TEXT PRIMARY KEY,
            content     TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mock_aid ON mock_forecasts(activity_id)")

    print(f"  Inserting mock_forecasts ...", end=" ", flush=True)
    n_mock = 0
    for aid, content in mock_forecasts.items():
        cur.execute(
            "INSERT OR REPLACE INTO mock_forecasts (activity_id, content) VALUES (?, ?)",
            (aid, content),
        )
        n_mock += 1
    print(f"{n_mock:,} rows")

    conn.commit()
    conn.close()

    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"\nDone. {db_path} ({size_mb:.1f} MB)")
    print(f"  activity_info: {n_inserted:,} rows")
    print(f"  mock_forecasts: {n_mock:,} rows")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    build(target)
