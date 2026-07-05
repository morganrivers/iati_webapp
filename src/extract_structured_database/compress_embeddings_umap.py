#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute (1) sector-distance and (2) country-distance in full embedding space (cosine),
plus UMAP maps in 2D/3D/4D, and a 2D plot colored by sector.

Inputs:
  - ../../data/outputs_targets_embeddings.jsonl
      JSONL lines with at least: {"activity_id": "...", "embedding": [...]}
  - ../../data/info_for_activity_forecasting_old_transaction_types.csv
      Must contain: activity_id, dac5, recipient_iso3_fractions, and the date columns used by pick_start_date()

Uses:
  - pick_start_date imported from helpers_for_ratings_and_final_activity_features.py (DO NOT rewrite)

Outputs:
  - ../../data/outputs_targets_context_maps.jsonl
      One line per activity_id with:
        decade, sector, countries+weights, sector_distance, country_distance, umap_2d/3d/4d
  - ../../data/targets_umap2_by_sector.png

Fail-fast:
  - crashes on malformed recipient_iso3_fractions strings (non-empty but unparsable)
  - crashes on small sector-decade or country-decade groups (prints diagnostics first)
  - does NOT crash on missing start date; prints count
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd



import sys
import pickle


UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


from helpers_for_ratings_and_final_activity_features import pick_start_date
from get_codes_we_like import parse_dac_codes, get_good_bad_and_target_codes, categorize_good_code
from repo_paths import DATA_DIR

# -------------------- Paths --------------------
EMB_PATH = DATA_DIR / "outputs_targets_embeddings.jsonl"
# EMB_PATH = Path("../../data/outputs_pdo_page_embeddings.jsonl")
# EMB_PATH = Path("../../data/pdo_scan_pid_docs_first30p_pdo_embeddings.jsonl")
INFO_CSV = DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"

OUT_JSONL = DATA_DIR / "outputs_targets_context_maps.jsonl"
OUT_PLOT = DATA_DIR / "targets_umap2_by_sector.png"
OUT_PLOT_HTML = DATA_DIR / "targets_umap2_by_sector.html"
MODEL_SAVE_PATH = DATA_DIR / "trained_umap_models.pkl"

# When INCLUDE_VAL_IN_FIT=True, UMAP is fit on train+val (for test-set evaluation).
# Run with this flag set to True AFTER determining the train/val cutoff date from
# C_run_GLM_rating.py split output; set LATEST_TRAIN_VAL_POINT accordingly.
# Output goes to a separate JSONL so the val-mode JSONL is untouched.
INCLUDE_VAL_IN_FIT = True
LATEST_TRAIN_VAL_POINT = "2016-01-01"  # update this to the start date of the earliest test activity
OUT_JSONL_TRAINVAL = DATA_DIR / "outputs_targets_context_maps_trainval.jsonl"
MODEL_SAVE_PATH_TRAINVAL = DATA_DIR / "trained_umap_models_trainval.pkl"


# -------------------- Thresholds / knobs --------------------
PCA_DIMS = 50
UMAP_SEED = 42
UMAP_ENSEMBLE = False   # True: average 5 Procrustes-aligned runs; False: single seed (faster)

MIN_SECTOR_DECADE_N = 5

# For country-decade, check BOTH:
MIN_COUNTRY_DECADE_N = 5          # number of contributing activities (counting an activity if it lists that country)
MIN_COUNTRY_DECADE_WEIGHT = 2.0   # total weight mass (sum of shares)


def load_jsonl_text_by_activity_id(path: Path) -> Dict[str, dict]:
    m: Dict[str, dict] = {}
    n_missing_id = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            aid = str(obj.get("activity_id", "")).strip()
            if not aid:
                n_missing_id += 1
                continue
            m[aid] = {
                "text": obj.get("text", None),
                "section": obj.get("section", None),
                "model": obj.get("model", None),
            }
    if n_missing_id > 0:
        print(f"WARNING load_jsonl_text_by_activity_id({path.name}): {n_missing_id} lines missing activity_id skipped")
    if not m:
        raise ValueError(f"load_jsonl_text_by_activity_id: no valid entries loaded from {path}")
    return m

# -------------------- Country parsing --------------------

def parse_country_location(s: object) -> Optional[List[Tuple[str, float]]]:
    """
    Returns:
      - None if missing/empty -> caller will assign GLOBAL
      - list[(iso3, w)] if present
    Crashes on malformed non-empty strings.
    """
    if not isinstance(s, str) or not s.strip():
        return None

    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        # non-empty string but nothing parseable -> treat as malformed
        print("ERROR: recipient_iso3_fractions present but empty after split:", repr(s))
        raise SystemExit(1)

    out: List[Tuple[str, float]] = []
    for token in parts:
        if ":" not in token:
            print("ERROR: recipient_iso3_fractions token missing ':':", repr(token), "in", repr(s))
            raise SystemExit(1)
        iso3, w = token.split(":", 1)
        iso3 = iso3.strip().upper()
        try:
            wf = float(w.strip())
        except Exception:
            print("ERROR: recipient_iso3_fractions weight not float:", repr(token), "in", repr(s))
            raise SystemExit(1)
        if not iso3 or len(iso3) != 3:
            print("ERROR: recipient_iso3_fractions iso3 invalid:", repr(iso3), "in", repr(s))
            raise SystemExit(1)
        out.append((iso3, wf))

    # normalize weights
    total = sum(w for _, w in out)
    if total <= 0:
        print("ERROR: recipient_iso3_fractions weights sum to <= 0:", repr(s))
        raise SystemExit(1)
    out = [(c, w / total) for c, w in out]
    return out


# -------------------- Embedding helpers --------------------

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        return v
    return v / n

def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))


# -------------------- Loading --------------------

def load_embeddings_jsonl(path: Path) -> Dict[str, np.ndarray]:
    m: Dict[str, np.ndarray] = {}
    dup = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            aid = str(obj.get("activity_id", "")).strip()
            emb = obj.get("embedding", None)
            if not aid or not isinstance(emb, list) or not emb:
                continue
            arr = np.array(emb, dtype=np.float32)
            if aid in m:
                dup += 1
            m[aid] = arr
    if dup:
        print(f"WARNING: {dup} duplicate activity_id lines in embeddings; last one wins.")
    return m

def load_info_aggregated(path: Path) -> Dict[str, dict]:
    """
    Aggregate info_for_activity_forecasting_old_transaction_types.csv by activity_id:
      - union dac5 codes across rows
      - pick first non-null recipient_iso3_fractions
      - pick first non-null among date columns used by pick_start_date
    """
    df = pd.read_csv(path, dtype={"activity_id": str})
    df["activity_id"] = df["activity_id"].astype(str)

    date_cols = ["actual_start_date", "original_planned_start_date", "txn_first_date"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    agg: Dict[str, dict] = {}

    for _, row in df.iterrows():
        aid = row.get("activity_id")
        if not isinstance(aid, str) or not aid:
            continue

        r = agg.get(aid)
        if r is None:
            r = {
                "activity_id": aid,
                "dac5_codes": set(),
                "recipient_iso3_fractions": None,
                "actual_start_date": pd.NaT,
                "original_planned_start_date": pd.NaT,
                "txn_first_date": pd.NaT,
            }
            agg[aid] = r

        # dac5 union
        codes = parse_dac_codes(row.get("dac5", ""))
        if codes:
            r["dac5_codes"].update(codes)

        # recipient_iso3_fractions: first non-null wins
        cl = row.get("recipient_iso3_fractions", None)
        if r["recipient_iso3_fractions"] is None and isinstance(cl, str) and cl.strip():
            r["recipient_iso3_fractions"] = cl

        # dates: first non-null per column wins
        for c in date_cols:
            if pd.isna(r[c]) and c in row and pd.notna(row[c]):
                r[c] = row[c]

    return agg


# -------------------- Sector assignment (majority; tie -> least common globally) --------------------

def compute_global_category_counts(activity_rows: Dict[str, dict], good_codes: set[str]) -> Counter:
    """
    Mimic your pie behavior: count category occurrences across GOOD code assignments.
    """
    cat_counts = Counter()
    for r in activity_rows.values():
        codes = r["dac5_codes"] & good_codes
        for code in codes:
            cat_counts[categorize_good_code(code)] += 1
    return cat_counts

def pick_sector_for_activity(codes: set[str], good_codes: set[str], global_cat_counts: Counter) -> Optional[str]:
    good = codes & good_codes
    if not good:
        return None

    per_activity = Counter()
    for code in good:
        per_activity[categorize_good_code(code)] += 1

    if not per_activity:
        return None

    max_ct = max(per_activity.values())
    tied = [cat for cat, ct in per_activity.items() if ct == max_ct]
    if len(tied) == 1:
        return tied[0]

    # tie-breaker: least common globally
    tied_sorted = sorted(tied, key=lambda c: (global_cat_counts.get(c, 0), c))
    return tied_sorted[0]


# -------------------- Main computation --------------------
