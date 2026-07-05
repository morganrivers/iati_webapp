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
from collections import Counter, defaultdict
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

from split_constants import LATEST_TRAIN_POINT

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

def main() -> None:
    GOOD_CODES, BAD_CODES, TARGET_CODES = get_good_bad_and_target_codes()

    print(f"Loading embeddings from {EMB_PATH} ...")
    emb_map = load_embeddings_jsonl(EMB_PATH)
    print(f"Loaded embeddings: {len(emb_map)}")
    print(f"Loading raw JSONL (for click-to-print) from {EMB_PATH} ...")
    text_by_aid = load_jsonl_text_by_activity_id(EMB_PATH)
    print(f"Loaded raw JSONL lines: {len(text_by_aid)}")

    # dimension check
    dims = {v.shape[0] for v in emb_map.values()}
    if len(dims) != 1:
        print("ERROR: multiple embedding dimensions found:", sorted(dims))
        raise SystemExit(1)
    D = next(iter(dims))
    print(f"Embedding dimension: {D}")

    print(f"Loading and aggregating info from {INFO_CSV} ...")
    info = load_info_aggregated(INFO_CSV)
    print(f"Aggregated info rows: {len(info)}")

    # Restrict to activities that have embeddings
    aids = sorted(emb_map.keys())
    missing_info = [aid for aid in aids if aid not in info]
    if missing_info:
        print(f"WARNING: {len(missing_info)} activity_ids have embeddings but no info CSV row. Example: {missing_info[:5]}")
    # We'll still process them with defaults (GLOBAL + no sector + no decade)

    # Compute global category counts for tie-breaker (over activities with embeddings and info)
    rows_for_counts = {aid: info[aid] for aid in aids if aid in info}
    global_cat_counts = compute_global_category_counts(rows_for_counts, GOOD_CODES)
    if global_cat_counts:
        print("Global category counts (GOOD code assignments, used for tie-breaks):")
        total = sum(global_cat_counts.values())
        for cat, cnt in global_cat_counts.most_common():
            print(f"  {cat}: {cnt} ({(cnt/total*100 if total else 0):.1f}%)")
    else:
        print("WARNING: no global category counts computed (no GOOD codes found). Tie-breaker will be weak.")

    # Build per-activity metadata table
    records = []
    n_missing_start = 0
    n_global_bucket = 0
    n_country_location_present_but_unparsable = 0  # should never increment; we crash
    n_no_good_sector = 0
    n_tie_break_used = 0  # approximate; we'll compute by checking tied categories

    for i, aid in enumerate(aids, start=1):
        emb = l2_normalize(emb_map[aid].astype(np.float32))

        r = info.get(aid, None)
        dac_codes = set()
        recipient_iso3_fractions = None
        date_row = None

        if r is not None:
            dac_codes = set(r.get("dac5_codes", set()))
            recipient_iso3_fractions = r.get("recipient_iso3_fractions", None)
            # create Series for pick_start_date
            date_row = pd.Series({
                "actual_start_date": r.get("actual_start_date", pd.NaT),
                "original_planned_start_date": r.get("original_planned_start_date", pd.NaT),
                "txn_first_date": r.get("txn_first_date", pd.NaT),
            })
        else:
            date_row = pd.Series({"actual_start_date": pd.NaT, "original_planned_start_date": pd.NaT, "txn_first_date": pd.NaT})

        start_dt = pick_start_date(date_row)
        decade = None
        if pd.isna(start_dt):
            n_missing_start += 1
            decade = -1
            decade_label = "UNKNOWN"
        else:
            year = int(pd.Timestamp(start_dt).year)
            if year <= 2014:
                decade = 0
                decade_label = "le_2014"
            else:
                decade = 1
                decade_label = "ge_2015"


        # sector category
        sector = pick_sector_for_activity(dac_codes, GOOD_CODES, global_cat_counts)
        if sector is None:
            n_no_good_sector += 1
            sector = "Uncategorized"
            continue
        print("sector")
        print(sector)
        if sector == "Improving Energy Policy" or sector == "Clean Energy Generation":
            sector = "Energy"
        if sector == "Forestry & Sustainable Agriculture":
            sector = "General Environmental Protection"

        # multi-country parsing + GLOBAL bucket fallback
        parsed = parse_country_location(recipient_iso3_fractions)
        if parsed is None:
            countries = ["GLOBAL"]
            weights = [1.0]
            n_global_bucket += 1
        else:
            countries = [c for c, _ in parsed]
            weights = [float(w) for _, w in parsed]

        records.append({
            "activity_id": aid,
            "embedding": emb,          # numpy array
            "decade": decade,          # int or None
            "decade_label": decade_label,   # str
            "sector": sector,          # str
            "countries": countries,    # list[str]
            "weights": weights,        # list[float], sums to 1
            "start_date": start_dt,    # pd.Timestamp or NaT
        })

        if i % 5000 == 0:
            print(f"Prepared metadata for {i}/{len(aids)} activities...")
    # decade_centroid is built later from train_records only (after the train/val
    # cutoff is applied) so that the country-fallback path doesn't leak val/test
    # embeddings.

    print(f"Activities with missing start date (picked): {n_missing_start} / {len(aids)}")
    print(f"Activities assigned to GLOBAL bucket (no recipient_iso3_fractions): {n_global_bucket} / {len(aids)}")
    print(f"Activities with no GOOD sector codes (sector=Uncategorized): {n_no_good_sector} / {len(aids)}")
    bucket_counts = Counter(r["decade_label"] for r in records)
    print("Start-date buckets:")
    for k in ["le_2014", "ge_2015", "UNKNOWN"]:
        if k in bucket_counts:
            print(f"  {k}: {bucket_counts[k]}")

    # Identify records for leak-free fitting of centroids/PCA/UMAP.
    # INCLUDE_VAL_IN_FIT=True: fit on train+val (use when evaluating on the held-out test set).
    # INCLUDE_VAL_IN_FIT=False: fit on train only (default, for val-set evaluation).
    if INCLUDE_VAL_IN_FIT:
        active_cutoff = LATEST_TRAIN_VAL_POINT
        active_out_jsonl = OUT_JSONL_TRAINVAL
        active_model_save_path = MODEL_SAVE_PATH_TRAINVAL
    else:
        active_cutoff = LATEST_TRAIN_POINT
        active_out_jsonl = OUT_JSONL
        active_model_save_path = MODEL_SAVE_PATH
    cutoff = pd.Timestamp(active_cutoff)
    train_records = [r for r in records if pd.isna(r["start_date"]) or r["start_date"] <= cutoff]
    print(f"Fitting on {len(train_records)} train-period activities (cutoff {active_cutoff}), transforming {len(records)} total")

    # Build decade-only centroids from train records (fallback for country distance
    # when an activity has no usable country buckets). Decades not represented in
    # train fall back to the training-era decade (0).
    decade_to_train_vecs: Dict[int, List[np.ndarray]] = defaultdict(list)
    for rec in train_records:
        decade_to_train_vecs[rec["decade"]].append(rec["embedding"])
    decade_centroid: Dict[int, np.ndarray] = {}
    for dec, vecs in decade_to_train_vecs.items():
        decade_centroid[dec] = l2_normalize(np.mean(np.stack(vecs, axis=0), axis=0))
    needed_decades = {rec["decade"] for rec in records}
    for dec in needed_decades:
        if dec not in decade_centroid and 0 in decade_centroid:
            decade_centroid[dec] = decade_centroid[0]

    ##############################################################################
    # IMPROVEMENT: Temporal walk-forward UMAP fitting (more realistic evaluation)
    #
    # Currently: PCA+UMAP are fit on train_records only (start_date <= 2013-02-06).
    # All val/test activities are projected via transform(), which is noisier than
    # being "on the map" and causes a meaningful R² drop on the validation set.
    #
    # Better approach for evaluation: for each val activity X, refit PCA+UMAP on
    # (all train records + all other records with start_date < X.start_date + X itself).
    # Then retrain the downstream RF on those new UMAP coords (train set only, outcomes
    # known). This simulates deployment: at prediction time, all historically-started
    # activities are available to inform the manifold, but you still don't know X's label.
    #
    # Adding X itself to the UMAP fit is "transductive" (uses X's features, not its label)
    # — this is accepted and avoids the noisier transform() approximation entirely.
    #
    # The cost: re-run this script N_val times (one per validation activity, sorted by
    # start_date). Each run takes ~seconds. Alternatively, run once per unique year-bucket
    # in the val set to reduce the number of fits to ~5-10.
    #
    # NOTE: The R² gain from this is real signal, not leakage. UMAP never sees outcomes.
    ##############################################################################


    # -------------------- Build sector-decade centroids --------------------
    # Train-only fit: use train records for the same (sector, decade); if too few,
    # fall back to the (sector, decade=0) train centroid (the training-era decade)
    # rather than to all records, to avoid leaking val/test embeddings.
    _sd_train: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for rec in train_records:
        _sd_train[(rec["sector"], rec["decade"])].append(rec["embedding"])

    # Iterate over the union of (sector, decade) keys that any record will look up.
    needed_keys = {(rec["sector"], rec["decade"]) for rec in records}
    sector_decade_to_vecs: Dict[Tuple[str, int], List[np.ndarray]] = {}
    fallback_keys: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    n_sd_train_only = 0
    for key in needed_keys:
        sector, _decade = key
        if len(_sd_train.get(key, [])) >= MIN_SECTOR_DECADE_N:
            sector_decade_to_vecs[key] = _sd_train[key]
            n_sd_train_only += 1
        else:
            fallback_key = (sector, 0)  # training-era decade
            sector_decade_to_vecs[key] = _sd_train.get(fallback_key, [])
            fallback_keys.append((key, fallback_key))
    print(f"Sector-decade groups: {n_sd_train_only}/{len(sector_decade_to_vecs)} fitted on train-only, rest fall back to (sector, decade=0) train centroid")

    sector_decade_counts = {k: len(v) for k, v in sector_decade_to_vecs.items()}
    small_sd = [(k, n) for k, n in sector_decade_counts.items() if n < MIN_SECTOR_DECADE_N]
    if small_sd:
        print("\nERROR: small sector-decade groups detected after train-only fallback (will crash so you can decide):")
        small_sd.sort(key=lambda x: x[1])
        for (sector, decade), n in small_sd[:50]:
            print(f"  (sector={sector!r}, decade={decade}) n_train={n}")
        raise SystemExit(1)

    sector_decade_centroid: Dict[Tuple[str, int], np.ndarray] = {}
    for k, vecs in sector_decade_to_vecs.items():
        m = np.mean(np.stack(vecs, axis=0), axis=0)
        m = l2_normalize(m)
        sector_decade_centroid[k] = m

    print(f"Built sector-decade centroids: {len(sector_decade_centroid)}")

    # -------------------- Build country-decade centroids (weighted) --------------------
    # Use train-only data where sufficient; fall back to all records for groups
    # lacking train data (e.g. decade=1 groups, since train cutoff < 2015).
    def _accumulate_country_vecs(recs):
        w_sum = defaultdict(float)
        act_cnt = defaultdict(int)
        sumvec = {}
        for rec in recs:
            e = rec["embedding"]
            decade = rec["decade"]
            for c, w in zip(rec["countries"], rec["weights"]):
                key = (c, decade)
                w_sum[key] += float(w)
                act_cnt[key] += 1
                if key not in sumvec:
                    sumvec[key] = (float(w) * e).astype(np.float32)
                else:
                    sumvec[key] += (float(w) * e).astype(np.float32)
        return w_sum, act_cnt, sumvec

    _tr_w_sum, _tr_act_cnt, _tr_sumvec = _accumulate_country_vecs(train_records)

    # Train-only fit: for each (country, decade) key that any record needs, use
    # train data for that key when sufficient; otherwise fall back to the
    # (country, decade=0) train centroid (training-era decade). Never use
    # val/test embeddings.
    needed_country_keys: set = set()
    for rec in records:
        for c in rec["countries"]:
            needed_country_keys.add((c, rec["decade"]))

    country_decade_weight_sum: dict = {}
    country_decade_activity_count: dict = {}
    country_decade_sumvec: dict = {}
    n_cd_train_only = 0
    for key in needed_country_keys:
        country, _decade = key
        if (_tr_act_cnt.get(key, 0) >= MIN_COUNTRY_DECADE_N and
                _tr_w_sum.get(key, 0.0) >= MIN_COUNTRY_DECADE_WEIGHT):
            src_key = key
            n_cd_train_only += 1
        else:
            src_key = (country, 0)  # training-era decade
        country_decade_weight_sum[key] = _tr_w_sum.get(src_key, 0.0)
        country_decade_activity_count[key] = _tr_act_cnt.get(src_key, 0)
        if src_key in _tr_sumvec:
            country_decade_sumvec[key] = _tr_sumvec[src_key]
    print(f"Country-decade groups: {n_cd_train_only}/{len(needed_country_keys)} fitted on train-only, rest fall back to (country, decade=0) train centroid")

    # Validate small groups and crash
    small_cd = []
    for key in country_decade_sumvec.keys():
        n = country_decade_activity_count[key]
        ws = country_decade_weight_sum[key]
        if n < MIN_COUNTRY_DECADE_N or ws < MIN_COUNTRY_DECADE_WEIGHT:
            small_cd.append((key, n, ws))

    if small_cd:
        print("\nERROR: small country-decade groups detected (will crash so you can decide):")
        small_cd.sort(key=lambda x: (x[1], x[2]))
        for (c, decade), n, ws in small_cd[:80]:
            print(f"  (country={c!r}, decade={decade}) n_activities={n} weight_sum={ws:.3f}")
        print("continuing...")
        # raise SystemExit(1)
    small_keys = set()

    if small_cd:
        print("\nWARNING: small country-decade groups detected (will be DROPPED from country mix):")
        small_cd.sort(key=lambda x: (x[1], x[2]))
        for (c, decade), n, ws in small_cd[:80]:
            print(f"  (country={c!r}, decade={decade}) n_activities={n} weight_sum={ws:.3f}")
        small_keys = {key for (key, _, _) in small_cd}

    country_decade_centroid: Dict[Tuple[str, int], np.ndarray] = {}
    for key, sumvec in country_decade_sumvec.items():
        if key in small_keys:
            continue
        ws = country_decade_weight_sum[key]
        m = sumvec / float(ws)
        m = l2_normalize(m)
        country_decade_centroid[key] = m

    print(f"Built country-decade centroids: {len(country_decade_centroid)}")

    for rec in records:
        dec = rec["decade"]
        kept_c = []
        kept_w = []
        dropped = 0.0

        for c, w in zip(rec["countries"], rec["weights"]):
            if (c, dec) in country_decade_centroid:
                kept_c.append(c)
                kept_w.append(float(w))
            else:
                dropped += float(w)

        tot = sum(kept_w)
        if tot > 0:
            kept_w = [w / tot for w in kept_w]  # renormalize
            rec["countries"] = kept_c
            rec["weights"] = kept_w
        else:
            # nothing usable left; we’ll use decade fallback in the distance computation
            rec["countries"] = []
            rec["weights"] = []

        rec["dropped_country_weight"] = dropped

    # -------------------- Distances per activity --------------------
    sector_dist = np.empty(len(records), dtype=np.float32)
    country_dist = np.empty(len(records), dtype=np.float32)

    for idx, rec in enumerate(records):
        e = rec["embedding"]
        decade = rec["decade"]

        s_key = (rec["sector"], decade)
        s_cent = sector_decade_centroid.get(s_key)
        if s_cent is None:
            print("ERROR: missing sector centroid for", s_key, "activity", rec["activity_id"])
            raise SystemExit(1)
        sector_dist[idx] = euclidean_distance(e, s_cent)

        if not rec["countries"]:
            mix = decade_centroid[decade]
        else:
            mix = np.zeros_like(e, dtype=np.float32)
            for c, w in zip(rec["countries"], rec["weights"]):
                mix += float(w) * country_decade_centroid[(c, decade)].astype(np.float32)
            mix = l2_normalize(mix)

        country_dist[idx] = euclidean_distance(e, mix)


        if (idx + 1) % 5000 == 0:
            print(f"Computed distances for {idx+1}/{len(records)} activities...")

    # -------------------- UMAP maps (2D/3D/4D) --------------------
    X_all = np.stack([rec["embedding"] for rec in records], axis=0)
    X_train_emb = np.stack([rec["embedding"] for rec in train_records], axis=0)
    from sklearn.decomposition import PCA
    import umap

    print(f"Running PCA -> {PCA_DIMS} dims (fit on {len(train_records)} train, transform {len(records)} total)...")
    pca = PCA(n_components=min(PCA_DIMS, X_train_emb.shape[1]), random_state=UMAP_SEED)
    pca.fit(X_train_emb)
    X_train_pca = pca.transform(X_train_emb)
    X_all_pca = pca.transform(X_all)
    evr = float(np.sum(pca.explained_variance_ratio_))
    print(f"PCA explained variance ratio sum: {evr:.4f}")

    if UMAP_ENSEMBLE:
        # ---- UMAP ensemble helpers ----
        UMAP_SEEDS = [42, 7, 13, 99, 123]
        UMAP_N_NEIGHBORS = 20  # increased from 15 for smoother, more stable manifold

        def _procrustes_rotation(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
            """Return rotation matrix R (k×k) s.t. (target @ R.T) minimises ||reference - target @ R.T||_F.
            Orthogonal Procrustes via SVD — rotation/reflection only, no scaling."""
            M = reference.T @ target
            U, _, Vt = np.linalg.svd(M)
            return U @ Vt

        def _fit_umap_ensemble(n_components: int):
            """Fit UMAP with each seed in UMAP_SEEDS, Procrustes-align all runs to seed-0, return:
              - U_all:          (N_all,  k) mean-aligned embedding for all activities  → JSONL output
              - train_ensemble: list of N (N_train, k) aligned training embeddings    → pkl / inference
            Raw_data (X_train_pca) is shared across seeds; only the embedding differs.
            """
            raw_train_list = []
            raw_all_list = []
            for seed in UMAP_SEEDS:
                r = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=UMAP_N_NEIGHBORS,
                    min_dist=0.1,
                    metric="euclidean",
                    random_state=seed,
                )
                r.fit(X_train_pca)
                raw_train_list.append(r.embedding_.astype(np.float64))
                raw_all_list.append(r.transform(X_all_pca).astype(np.float64))

            # Compute Procrustes rotation matrices from training-set embeddings
            rotations = [np.eye(n_components)]
            for i in range(1, len(UMAP_SEEDS)):
                rotations.append(_procrustes_rotation(raw_train_list[0], raw_train_list[i]))

            # Apply the same rotations to both training and all-activity embeddings
            aligned_train = [raw_train_list[0]] + [
                raw_train_list[i] @ rotations[i].T for i in range(1, len(UMAP_SEEDS))
            ]
            aligned_all = [raw_all_list[0]] + [
                raw_all_list[i] @ rotations[i].T for i in range(1, len(UMAP_SEEDS))
            ]

            U_all = np.mean(aligned_all, axis=0).astype(np.float32)
            train_ensemble = [emb.astype(np.float32) for emb in aligned_train]
            return U_all, train_ensemble

        print(f"Running UMAP 2D ensemble ({len(UMAP_SEEDS)} seeds, n_neighbors={UMAP_N_NEIGHBORS})...")
        U2, _ = _fit_umap_ensemble(2)

        print(f"Running UMAP 3D ensemble ({len(UMAP_SEEDS)} seeds, n_neighbors={UMAP_N_NEIGHBORS})...")
        U3, umap3_train_ensemble = _fit_umap_ensemble(3)

        print(f"Running UMAP 4D ensemble ({len(UMAP_SEEDS)} seeds, n_neighbors={UMAP_N_NEIGHBORS})...")
        U4, _ = _fit_umap_ensemble(4)

        umap3_n_neighbors = UMAP_N_NEIGHBORS
        umap3_embedding = umap3_train_ensemble[0]
    else:
        UMAP_N_NEIGHBORS = 15

        def _fit_single_umap(n_components: int):
            r = umap.UMAP(
                n_components=n_components,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=0.1,
                metric="euclidean",
                random_state=UMAP_SEED,
            )
            r.fit(X_train_pca)
            return r.transform(X_all_pca).astype(np.float32), r

        print("Running UMAP 2D ...")
        U2, _ = _fit_single_umap(2)
        print("Running UMAP 3D ...")
        U3, reducer_3d = _fit_single_umap(3)
        print("Running UMAP 4D ...")
        U4, _ = _fit_single_umap(4)

        umap3_train_ensemble = None
        umap3_n_neighbors = UMAP_N_NEIGHBORS
        umap3_embedding = U3  # all-activity transform (original behaviour)

    # -------------------- SAVE MODELS & CENTROIDS --------------------
    print("Saving fitted models and centroids...")

    # Store only serializable arrays — not the umap objects themselves —
    # so the webapp can load the pkl without importing umap-learn.
    models_to_save = {
        'pca': pca,
        'umap3_raw_data':    X_train_pca.astype(np.float32),
        'umap3_embedding':   umap3_embedding,
        'umap3_n_neighbors': umap3_n_neighbors,
        'sector_decade_centroid': sector_decade_centroid,
        'country_decade_centroid': country_decade_centroid,
        'decade_centroid': decade_centroid,
        'global_cat_counts': global_cat_counts,
        'GOOD_CODES': GOOD_CODES,
    }
    if umap3_train_ensemble is not None:
        models_to_save['umap3_ensemble'] = umap3_train_ensemble

    with active_model_save_path.open('wb') as f:
        pickle.dump(models_to_save, f)

    print(f"Saved models to {active_model_save_path}")

    # -------------------- Write derived JSONL --------------------
    print(f"Writing {active_out_jsonl} ...")
    with active_out_jsonl.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            out = {
                "activity_id": rec["activity_id"],
                "decade": int(rec["decade"]),
                "decade_label": rec["decade_label"],       # keep the human-readable bucket
                "sector": rec["sector"],
                "countries": rec["countries"],
                "country_weights": [float(w) for w in rec["weights"]],
                "sector_distance": float(sector_dist[idx]),
                "dropped_country_weight": float(rec.get("dropped_country_weight", 0.0)),
                "country_distance": float(country_dist[idx]),
                "umap_2d": [float(U2[idx, 0]), float(U2[idx, 1])],
                "umap_3d": [float(x) for x in U3[idx, :].tolist()],
                "umap_4d": [float(x) for x in U4[idx, :].tolist()],
            }
            f.write(json.dumps(out) + "\n")

    print(f"Wrote: {active_out_jsonl}")

    import matplotlib.colors as mcolors

    print(f"Plotting 2D UMAP by sector -> {OUT_PLOT} ...")
    sectors = [rec["sector"] for rec in records]
    uniq = sorted(set(sectors))
    print(f"Unique sectors in plot: {len(uniq)}")
    sector_to_idx = {s: i for i, s in enumerate(uniq)}

    cvals = np.array([sector_to_idx[s] for s in sectors], dtype=np.int32)

    import matplotlib.pyplot as plt
    base = list(plt.get_cmap("tab20").colors)
    if len(uniq) <= 20:
        colors = base[:len(uniq)]
    else:
        colors = [base[i % 20] for i in range(len(uniq))]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(uniq) + 0.5, 1), cmap.N)

    plt.figure(figsize=(14, 10))
    plt.scatter(U2[:, 0], U2[:, 1], c=cvals, s=8, alpha=0.8, cmap=cmap, norm=norm)
    plt.title("UMAP 2D of Targets Embeddings (colored by sector)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    handles = [
        plt.Line2D([], [], linestyle="", marker="o", markersize=6,
                   markerfacecolor=colors[sector_to_idx[s]], markeredgecolor="none")
        for s in uniq
    ]
    plt.legend(handles, uniq, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    plt.close()



    # import plotly.express as px

    # # Build hover text / general info
    # def fmt_countries(rec, k=5):
    #     parts = [f"{c}:{w:.3f}" for c, w in zip(rec["countries"], rec["weights"])]
    #     if len(parts) > k:
    #         parts = parts[:k] + ["..."]
    #     return " | ".join(parts)

    # dfp = pd.DataFrame({
    #     "activity_id": [r["activity_id"] for r in records],
    #     "sector": [r["sector"] for r in records],
    #     "decade": [r["decade"] for r in records],
    #     "decade_label": [r.get("decade_label", str(r["decade"])) for r in records],
    #     "sector_distance": sector_dist.astype(float),
    #     "country_distance": country_dist.astype(float),
    #     "dropped_country_weight": [float(r.get("dropped_country_weight", 0.0)) for r in records],
    #     "countries": [fmt_countries(r) for r in records],
    #     "umap1": U2[:, 0].astype(float),
    #     "umap2": U2[:, 1].astype(float),
    # })

    # fig = px.scatter(
    #     dfp,
    #     x="umap1",
    #     y="umap2",
    #     color="sector",
    #     hover_data={
    #         "activity_id": True,
    #         "sector": True,
    #         "decade_label": True,
    #         "sector_distance": ":.4f",
    #         "country_distance": ":.4f",
    #         "dropped_country_weight": ":.4f",
    #         "countries": True,
    #         "umap1": False,
    #         "umap2": False,
    #         "decade": False,
    #     },
    #     title="UMAP 2D of Targets Embeddings (hover for details)",
    # )

    # fig.update_traces(marker=dict(size=6, opacity=0.8))
    # fig.write_html(OUT_PLOT_HTML, include_plotlyjs="cdn")
    # print(f"Saved interactive plot: {OUT_PLOT_HTML}")


    import matplotlib.colors as mcolors

    print("Interactive UMAP: click a point to print details to terminal...")

    # --- color mapping (discrete, legend matches) ---
    sectors = [rec["sector"] for rec in records]
    uniq = sorted(set(sectors))
    sector_to_idx = {s: i for i, s in enumerate(uniq)}
    cvals = np.array([sector_to_idx[s] for s in sectors], dtype=np.int32)
    import matplotlib.pyplot as plt
    base = list(plt.get_cmap("tab20").colors)
    colors = base[:len(uniq)] if len(uniq) <= 20 else [base[i % 20] for i in range(len(uniq))]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(uniq) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(14, 10))
    sc = ax.scatter(
        U2[:, 0], U2[:, 1],
        c=cvals, s=12, alpha=0.8,
        cmap=cmap, norm=norm,
        picker=True,   # <-- makes points clickable
    )

    ax.set_title("UMAP 2D of Targets Embeddings (click a point to print)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    handles = [
        plt.Line2D([], [], linestyle="", marker="o", markersize=6,
                   markerfacecolor=colors[sector_to_idx[s]], markeredgecolor="none")
        for s in uniq
    ]
    ax.legend(handles, uniq, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    def _fmt_countries(rec, k=8):
        parts = [f"{c}:{w:.3f}" for c, w in zip(rec["countries"], rec["weights"])]
        return " | ".join(parts[:k]) + (" | ..." if len(parts) > k else "")

    def on_pick(event):
        if event.artist != sc:
            return
        i = int(event.ind[0])  # first point if multiple are within click radius
        rec = records[i]
        if(rec.get('decade_label', rec['decade']) == "ge_2015"):
            print("hidden.")
        else:
            i = int(event.ind[0])
            rec = records[i]
            aid = rec["activity_id"]

            t = text_by_aid.get(aid, {})
            text = t.get("text")
            section = t.get("section")
            model = t.get("model")

            print("\n--- CLICK ---")
            print(f"activity_id: {aid}")
            if section is not None:
                print(f"section: {section}")
            if model is not None:
                print(f"model: {model}")
            print(f"sector: {rec['sector']}")
            print(f"decade: {rec.get('decade_label', rec['decade'])} ({rec['decade']})")
            print(f"sector_distance: {float(sector_dist[i]):.6f}")
            print(f"country_distance: {float(country_dist[i]):.6f}")
            print(f"dropped_country_weight: {float(rec.get('dropped_country_weight', 0.0)):.6f}")
            print("text:")
            print(text if isinstance(text, str) else "<NO TEXT FIELD>")

            print(
                f"countries: {_fmt_countries(rec)}\n"
                f"umap2: ({float(U2[i,0]):.4f}, {float(U2[i,1]):.4f})\n"
            )

    fig.canvas.mpl_connect("pick_event", on_pick)

    plt.tight_layout()

    # optional: still save the static png
    plt.savefig(OUT_PLOT, dpi=200)
    print(f"Saved png: {OUT_PLOT}")

    plt.show()

if __name__ == "__main__":
    main()
