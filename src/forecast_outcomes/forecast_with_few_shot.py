#!/usr/bin/env python3
from pprint import pprint
import numpy as np 
import random
import pandas as pd 
import pprint
import json
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import hashlib


# LIST_OF_ALL_RECENT_VARIANTS = ("short_well_or_badly","short_well_or_badly_add_grades", "exactly_like_halawi_et_al","choose_your_own_adventure","consider_the_knn", "adjust_based_on_random_forest")
LIST_OF_ALL_RECENT_VARIANTS = ("short_well_or_badly", "exactly_like_halawi_et_al","choose_your_own_adventure","consider_the_knn", "adjust_based_on_random_forest","generate_rag_queries","short_well_or_badly_rag_added", "exactly_like_halawi_et_al_rag_added", "summarize_knn", "exactly_like_halawi_et_al_rag_added_no_knn_no_rag", "exactly_like_halawi_et_al_rag_added_forced_rf", "exactly_like_halawi_et_al_rag_added_no_knn_no_rag_forced_rf", "exactly_like_halawi_et_al_rag_added_forced_rf_with_explanation", "exactly_like_halawi_et_al_better_model_rag_added_forced_rf")

# ---------------------------------------------------------------------
# Imports from your utils
# ---------------------------------------------------------------------

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
 
from get_all_pages_within_category import load_and_filter_rows
from extracting_and_grading_helper_functions import (
    consolidate_rows_by_activity,
    loop_over_rows_to_call_model,
    load_generic_jsonl_and_put_into_bundles,
)
from extract_pdfs_as_txt import (
    normalized_basename,
    process_one,  # we pass our own output directory
)
# This module name is assumed; rename if needed to wherever get_ratings_text lives
from helpers_for_ratings_and_final_activity_features import get_ratings_text, get_rating_scale_info_from_rating_object, get_rating_scale_info, get_text_to_describe_rating_distribution, pick_start_date, compute_training_distribution_by_prefix, load_good_overall_ids
# import C_run_GLM as stats

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

DATA_DIR = Path("../../data")
ACTIVITY_INFO_CSV = DATA_DIR / "info_for_activity_forecasting.csv"
MERGED_OVERALL_RATINGS = DATA_DIR / "merged_overall_ratings.jsonl"
RETROSPECTIVE_FORECAST_JSONL = DATA_DIR / "outputs_retrospective_forecast.jsonl"
CHATGPT_SUMMARIES_JSONL = DATA_DIR / "outputs_summaries.jsonl"
RISKS_JSONL = DATA_DIR / "outputs_risks.jsonl"
INFO_FOR_ACTIVITY_FORECASTING = '../../data/info_for_activity_forecasting_with_cpia_imputed.csv'
OUT_MISC = Path("../../data/outputs_misc.jsonl")


# Where we store PDF→txt outputs (new folder)
TXT_OUTPUT_DIR = DATA_DIR / "iati_all_pdfs_txt_format"

FEW_SHOT_KS = [1, 7, 20]
MAX_TARGET_CHARS = 999999999 #6000
MAX_NEIGHBOR_CHARS = 999999999 #1500

SIMILARITY_TOP_N = 1000  # how many candidates before we pick K neighbors
# MODEL_NAME = "chatgpt"  # your gpt-3.5 wrapper
MODEL_NAME = "gemini"  # your gpt-3.5 wrapper



import ast
import json
import re


def format_risks_if_listlike(risks_summary: str) -> str:
    """
    If risks_summary looks like a serialized list of strings and parses cleanly,
    return a human-readable bullet list. Otherwise return the original string.
    """
    if risks_summary is None:
        return risks_summary

    t = str(risks_summary).strip()
    if not t or t == "NO RESPONSE":
        return risks_summary

    # "clearly parsable format" gate: must look like a bracketed list
    if not (t.startswith("[") and t.endswith("]")):
        return risks_summary

    obj = None
    # Try JSON first (double quotes)
    try:
        obj = json.loads(t)
    except Exception:
        pass

    # Then try Python literal (single quotes etc.)
    if obj is None:
        try:
            obj = ast.literal_eval(t)
        except Exception:
            return risks_summary

    if not isinstance(obj, (list, tuple)):
        return risks_summary

    if not all(isinstance(x, str) for x in obj):
        return risks_summary

    items = [re.sub(r"\s+", " ", x).strip() for x in obj]
    items = [x for x in items if x]
    if len(items) == 0:
        return risks_summary

    return "\n".join(f"- {x}" for x in items)


# def load_ml_model_preds_for_prompts(path="../../data/ridge_plus_rf_predictions.csv", col="pred_rf"):
def load_ml_model_preds_for_prompts(path="../../data/best_model_predictions.csv", col="pred_rf_llm_modded"):
    df = pd.read_csv(path, dtype={"activity_id": str})
    # if you saved with index=activity_id, pandas will read it as a normal column unless you used index_col
    if "activity_id" not in df.columns:
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str)
        return df[col].astype(float).to_dict()

    return pd.Series(df[col].astype(float).values, index=df["activity_id"].astype(str)).to_dict()


def load_stat_model_interpretations(path="../../data/stat_model_interpretations.csv"):
    df = pd.read_csv(path, dtype={"activity_id": str})
    return dict(zip(df["activity_id"].astype(str), df["interpretation"].astype(str)))



def rf_pred_label_and_number(v: float) -> tuple[str, str]:
    """
    Returns (label, parenthetical detail).
    label is the nearest discrete class label.
    detail is '(pred=..., closer/midway ...)'.
    """
    RATING_MAP = {
        0: "Highly Unsatisfactory",
        1: "Unsatisfactory",
        2: "Moderately Unsatisfactory",
        3: "Moderately Satisfactory",
        4: "Satisfactory",
        5: "Highly Satisfactory",
    }

    x = float(v)
    x = min(5.0, max(0.0, x))

    k = int(np.floor(x))
    frac = x - k

    lower = RATING_MAP[k]
    upper = RATING_MAP[min(5, k + 1)]
    nearest = RATING_MAP[int(np.round(x))]

    if k == 5:
        pos = "at the top category"
    elif frac < 0.25:
        pos = f"closer to {lower} than {upper}"
    elif frac <= 0.75:
        pos = f"midway between {lower} and {upper}"
    else:
        pos = f"closer to {upper} than {lower}"

    return nearest, f"(pred={x:.2f}; {pos})"



# ---------------------------------------------------------------------
# Basic loaders
# ---------------------------------------------------------------------
def _load_risks_summaries() -> Dict[str, str]:
    """
    activity_id -> risks_summary
    Pulled from outputs_risks.jsonl (same style as other JSONLs).
    """
    out: Dict[str, str] = {}
    if not RISKS_JSONL.exists():
        return out

    with RISKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            aid = (data.get("activity_id") or "").strip()
            if not aid:
                continue

            # Prefer a 'response' dict if present
            resp = data.get("response")
            text = ""
            if isinstance(resp, dict):
                text = resp.get("content") or resp.get("text") or ""

            # Fall back to response_text
            if not text:
                raw = (data.get("response_text") or "").strip()
                if raw:
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = raw

                    if isinstance(parsed, dict):
                        # If risks are nested in a dict, adjust this to the right key
                        text = parsed.get("risks_summary") or json.dumps(parsed)
                    else:
                        text = str(parsed)

            text = str(text).strip()
            if not text:
                continue

            out[aid] = text

    return out

def _load_chatgpt_descriptions() -> Dict[str, str]:
    """
    activity_id -> chatgpt_description (one-line or paragraph summary)
    Pulled from outputs_summaries.jsonl.
    """
    out: Dict[str, str] = {}
    if not CHATGPT_SUMMARIES_JSONL.exists():
        return out

    with CHATGPT_SUMMARIES_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            aid = (data.get("activity_id") or "").strip()
            if not aid:
                continue

            # Try to mirror your other JSONL structure:
            # 1) if there's a 'response' dict, use its content/text
            # 2) otherwise fallback to 'response_text'
            desc = ""
            resp = data.get("response")
            if isinstance(resp, dict):
                desc = resp.get("content") or resp.get("text") or ""
            if not desc:
                desc = data.get("response_text") or ""

            desc = str(desc).strip()
            if not desc:
                continue

            out[aid] = desc

    return out

def _load_activity_info() -> Dict[str, Dict[str, str]]:
    """activity_id -> row dict from info_for_activity_forecasting.csv."""
    out: Dict[str, Dict[str, str]] = {}
    import csv
    with ACTIVITY_INFO_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            aid = (r.get("activity_id") or "").strip()
            if aid:
                out[aid] = r
    # merge ChatGPT descriptions into this map
    chatgpt_descriptions = _load_chatgpt_descriptions()
    for aid, desc in chatgpt_descriptions.items():
        row = out.setdefault(aid, {})
        row["chatgpt_description"] = desc

    # merge risk summaries into this map
    risks_summaries = _load_risks_summaries()
    for aid, risks_text in risks_summaries.items():
        row = out.setdefault(aid, {})
        row["risks_summary"] = risks_text

    return out


# def _load_outcome_ratings() -> Dict[str, Dict[str, Any]]:
#     """
#     activity_id -> {"rating_value": ..., "min": ..., "max": ...}
#     matching your existing merged_overall_ratings.jsonl structure.
#     """
#     ratings: Dict[str, Dict[str, Any]] = {}
#     with MERGED_OVERALL_RATINGS.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             data = json.loads(line)
#             activity_id = data.get("activity_id")
#             if not activity_id:
#                 continue
#             response_text = data.get("response_text", "{}")
#             try:
#                 response_data = json.loads(response_text)
#             except Exception:
#                 continue
#             rating_value = response_data.get("rating_value")
#             if rating_value is None:
#                 continue
#             rating_min = response_data.get("min")
#             rating_max = response_data.get("max")
#             ratings[activity_id] = {
#                 "rating_value": rating_value,
#                 "min": rating_min,
#                 "max": rating_max,
#             }
#     return ratings


def load_mock_forecasts(path: Path) -> Dict[str, str]:
    """
    activity_id -> full mock forecast text
    Assumes loop_over_rows_to_call_model wrote a "response" dict with "content",
    or falls back to "response_text".
    """
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            aid = data.get("activity_id")
            if not aid:
                continue
            resp = data.get("response")
            content = ""
            if isinstance(resp, dict):
                content = resp.get("content", "") or resp.get("text", "")
            if not content:
                content = data.get("response_text", "")
            if not content:
                continue
            out[aid] = str(content)
    return out


# ---------------------------------------------------------------------
# Text extraction for activities (from PDFs via your converter)
# ---------------------------------------------------------------------

def ensure_txt_for_pdf(pdf_path: Path, allow_ocr: bool = True) -> Path:
    """
    Make sure there is a .txt file for this PDF in TXT_OUTPUT_DIR.
    Uses your process_one() helper from extract_iati_texts.
    """
    TXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = normalized_basename(pdf_path)
    txt_path = TXT_OUTPUT_DIR / f"{base}.txt"
    print("txt_path, exists?")
    print(txt_path, txt_path.exists())
    if not txt_path.exists():
        status, msg = process_one(pdf_path, TXT_OUTPUT_DIR, allow_ocr, only_newer=False)
        print(msg)
        if status != "OK":
            print(f"Warning: conversion for {pdf_path} returned status={status}")
    return txt_path



# def get_knn_neighbors(
#     target_aid,
#     ratings,
#     mock_forecasts,
#     top_n,
#     k,
#     rating_stats,
#     variant: str = "",
#     pool_size: int = 20,   # try 10 first; 20 if you want more variety
# ) -> List[str]:
#     pprint.pprint("target_aid")
#     pprint.pprint(target_aid)
#     pprint.pprint("ratings")
#     print(len(ratings))
#     pprint.pprint("mock_forecasts")
#     print(len(mock_forecasts))
#     pprint.pprint("top_n")
#     pprint.pprint(top_n)
#     pprint.pprint("k")
#     pprint.pprint(k)
#     print("rating_stats")
#     print(len(rating_stats))
#     pprint.pprint("variant")
#     pprint.pprint(variant)
#     pprint.pprint("pool_size")
#     pprint.pprint(pool_size)
#     """
#     Return up to k neighbor activity_ids for target_aid that:
#       * are in ratings
#       * have a valid rating scale via get_ratings_text
#       * have baseline bundles (in baseline_map)

#     Neighbor selection logic:

#       1. Build a similarity-ordered candidate list from df_sim (up to top_n).
#       2. For each candidate, use rating_stats["aid_fraction"][aid] in [0,1] to
#          put it into one of 6 equal-width bins along the scale (worst→best).
#          (bin 1=worst, 6=best).
#       3. Use rating_stats["six_percents"] to compute target counts per bin
#          so that the k neighbors roughly follow the same distribution as the
#          full training set.
#       4. Walk candidates in similarity order, selecting neighbors to satisfy
#          those per-bin target counts.
#       5. If not enough neighbors have been chosen (due to missing bins, etc.),
#          fill the remaining slots with the most similar unused candidates.
#       6. Try to enforce at least one neighbor above and one below the midpoint
#          (fraction 0.5), if possible.
#     """

#     def six_bin(f: float) -> int:
#         """Map fraction f in [0,1] to a bin 1..6 (worst→best)."""
#         f = float(f)
#         if f < 0.0:
#             f = 0.0
#         elif f > 1.0:
#             f = 1.0
#         idx = min(5, int(f * 6))  # 0..5
#         return idx + 1            # 1..6

#     target_aid_str = str(target_aid)
#     # If rating_stats was computed "by_prefix", select the best matching prefix
#     if isinstance(rating_stats, dict) and "aid_fraction" not in rating_stats:
#         byp = rating_stats.get("by_prefix")
#         if isinstance(byp, dict):
#             keys = [k for k in byp.keys() if target_aid_str.startswith(k)]
#             if keys:
#                 rating_stats = byp[max(keys, key=len)]
#             else:
#                 rating_stats = None

#     # Deterministic per (variant, target_aid), but different across variants
#     seed_material = f"{variant}|{target_aid_str}".encode("utf-8")
#     seed_int = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "big")
#     rng = random.Random(seed_int)

#     # allowed_ids: Set[str] = {
#     #     str(aid)
#     #     for aid in ratings.keys()
#     #     if str(aid) != target_aid_str and str(aid) in baseline_map
#     # }
#     # all activities that have BOTH a rating and a mock forecast,
#     # excluding the target itself
#     allowed_ids: Set[str] = {
#         str(aid)
#         for aid in mock_forecasts.keys()
#         if str(aid) != target_aid_str and str(aid) in ratings
#     }

#     print("allowed_ids len")
#     print(len(allowed_ids))

#     if not allowed_ids:
#         print("no allowed ids!")
#         return []

#     results = find_similar_activities_semantic(
#         target_aid_str,
#         csv_path=str(ACTIVITY_INFO_CSV),
#         top_n=top_n,
#         allowed_ids=allowed_ids,
#     )
#     if len(results) < 2:
#         print("ERROR: seems nothing was found in similarity search..")
#         return []

#     df_sim, _search_item = results
#     print("len results")
#     print(len(df_sim))

#     # pprint.pprint("rating_stats")
#     # pprint.pprint(rating_stats)
#     # Fallback: no rating_stats, just take top-k similar with valid scales
#     # if not rating_stats or "aid_fraction" not in rating_stats:
#     #     neighbor_ids: List[str] = []
#     #     for _, row in df_sim.iterrows():
#     #         aid = str(row["activity_id"])
#     #         if aid not in ratings:
#     #             continue
#     #         if get_rating_scale_info(aid, ratings) is None:
#     #             continue
#     #         # if aid not in baseline_map:
#     #         #     continue
#     #         neighbor_ids.append(aid)
#     #         if len(neighbor_ids) >= k:
#     #             break
#     #     print("NOT RATING STATS!")
#     #     print("len(neighbor_ids)")
#     #     print(len(neighbor_ids))
#     #     return neighbor_ids
#     if not rating_stats or "aid_fraction" not in rating_stats:
#         from collections import Counter
#         reasons = Counter()
#         neighbor_ids: List[str] = []
#         for _, row in df_sim.iterrows():
#             aid = str(row["activity_id"])
#             if aid not in ratings:
#                 reasons["no_rating"] += 1
#                 continue
#             if get_rating_scale_info(aid, ratings) is None:
#                 reasons["bad_scale"] += 1
#                 continue
#             neighbor_ids.append(aid)
#             if len(neighbor_ids) >= k:
#                 break
#         print("[DEBUG fallback] target:", target_aid_str, "reasons:", dict(reasons), "picked:", len(neighbor_ids))
#         return neighbor_ids


#     aid_fraction_map: Dict[str, float] = rating_stats.get("aid_fraction") or {}
#     global_six_percents: Dict[int, float] = rating_stats.get("six_percents") or {}

#     # Build candidate list (up to top_n, in similarity order), with frac + bin
#     candidates: List[Dict[str, Any]] = []
#     # for _, row in df_sim.iterrows():
#     #     aid = str(row["activity_id"])
#     #     if aid not in ratings:
#     #         continue
#     #     # if aid not in baseline_map:
#     #     #     continue
#     #     if get_rating_scale_info(aid, ratings) is None:
#     #         continue

#     #     frac = aid_fraction_map.get(aid)
#     #     if frac is None:
#     #         continue

#     #     cand_bin = six_bin(frac)
#     #     cand = {
#     #         "aid": aid,
#     #         "frac": frac,
#     #         "bin": cand_bin,
#     #         "similarity": float(row["similarity"]),
#     #     }
#     #     candidates.append(cand)
#     from collections import Counter

#     reasons = Counter()
#     candidates = []

#     for _, row in df_sim.iterrows():
#         aid = str(row["activity_id"])

#         if aid not in ratings:
#             reasons["no_rating"] += 1
#             continue

#         if get_rating_scale_info(aid, ratings) is None:
#             reasons["bad_scale"] += 1
#             continue

#         frac = aid_fraction_map.get(aid)
#         if frac is None:
#             reasons["no_frac"] += 1
#             continue

#         candidates.append({
#             "aid": aid,
#             "frac": frac,
#             "bin": six_bin(frac),
#             "similarity": float(row["similarity"]),
#         })

#     if not candidates:
#         print("[DEBUG] no candidates for target:", target_aid_str)
#         print("[DEBUG] filter reasons:", dict(reasons))
#         print("[DEBUG] df_sim rows:", len(df_sim))
#         return []

#     if not candidates:
#         print("ERROR: no candidates")
#         return []

#     total_k = min(k, len(candidates))
#     if total_k <= 0:
#         print("ERROR: negative K or negative candidates")
#         return []

#     # Candidates are currently in similarity order.
#     # For stochasticity: shuffle only within a top window, keep the tail ordered.
#     # If variant is empty, keep original deterministic behavior.
#     if variant:
#         m = min(pool_size, len(candidates))
#         head = candidates[:m].copy()
#         rng.shuffle(head)
#         candidates_iter = head + candidates[m:]
#     else:
#         candidates_iter = candidates

#     # Compute target counts per bin, proportional to global_six_percents
#     bin_ids = [1, 2, 3, 4, 5, 6]
#     base_counts: Dict[int, int] = {b: 0 for b in bin_ids}
#     fractional: List[Any] = []
#     remaining = total_k

#     for b in bin_ids:
#         pct = float(global_six_percents.get(b, 0.0))
#         exact = total_k * pct / 100.0
#         base = int(exact)
#         base_counts[b] = base
#         fractional.append((exact - base, b))
#         remaining -= base

#     # Distribute leftover slots by largest fractional part
#     fractional.sort(reverse=True)
#     i = 0
#     while remaining > 0 and i < len(fractional):
#         b = fractional[i][1]
#         base_counts[b] += 1
#         remaining -= 1
#         i += 1

#     selected: List[str] = []
#     selected_set: Set[str] = set()
#     selected_counts: Dict[int, int] = {b: 0 for b in bin_ids}

#     # First pass: respect per-bin target counts in similarity order
#     for cand in candidates_iter:
#         if len(selected) >= total_k:
#             break
#         aid = cand["aid"]
#         if aid in selected_set:
#             continue
#         b = cand["bin"]
#         if selected_counts[b] < base_counts.get(b, 0):
#             selected.append(aid)
#             selected_set.add(aid)
#             selected_counts[b] += 1

#     # Second pass: fill remaining slots with nearest unused candidates
#     if len(selected) < total_k:
#         for cand in candidates_iter:
#             if len(selected) >= total_k:
#                 break
#             aid = cand["aid"]
#             if aid in selected_set:
#                 continue
#             selected.append(aid)
#             selected_set.add(aid)

#     # Try to enforce at least one neighbor above and one below the midpoint
#     if aid_fraction_map and total_k >= 2:
#         def is_above(aid_str: str) -> bool:
#             f = aid_fraction_map.get(aid_str)
#             return f is not None and f > 0.5

#         def is_below(aid_str: str) -> bool:
#             f = aid_fraction_map.get(aid_str)
#             return f is not None and f < 0.5

#         has_above = any(is_above(a) for a in selected)
#         has_below = any(is_below(a) for a in selected)

#         if (not has_above or not has_below) and len(selected) >= 2:
#             # Build a quick index of candidates by aid for easy lookup
#             cand_by_aid = {c["aid"]: c for c in candidates}

#             # Try swapping from the end (least similar among chosen)
#             for i in range(len(selected) - 1, -1, -1):
#                 aid_to_replace = selected[i]

#                 if not has_above and is_below(aid_to_replace):
#                     replacement = None
#                     for cand in candidates:
#                         aid = cand["aid"]
#                         if aid in selected_set:
#                             continue
#                         if is_above(aid):
#                             replacement = aid
#                             break
#                     if replacement is not None:
#                         selected_set.remove(aid_to_replace)
#                         selected[i] = replacement
#                         selected_set.add(replacement)
#                         has_above = True
#                         break

#                 elif not has_below and is_above(aid_to_replace):
#                     replacement = None
#                     for cand in candidates:
#                         aid = cand["aid"]
#                         if aid in selected_set:
#                             continue
#                         if is_below(aid):
#                             replacement = aid
#                             break
#                     if replacement is not None:
#                         selected_set.remove(aid_to_replace)
#                         selected[i] = replacement
#                         selected_set.add(replacement)
#                         has_below = True
#                         break
#     if variant:
#         rng.shuffle(selected)
#     return selected
from typing import Dict, Any, List, Set

def get_knn_neighbors(
    target_aid,
    ratings,
    mock_forecasts,
    top_n,
    k,
    rating_stats,
    similarity_fn,
    variant=None
) -> List[str]:
    """
    Prefer same-prefix rating_stats (from compute_training_distribution_by_prefix),
    BUT if that prevents us from selecting:
      - at least one LOW (HU/U/MU),
      - one MID (MS),
      - one HIGH (S/HS),
    or prevents us from reaching k total neighbors,
    then expand to the global/overall fraction map and fill missing buckets / remaining slots.

    Selection rules:
      1) Build similarity-ranked df_sim (already done by find_similar_activities_semantic).
      2) Build candidate list from df_sim that:
           - are in ratings
           - have a valid rating scale via get_rating_scale_info
           - have a known fraction (prefix-first, global if expanded)
      3) Pick bucket exemplars in order LOW, MID, HIGH (most similar first).
      4) If any bucket missing OR selected < k -> expand to global and try again for missing buckets.
      5) Fill remaining slots up to k by most-similar unused candidates.

    Debug printouts explain:
      - how many df_sim rows we got
      - how many prefix vs global fraction entries exist
      - how many candidates we have under prefix-only and under global-expanded
      - which buckets were found / missing
      - whether we expanded
      - final selected ids
    """

    target_aid_str = str(target_aid)

    # All activities that have BOTH a rating and a mock forecast, excluding the target itself
    allowed_ids: Set[str] = {
        str(aid)
        for aid in mock_forecasts.keys()
        if str(aid) != target_aid_str and str(aid) in ratings
    }

    if not allowed_ids:
        print(f"[KNN] {target_aid_str}: no allowed ids (mock_forecasts∩ratings empty)")
        return []

    try:
        results = similarity_fn(
            target_aid_str,
            top_n=top_n,
            allowed_ids=allowed_ids,
        )
    except ValueError as e:
        if "not found in" in str(e):
            print(f"[KNN] {target_aid_str}: WARNING {e}; returning []")
            return []
        raise

    if len(results) < 2:
        print(f"[KNN] {target_aid_str}: ERROR similarity search returned <2 items; returning []")
        return []

    df_sim, _search_item = results
    print(f"[KNN] {target_aid_str}: df_sim rows={len(df_sim)} (top_n={top_n})")

    # ------------------------------------------------------------------
    # Resolve prefix stats + global stats from rating_stats
    # rating_stats can be either:
    #   - output of compute_training_distribution_from_scales (has aid_fraction)
    #   - output of compute_training_distribution_by_prefix (has overall/by_prefix)
    # ------------------------------------------------------------------

    primary_stats = None  # same-prefix stats (preferred)
    overall_stats = None  # global stats (fallback/expansion)

    if isinstance(rating_stats, dict) and "aid_fraction" in rating_stats:
        # Already "flat" stats
        primary_stats = rating_stats
        overall_stats = rating_stats
        print(f"[KNN] {target_aid_str}: rating_stats is flat (no by_prefix).")
    elif isinstance(rating_stats, dict):
        overall_stats = rating_stats.get("overall")
        byp = rating_stats.get("by_prefix")

        # choose best matching prefix if possible
        if isinstance(byp, dict):
            keys = [p for p in byp.keys() if target_aid_str.startswith(p)]
            chosen_prefix = max(keys, key=len) if keys else None
            primary_stats = (byp.get(chosen_prefix) if chosen_prefix else None)
            print(f"[KNN] {target_aid_str}: prefix match={chosen_prefix!r}")
        else:
            print(f"[KNN] {target_aid_str}: WARNING rating_stats has no usable by_prefix dict.")
            primary_stats = None

        if overall_stats is None:
            print(f"[KNN] {target_aid_str}: WARNING rating_stats missing 'overall'.")
    else:
        print(f"[KNN] {target_aid_str}: WARNING rating_stats not a dict.")
        primary_stats = None
        overall_stats = None

    # If no prefix-specific stats, fall back to overall stats (but keep bucket logic)
    if (not primary_stats or "aid_fraction" not in primary_stats) and overall_stats and "aid_fraction" in overall_stats:
        print(f"[KNN] {target_aid_str}: no prefix stats; using overall stats as primary")
        primary_stats = overall_stats

    # # If no usable stats at all, fallback to top-k most similar with valid scales
    # if not primary_stats or not isinstance(primary_stats, dict) or "aid_fraction" not in primary_stats:
    #     print(f"[KNN] {target_aid_str}: WARNING no usable prefix stats; fallback = top-k most similar")
    #     neighbor_ids: List[str] = []
    #     for _, row in df_sim.iterrows():
    #         aid = str(row["activity_id"])
    #         if aid not in ratings:
    #             continue
    #         if get_rating_scale_info(aid, ratings) is None:
    #             continue
    #         neighbor_ids.append(aid)
    #         if len(neighbor_ids) >= k:
    #             break
    #     print(f"[KNN] {target_aid_str}: fallback picked {len(neighbor_ids)}/{k}")
        # return neighbor_ids

    if not overall_stats or not isinstance(overall_stats, dict) or "aid_fraction" not in overall_stats:
        # We can still operate prefix-only; expansion just won't be possible.
        print(f"[KNN] {target_aid_str}: WARNING no usable overall stats; expansion disabled.")
        overall_stats = None

    primary_aid_fraction: Dict[str, float] = {
        str(a): float(f) for a, f in (primary_stats.get("aid_fraction") or {}).items()
    }
    overall_aid_fraction: Dict[str, float] = {}
    if overall_stats:
        overall_aid_fraction = {
            str(a): float(f) for a, f in (overall_stats.get("aid_fraction") or {}).items()
        }

    print(f"[KNN] {target_aid_str}: primary_aid_fraction size={len(primary_aid_fraction)}")
    if overall_stats:
        print(f"[KNN] {target_aid_str}: overall_aid_fraction size={len(overall_aid_fraction)}")

    # ------------------------------------------------------------------
    # Helpers: bucket logic based on fraction -> int 0..5
    # 0=HU,1=U,2=MU,3=MS,4=S,5=HS
    # ------------------------------------------------------------------

    def rating_int_from_fraction(f: float) -> int:
        # clamp defensively
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
        return int(round(f * 5))  # 0..5

    def is_low_from_fraction(f: float) -> bool:
        return rating_int_from_fraction(f) <= 2

    def is_mid_from_fraction(f: float) -> bool:
        return rating_int_from_fraction(f) == 3

    def is_high_from_fraction(f: float) -> bool:
        return rating_int_from_fraction(f) >= 4

    def build_candidates(use_overall: bool) -> List[Dict[str, Any]]:
        """
        Build candidates in similarity order.
        If use_overall=False: only accept candidates with prefix fraction.
        If use_overall=True: accept candidates with prefix fraction OR overall fraction.
        """
        cands: List[Dict[str, Any]] = []
        for _, row in df_sim.iterrows():
            aid = str(row["activity_id"])
            if aid not in ratings:
                continue
            if get_rating_scale_info(aid, ratings) is None:
                continue

            f = primary_aid_fraction.get(aid)
            if f is None and use_overall and overall_aid_fraction:
                f = overall_aid_fraction.get(aid)

            if f is None:
                continue

            cands.append({
                "aid": aid,
                "frac": float(f),
                "similarity": float(row["similarity"]),
            })
        return cands

    def pick_first(cands: List[Dict[str, Any]], pred, selected_set: Set[str]) -> str | None:
        for c in cands:
            aid = c["aid"]
            if aid in selected_set:
                continue
            if pred(c["frac"]):
                return aid
        return None

    def pick_any(cands: List[Dict[str, Any]], selected_set: Set[str]) -> str | None:
        for c in cands:
            aid = c["aid"]
            if aid not in selected_set:
                return aid
        return None

    # ------------------------------------------------------------------
    # 1) Prefix-only selection first
    # ------------------------------------------------------------------

    cands_primary = build_candidates(use_overall=False)
    print(f"[KNN] {target_aid_str}: prefix candidates={len(cands_primary)}")

    # If there are literally no prefix candidates, go straight to overall if possible
    expanded = False
    if not cands_primary and overall_stats:
        print(f"[KNN] {target_aid_str}: no prefix candidates; expanding immediately to overall")
        cands_primary = build_candidates(use_overall=True)
        expanded = True
        print(f"[KNN] {target_aid_str}: overall candidates={len(cands_primary)}")

    if not cands_primary:
        print(f"[KNN] {target_aid_str}: ERROR no candidates even after expansion; returning []")
        return []

    selected: List[str] = []
    selected_set: Set[str] = set()

    got_low = got_mid = got_high = False

    # Pick LOW, MID, HIGH from prefix-only candidates (or overall if we already expanded)
    for name, pred in (("LOW", is_low_from_fraction), ("MID", is_mid_from_fraction), ("HIGH", is_high_from_fraction)):
        aid = pick_first(cands_primary, pred, selected_set)
        if aid:
            selected.append(aid)
            selected_set.add(aid)
            if name == "LOW": got_low = True
            if name == "MID": got_mid = True
            if name == "HIGH": got_high = True
            print(f"[KNN] {target_aid_str}: picked {name}={aid}")
        else:
            print(f"[KNN] {target_aid_str}: missing {name} in prefix candidates")

    # Decide if we need to expand
    need_bucket_fix = not (got_low and got_mid and got_high)
    need_fill_k = len(selected) < min(k, len(cands_primary))
    need_expand = (not expanded) and overall_stats and (need_bucket_fix or len(selected) < min(3, k) or len(selected) < k)

    if need_expand:
        expanded = True
        print(
            f"[KNN] {target_aid_str}: expanding to overall because "
            f"got_low/mid/high={got_low}/{got_mid}/{got_high}, selected={len(selected)}, k={k}"
        )
        cands_all = build_candidates(use_overall=True)
        print(f"[KNN] {target_aid_str}: overall candidates={len(cands_all)}")

        # Fill missing buckets first (global allowed now)
        if not got_low:
            aid = pick_first(cands_all, is_low_from_fraction, selected_set)
            if aid:
                selected.append(aid); selected_set.add(aid); got_low = True
                print(f"[KNN] {target_aid_str}: (expanded) filled LOW={aid}")
        if not got_mid:
            aid = pick_first(cands_all, is_mid_from_fraction, selected_set)
            if aid:
                selected.append(aid); selected_set.add(aid); got_mid = True
                print(f"[KNN] {target_aid_str}: (expanded) filled MID={aid}")
        if not got_high:
            aid = pick_first(cands_all, is_high_from_fraction, selected_set)
            if aid:
                selected.append(aid); selected_set.add(aid); got_high = True
                print(f"[KNN] {target_aid_str}: (expanded) filled HIGH={aid}")

        # Then fill to k using overall candidate list
        target_total = min(k, len(cands_all))
        while len(selected) < target_total:
            aid = pick_any(cands_all, selected_set)
            if aid is None:
                break
            selected.append(aid)
            selected_set.add(aid)

        print(f"[KNN] {target_aid_str}: final selected={len(selected)}/{k} (expanded=True)")
        print(f"[KNN] {target_aid_str}: selected ids={selected}")
        return selected

    # If we did not expand, we still may want to fill to k using prefix candidates
    target_total = min(k, len(cands_primary))
    while len(selected) < target_total:
        aid = pick_any(cands_primary, selected_set)
        if aid is None:
            break
        selected.append(aid)
        selected_set.add(aid)

    print(f"[KNN] {target_aid_str}: final selected={len(selected)}/{k} (expanded={expanded})")
    print(f"[KNN] {target_aid_str}: got_low/mid/high={got_low}/{got_mid}/{got_high}")
    print(f"[KNN] {target_aid_str}: selected ids={selected}")

    return selected


# def get_knn_neighbors(
#     target_aid,
#     ratings,
#     mock_forecasts,
#     top_n,
#     k,
#     rating_stats,
#     variant=None
# ) -> List[str]:
#     """
#     Return up to k neighbor activity_ids for target_aid that:
#       * are in ratings
#       * have a valid rating scale via get_ratings_text
#       * have baseline bundles (in baseline_map)

#     Neighbor selection logic:

#       1. Build a similarity-ordered candidate list from df_sim (up to top_n).
#       2. For each candidate, use rating_stats["aid_fraction"][aid] in [0,1] to
#          put it into one of 6 equal-width bins along the scale (worst→best).
#          (bin 1=worst, 6=best).
#       3. Use rating_stats["six_percents"] to compute target counts per bin
#          so that the k neighbors roughly follow the same distribution as the
#          full training set.
#       4. Walk candidates in similarity order, selecting neighbors to satisfy
#          those per-bin target counts.
#       5. If not enough neighbors have been chosen (due to missing bins, etc.),
#          fill the remaining slots with the most similar unused candidates.
#       6. Try to enforce at least one neighbor above and one below the midpoint
#          (fraction 0.5), if possible.
#     """

#     def six_bin(f: float) -> int:
#         """Map fraction f in [0,1] to a bin 1..6 (worst→best)."""
#         f = float(f)
#         if f < 0.0:
#             f = 0.0
#         elif f > 1.0:
#             f = 1.0
#         idx = min(5, int(f * 6))  # 0..5
#         return idx + 1            # 1..6

#     target_aid_str = str(target_aid)

#     # allowed_ids: Set[str] = {
#     #     str(aid)
#     #     for aid in ratings.keys()
#     #     if str(aid) != target_aid_str and str(aid) in baseline_map
#     # }
#     # all activities that have BOTH a rating and a mock forecast,
#     # excluding the target itself
#     allowed_ids: Set[str] = {
#         str(aid)
#         for aid in mock_forecasts.keys()
#         if str(aid) != target_aid_str and str(aid) in ratings
#     }

#     if not allowed_ids:
#         print("no allowed ids!")
#         return []
#     try:
#         results = find_similar_activities_semantic(
#             target_aid_str,
#             csv_path=str(ACTIVITY_INFO_CSV),
#             top_n=top_n,
#             allowed_ids=allowed_ids,
#         )
#     except ValueError as e:
#         if "not found in embeddings file" in str(e):
#             print("WARNING: failure to find activity id in embeddings file!")
#             return []
#         else:
#             raise


#     if len(results) < 2:
#         print("ERROR: seems nothing was found in similarity search..")
#         return []

#     df_sim, _search_item = results

#     print("len results")
#     print(len(df_sim))


#     # Fallback: no rating_stats, just take top-k similar with valid scales
#     if not rating_stats or "aid_fraction" not in rating_stats:
#         neighbor_ids: List[str] = []
#         for _, row in df_sim.iterrows():
#             aid = str(row["activity_id"])
#             if aid not in ratings:
#                 continue
#             if get_rating_scale_info(aid, ratings) is None:
#                 continue
#             # if aid not in baseline_map:
#             #     continue
#             neighbor_ids.append(aid)
#             if len(neighbor_ids) >= k:
#                 break
#         return neighbor_ids

#     aid_fraction_map: Dict[str, float] = rating_stats.get("aid_fraction") or {}
#     global_six_percents: Dict[int, float] = rating_stats.get("six_percents") or {}

#     # Build candidate list (up to top_n, in similarity order), with frac + bin
#     candidates: List[Dict[str, Any]] = []
#     for _, row in df_sim.iterrows():
#         aid = str(row["activity_id"])
#         if aid not in ratings:
#             continue
#         # if aid not in baseline_map:
#         #     continue
#         if get_rating_scale_info(aid, ratings) is None:
#             continue

#         frac = aid_fraction_map.get(aid)
#         if frac is None:
#             continue

#         cand_bin = six_bin(frac)
#         cand = {
#             "aid": aid,
#             "frac": frac,
#             "bin": cand_bin,
#             "similarity": float(row["similarity"]),
#         }
#         candidates.append(cand)

#     if not candidates:
#         print("ERROR: no candidates")
#         return []

#     total_k = min(k, len(candidates))
#     if total_k <= 0:
#         print("ERROR: negative K or negative candidates")
#         return []

#     # Compute target counts per bin, proportional to global_six_percents
#     bin_ids = [1, 2, 3, 4, 5, 6]
#     base_counts: Dict[int, int] = {b: 0 for b in bin_ids}
#     fractional: List[Any] = []
#     remaining = total_k

#     for b in bin_ids:
#         pct = float(global_six_percents.get(b, 0.0))
#         exact = total_k * pct / 100.0
#         base = int(exact)
#         base_counts[b] = base
#         fractional.append((exact - base, b))
#         remaining -= base

#     # Distribute leftover slots by largest fractional part
#     fractional.sort(reverse=True)
#     i = 0
#     while remaining > 0 and i < len(fractional):
#         b = fractional[i][1]
#         base_counts[b] += 1
#         remaining -= 1
#         i += 1

#     selected: List[str] = []
#     selected_set: Set[str] = set()
#     selected_counts: Dict[int, int] = {b: 0 for b in bin_ids}

#     # First pass: respect per-bin target counts in similarity order
#     for cand in candidates:
#         if len(selected) >= total_k:
#             break
#         aid = cand["aid"]
#         if aid in selected_set:
#             continue
#         b = cand["bin"]
#         if selected_counts[b] < base_counts.get(b, 0):
#             selected.append(aid)
#             selected_set.add(aid)
#             selected_counts[b] += 1

#     # Second pass: fill remaining slots with nearest unused candidates
#     if len(selected) < total_k:
#         for cand in candidates:
#             if len(selected) >= total_k:
#                 break
#             aid = cand["aid"]
#             if aid in selected_set:
#                 continue
#             selected.append(aid)
#             selected_set.add(aid)



#     # Enforce coverage of key rating regions without shrinking the rating range
#     if aid_fraction_map and total_k >= 2:

#         def rating_int(aid_str: str) -> int | None:
#             f = aid_fraction_map.get(aid_str)
#             return None if f is None else int(round(f * 5))  # 0..5

#         def is_mu_or_worse(aid_str: str) -> bool:
#             r = rating_int(aid_str)
#             return r is not None and r <= 2  # Moderately Unsatisfactory or worse

#         def is_ms_or_worse(aid_str: str) -> bool:
#             r = rating_int(aid_str)
#             return r is not None and r <= 3  # Moderately Satisfactory or worse

#         def is_s_or_better(aid_str: str) -> bool:
#             r = rating_int(aid_str)
#             return r is not None and r >= 4  # Satisfactory or Highly Satisfactory

#         def current_range(sel: list[str]) -> int:
#             rs = [rating_int(a) for a in sel]
#             rs = [r for r in rs if r is not None]
#             if not rs:
#                 return 0
#             return max(rs) - min(rs)

#         def try_inject(
#             *,
#             missing_pred,
#             replacement_pred,
#             replaceable_pred,
#         ) -> None:
#             nonlocal selected, selected_set

#             if any(missing_pred(a) for a in selected):
#                 return  # already represented

#             # find best (most similar) candidate of desired type not already selected
#             replacement = None
#             for cand in candidates:  # candidates are in similarity order
#                 aid = cand["aid"]
#                 if aid in selected_set:
#                     continue
#                 if replacement_pred(aid):
#                     replacement = aid
#                     break
#             if replacement is None:
#                 return  # no suitable candidate exists

#             base_range = current_range(selected)

#             # replace from the end (least similar among chosen), but only if range doesn't shrink
#             for i in range(len(selected) - 1, -1, -1):
#                 to_replace = selected[i]
#                 if not replaceable_pred(to_replace):
#                     continue

#                 new_sel = selected[:]
#                 new_sel[i] = replacement

#                 if current_range(new_sel) >= base_range:
#                     selected_set.remove(to_replace)
#                     selected[i] = replacement
#                     selected_set.add(replacement)
#                     return

#         # 1) If MU-or-worse is missing, inject one (replace anything if needed, but don't shrink range)
#         try_inject(
#             missing_pred=is_mu_or_worse,
#             replacement_pred=is_mu_or_worse,
#             replaceable_pred=lambda a: True,
#         )

#         # 2) If S-or-better is missing, inject one by replacing MS-or-worse,
#         #    but only if it doesn't shrink the range.
#         try_inject(
#             missing_pred=is_s_or_better,
#             replacement_pred=is_s_or_better,
#             replaceable_pred=is_ms_or_worse,
#         )

#     return selected
    
def build_few_shot_block(
    neighbor_ids: List[str],
    baseline_map: Dict[str, Dict[str, Any]],
    activity_info: Dict[str, Dict[str, str]],
    ratings: Dict[str, Dict[str, Any]],
    mock_forecasts: Dict[str, str],
    include_mock_forecast: bool = True,
) -> str:
    """
    Build the few-shot examples block:
    - basic meta (title, locations, scope)
    - optional ChatGPT description + risks_summary
    - optional full mock forecast text
    - final evaluation rating as a label.
    """
    lines: List[str] = []
    few_shot_actual = 0
    for idx, aid in enumerate(neighbor_ids, start=1):
        # bundle = baseline_map[aid]
        meta = activity_info.get(aid, {})
        title = (meta.get("activity_title") or "").strip()
        locations = (meta.get("country_location") or "").strip()
        scope = (meta.get("activity_scope") or "").strip()
        # chatgpt_description = (meta.get("chatgpt_description") or "").strip()
        # Prefer ChatGPT summary; if missing, fall back to activity_description from the CSV
        chatgpt_description = (
            meta.get("chatgpt_description")
            or meta.get("activity_description")
            or ""
        ).strip()

        risks_summary = (meta.get("risks_summary") or "").strip()

        rating_scale = get_rating_scale_info(aid, ratings)
        if rating_scale is None:
            print("no rating scale... skipping")
            continue
        final_result_for_prompt = rating_scale["final_result_for_prompt"]


        mock_text = mock_forecasts.get(aid, "").strip()

        lines.append(f"\nEXAMPLE {idx}:")
        if title:
            lines.append(f"ACTIVITY TITLE: {title}")
        # if variant == "A":
        #     if locations:
        #         lines.append(f"ACTIVITY LOCATIONS: {locations}")
        # # if scope:
        # #     lines.append(f"ACTIVITY SCOPE: {scope}")
        #     if chatgpt_description:
        #         lines.append(f"ACTIVITY SUMMARY: {chatgpt_description}")
        if risks_summary and risks_summary != "NO RESPONSE":
            formatted = format_risks_if_listlike(risks_summary)
            if "\n" in formatted:
                lines.append("ACTIVITY RISKS:\n" + formatted)
            else:
                lines.append(f"ACTIVITY RISKS: {formatted}")
        lines.append("")
        if include_mock_forecast and mock_text:
            lines.append(f"EXAMPLE FORECAST FOR EXAMPLE {idx}:")
            lines.append(mock_text)
            lines.append("")
            lines.append(f"END EXAMPLE FORECAST")
            lines.append(f"RATING SCALE FROM BEST TO WORST: {rating_scale['options_text']}")
            lines.append(f"FINAL EVALUATION OUTCOME FOR EXAMPLE FORECAST: {final_result_for_prompt}")
        lines.append("")
        few_shot_actual += 1
    # print("few_shot_actual")
    # print(few_shot_actual)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------
# Prompt builder (actual forecast, using the same structure as mock forecasts,
# but now with KNN few-shot examples injected)
# ---------------------------------------------------------------------

def make_scratchpad_methods(
    num_options: int,
    midpoint_low_text: str,
    midpoint_high_text: str,
    final_result_for_prompt: str,
) -> List[str]:
    scratchpad_method_bothsides = f"""
1. Provide reasons why the overall success might be rated {midpoint_low_text}.
2. Provide reasons why the overall success might be rated {midpoint_high_text}.
3. Aggregate your considerations, and decide on the final outcome among the {num_options} options.
4. Provide the final forecast on the last line beginning with 'FORECAST: ' followed by only the forecast with no extra words.
""".strip()

    scratchpad_method_tree = f"""
1. Develop a decision tree outlining possible paths among the {num_options} possible outcomes on this rating scale.
2. Analyze qualitatively how likely each branch is, based only on information that would be available at the start of the activity.
3. Use the decision tree to arrive at an initial broad judgment of whether the activity is more likely to be closer to {midpoint_low_text} or closer to {midpoint_high_text}.
4. Refine that broad judgment to select exactly one outcome from the {num_options} options.
5. Provide the final forecast on the last line beginning with 'FORECAST: ' followed by only the forecast with no extra words.
""".strip()

    return [scratchpad_method_bothsides, scratchpad_method_tree]



def build_prompts_with_few_shot(
    baseline_bundles: List[Dict[str, Any]],
    activity_info: Dict[str, Dict[str, str]],
    ratings: Dict[str, Dict[str, Any]],
    mock_forecasts: Dict[str, str],
    few_shot_k: int,
    variant_base: str,
    similarity_fn,
    rating_stats: Optional[Dict[str, Any]] = None,
    call_idx="",
    rag_answers_by_aid=None,
    knn_summary_by_aid=None,   # <-- ADD THIS=None,
    prev_by_stage: Dict[str, str] | None = None,
    stage: str | None = None,   # "s1", "s2", "s3"
    extra_rf_preds: Optional[Dict[str, float]] = None,
    default_rating_scale: Optional[Dict] = None,
) -> Dict[str, Dict[str, str]]:

    print("in build prompts, len ratings")
    print(len(ratings))
    # variant = variant + "_" + str(call_idx)
    # variant_base = variant.split("_aid_")[0]
    variant = variant_base + "_idx_" + str(call_idx)
    baseline_map = {str(b["activity_id"]): b for b in baseline_bundles}
    prompts: Dict[str, Dict[str, str]] = {}


    # decide per-variant switches
    include_mock_forecast = variant_base in LIST_OF_ALL_RECENT_VARIANTS
    # include_mock_forecast = variant_base == "B" or variant_base in LIST_OF_ALL_RECENT_VARIANTS
    # use_random_scratchpad = (variant == "A")
    use_random_scratchpad = False

    planned_by_aid: Dict[str, float] = {}
    _llm_expend_path = DATA_DIR / "llm_planned_expenditure.jsonl"
    with open(_llm_expend_path) as _f:
        for _line in _f:
            _rec = json.loads(_line)
            _v = _rec.get("planned_expenditure_usd")
            if _v and _v > 0:
                planned_by_aid[str(_rec["activity_id"])] = float(np.log(_v))

    n_knn_empty = 0
    if variant_base == "adjust_based_on_random_forest":
        rf_preds = load_ml_model_preds_for_prompts("../../data/best_model_predictions.csv", col="pred_rf_llm_modded")

    if "forced_rf" in variant_base:
        rf_preds = load_ml_model_preds_for_prompts("../../data/best_model_predictions.csv", col="pred_rf_llm_modded")
        if extra_rf_preds:
            rf_preds.update(extra_rf_preds)

    stat_interpretations = {}
    if "_with_explanation" in variant_base:
        stat_interpretations = load_stat_model_interpretations("../../data/stat_model_interpretations.csv")

    for bundle in baseline_bundles:

        aid = str(bundle["activity_id"])

        knn_summary_provided = (knn_summary_by_aid is not None)

        knn_txt = ""
        if knn_summary_provided:
            knn_txt = (knn_summary_by_aid.get(aid) or "").strip()
            if knn_txt == "NO RESPONSE":
                knn_txt = ""
                print("Error: unhandled no response condition")
                return

        # print("variant_base")
        # print(variant_base)
        # quit()
        is_rag_variant = (variant_base == "generate_rag_queries")
        is_knn_summary = (variant_base == "summarize_knn")
        # print("is_knn_summary")
        # print(is_knn_summary)
        early_only = (str(aid)[:5] == "DE-1-") or (str(aid)[:4] == "DE-1")  # robust-ish

        prev_by_stage = prev_by_stage or {}
        s1_text = (prev_by_stage.get("s1", {}).get(aid) or "").strip()
        s2_text = (prev_by_stage.get("s2", {}).get(aid) or "").strip()
        s3_text = (prev_by_stage.get("s3", {}).get(aid) or "").strip()

        # print("ratings")
        # print(ratings)
        rating_scale = get_rating_scale_info(aid, ratings)

        if rating_scale is None:
            if default_rating_scale is not None:
                rating_scale = default_rating_scale
            else:
                print("ERROR: skipping, because missing rating scale")
                continue

        num_options = rating_scale["num_options"]
        midpoint_low_text = rating_scale["midpoint_low_text"]
        midpoint_high_text = rating_scale["midpoint_high_text"]
        options_text = rating_scale["options_text"]
        final_result_for_prompt = rating_scale["final_result_for_prompt"]

        text_to_describe_rating_distribution = get_text_to_describe_rating_distribution(aid,ratings,rating_stats,num_options)

        # print("len(set(ratings))")
        # print(len(set(ratings)))
        # print("len(set(mock_forecasts))")
        # print(len(set(mock_forecasts)))
        # print("len(set(mock_forecasts) & set(ratings))")
        # print(len(set(mock_forecasts) & set(ratings)))
        # Build KNN neighbors for this activity

        get_fewshot_directly_instead_of_knn_summary = False
        if knn_summary_by_aid and (knn_summary_by_aid.get(str(aid)) or "").strip() == "":
            # in this case, we're trying to insert knn summary by aid, but this one didn't have any few-shot to get.
            get_fewshot_directly_instead_of_knn_summary = True
            if few_shot_k > 0:
                # print("WARNING: in this case, we're trying to insert knn summary by aid, but this one didn't have any few-shot summary to get, so we insert directly.")
                print("WARNING: knn_summary had empty text for this aid (but attempted few-shot). Falling back to direct few-shot.")
                n_knn_empty += 1
        # get_fewshot_directly_instead_of_knn_summary = False

        # if knn_summary_by_aid is not None:
        #     knn_txt = (knn_summary_by_aid.get(aid) or "").strip()

        #     # only treat as "missing summary" if the file had an entry for this aid
        #     if aid in knn_summary_by_aid and knn_txt == "":
        #         get_fewshot_directly_instead_of_knn_summary = True
        #         if few_shot_k > 0:
        #             print("WARNING: knn_summary had empty text for this aid. Falling back to direct few-shot.")
        #             n_knn_empty += 1
        # else:
        #     knn_txt = ""

        add_in_fewshot_examples = False
        if few_shot_k == 0:
            # if ther's nothing to get, then definitely don't insert anything.
            add_in_fewshot_examples = False
            if is_knn_summary:
                print("WARNING: could not insert knn summary: few shot k was set to zero.")
                continue
        elif get_fewshot_directly_instead_of_knn_summary:
            # if we passed in a summarized knn, then we definitely want the example fewshot in there
            # (as long as there are some to get)
            add_in_fewshot_examples = True
        elif is_knn_summary:
            # there are examples, but we didn't try to insert a summary. 
            # The only case we want to insert examples summary is if we're trying to summarize it.
            add_in_fewshot_examples = True

        if not add_in_fewshot_examples:
            neighbor_ids = []
            few_shot_block = None
        else:
            neighbor_ids = get_knn_neighbors(
                target_aid=aid,
                ratings=ratings,
                mock_forecasts=mock_forecasts,
                top_n=SIMILARITY_TOP_N,
                k=few_shot_k,
                rating_stats=rating_stats,
                similarity_fn=similarity_fn,
                variant=variant,
            )
            print("N_NEIGHBORS:")
            print(len(neighbor_ids))

        if not neighbor_ids:
            few_shot_block = None
            if is_knn_summary:
                print("KNN summary prompt found no neighbors! skipping this prompt")
                continue
        else:
            few_shot_block = build_few_shot_block(
                neighbor_ids=neighbor_ids,
                baseline_map=baseline_map,
                activity_info=activity_info,
                ratings=ratings,
                mock_forecasts=mock_forecasts,
                include_mock_forecast=include_mock_forecast,
            )

            if not few_shot_block:
                print("ERROR: one of them had NO few shot block... skipping this prompt")
                continue


        meta = activity_info.get(aid, {})
        title = (meta.get("activity_title") or bundle.get("title") or "").strip()
        orgs = (meta.get("reporting_orgs") or "").strip()
        locations = (meta.get("country_location") or "").strip()
        scope = (meta.get("activity_scope") or "").strip()
        gdp_percap = (meta.get("gdp_percap") or "").strip()
        implementing_org_type = (meta.get("implementing_org_type") or "").strip()
        planned_start_date = (
            meta.get("original_planned_start_date")
            if meta.get("original_planned_start_date")
            else meta.get("actual_start_date")
        )
        planned_end_date = meta.get("original_planned_close_date")
        risks_summary = (meta.get("risks_summary") or "").strip()
        chatgpt_description = (
            meta.get("chatgpt_description")
            # or meta.get("activity_description") # too risky to leak data
            or ""
        ).strip()

        if "exactly_like_halawi_et_al" in variant_base or variant_base == "onlysummary_no_knn_no_rag":

            # print("indeed, exactly like halawi et al!")
            # print(f"stage: {stage}")
            # quit()
            if stage is None:
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
                scratchpad_method = f"""{rating_text_dist}1. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the answer might be \"Moderately Satisfactory\" or worse. Rate the strength of each reason.
2. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the forecast might be \"Satisfactory\" or \"Highly Satisfactory\". Rate the strength of each reason. 
3. Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver).
4. Output an initial forecast given steps 1-3 on the following rating scale: {options_text}.
5. Evaluate whether your forecast is too extreme, or not extreme enough (whether "moderately", no modifier, or "highly" is the best choice). Also, consider anything else that might affect the forecast that you did not before consider.
6. At the very end, on the last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale, with no extra words.""".strip()
            elif stage == "s1":
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
                scratchpad_method = f"""{rating_text_dist}\nUsing your knowledge of the world and topic, as well as the information provided, provide a few reasons why the answer might be \"Moderately Satisfactory\" or worse. Rate the strength of each reason."""
            elif stage == "s2":
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
                scratchpad_method = f"""{rating_text_dist}A few reasons you said why the answer might be \"Moderately Satisfactory\" or worse:\n{s1_text}\n\nYOUR TASK:\nUsing your knowledge of the world and topic, as well as the information provided, provide a few reasons why the forecast might be \"Satisfactory\" or \"Highly Satisfactory\". Rate the strength of each reason."""

            elif stage == "s3":
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
                # Check if this is the forced_rf variant - if so, override s3 behavior
                if "forced_rf" in variant_base:
                    # Get the RF prediction for this activity
                    rf_pred_value = rf_preds.get(str(aid))
                    if rf_pred_value is None:
                        print(f"WARNING: No RF prediction found for {aid}, skipping")
                        continue
                    rf_pred_label, rf_pred_detail = rf_pred_label_and_number(rf_pred_value)
                    rf_pred_text = f"{rf_pred_label} {rf_pred_detail}"
                    explanation_line = f"\n\n{stat_interpretations[str(aid)]}" if str(aid) in stat_interpretations else ""

                    scratchpad_method = f"""{rating_text_dist}TASK: Your goal is to produce a forecast that arrives at the same rating as the statistical method predictor, which forecasts the correct rating ~55% of the time. The statistical model (trained on planned duration, total planned disbursement, and other quantitative/categorical information) predicts this activity will receive: {rf_pred_text}. Ensure to weigh the interpreted drivers from the statistical model into your response: {explanation_line}

Here are a few reasons that you said the answer might be "Moderately Satisfactory" or worse:
{s1_text}

Here are a few reasons that you said the answer might be "Satisfactory" or better:
{s2_text}

YOUR TASK:
Aggregate your considerations and provide reasoning that supports arriving at the statistical model's prediction of {rf_pred_label}. Focus on the key drivers most likely to cause this outcome, and provide commentary on the degree to which the core goals of the activity will likely be achieved. Think like a superforecaster (e.g. Nate Silver).

On the very last line of your response, write 'FORECAST: {rf_pred_label}'"""
                elif s1_text == "" or s2_text == "":
                    scratchpad_method = f"""{rating_text_dist}\nYOUR TASK:\n Aggregate the your considerations above. Think like a superforecaster (e.g. Nate Silver). On the very last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale with no extra words: {options_text}.""".strip()
                else:
                    # scratchpad_method = f"""{rating_text_dist}Here are a few reasons that you said the answer might be \"Moderately Satisfactory\" or worse:\n{s1_text}\nHere are a few reasons that you said the answer might be \"Satisfactory\" or better:\n{s2_text}\n\nYOUR TASK:\n Aggregate the your considerations above. Think like a superforecaster (e.g. Nate Silver). Output an initial forecast on this rating scale: {options_text}.""".strip()
                    scratchpad_method = f"""{rating_text_dist}Here are a few reasons that you said the answer might be \"Moderately Satisfactory\" or worse:\n{s1_text}\nHere are a few reasons that you said the answer might be \"Satisfactory\" or better:\n{s2_text}\n\nYOUR TASK:\n Aggregate the your considerations above. Think like a superforecaster (e.g. Nate Silver). On the very last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale with no extra words: {options_text}.""".strip()
            elif stage == "final":
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
                # scratchpad_method = f"""{rating_text_dist}Here are a few reasons that you said the answer might be \"Moderately Satisfactory\" or worse:\n{s1_text}\n Here are a few reasons that you said the answer might be \"Satisfactory\" or better:\n{s2_text}\n Next, aggregating your considerations and thinking like a superforecaster (e.g. Nate Silver), your first forecast attempt at a forecast went as follows: {s3_text}.\n\nYOUR TASK:\n Evaluate whether your forecast is too extreme, or not extreme enough (whether "moderately", no modifier, or "highly" is the best choice), or if you chose "Moderately Satisfactory", whether in fact the forecast should really be "Moderately Unsatisfactory". Also, consider anything else that might affect the forecast that you did not before consider. On the very last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale with no extra words: {options_text}."""
                scratchpad_method = f"""{rating_text_dist}Here are a few reasons that you said the answer might be \"Moderately Satisfactory\" or worse:\n{s1_text}\n Here are a few reasons that you said the answer might be \"Satisfactory\" or better:\n{s2_text}\n Next, aggregating your considerations and thinking like a superforecaster (e.g. Nate Silver), your first forecast attempt at a forecast went as follows: {s3_text}.\n\nYOUR TASK:\n Consider anything else that might affect the forecast that you did not before consider. On the very last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale with no extra words: {options_text}."""
            else:
                print("ERROR, unknown stage!")
                raise ValueError(f"Unknown stage: {stage}")
        elif "choose_your_own_adventure" in variant_base:
            rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
            scratchpad_method = f"""\n{rating_text_dist}Provide the following format for your response:\n\n1. Determine a suitable strategy for breaking down the challenge of forecasting this activity's final overall success evaluation, that would produce an excellent forecast on this specific activity's success.
2. Reason about what the most likely outcome will be using this forecasting strategy on the following rating scale: {options_text}.
3. At the very end, on the last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale with the most likely forecast outcome, with no extra words.""".strip()
        elif "consider_the_knn" in variant_base:
            rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
            scratchpad_method = f"""\n{rating_text_dist}Provide the following format for your response:\n\n1. List all the separate reasons that the example forecasts described above went well or badly, given their evaluations, if such considerations could apply to this activity. Rate the applicability of each consideration.
2. Aggregate the considerations into an initial forecast for this activity's success on the following scale: {options_text}.
4. Evaluate whether your forecast is too extreme, or not extreme enough (whether "moderately", no modifier, or "highly" would be appropriate). Also, consider anything else that might affect the forecast that you did not before consider.
3. At the very end, on the last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale with the most likely forecast outcome, with no extra words.""".strip()
        elif "adjust_based_on_random_forest" in variant_base:
            rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
            scratchpad_method = f"""\n{rating_text_dist}Provide the following format for your response:\n\n1. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the evaluation might be better than the random forest model prediction. Rate the strength of each reason.
            2. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the evaluation might be the same as or worse than the random forest model prediction. Rate the strength of each reason. 
            3. Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver).
            4. Output an initial forecast given steps 1-3 on the following rating scale: {options_text}.
            5. Evaluate whether your forecast is too extreme, or not extreme enough (whether "moderately", no modifier, or "highly" is the best choice). Also, consider anything else that might affect the forecast that you did not before consider.
            6. At the very end, on the last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale, with no extra words.""".strip()

        else:
            # Scratchpad instructions: random style for A, fixed for B/C
            if use_random_scratchpad:
                scratchpad_options = make_scratchpad_methods(
                    num_options=num_options,
                    midpoint_low_text=midpoint_low_text,
                    midpoint_high_text=midpoint_high_text,
                    final_result_for_prompt=final_result_for_prompt,
                )
                # scratchpad_method = random.choice(scratchpad_options)
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
                scratchpad_method = rating_text_dist + "\nProvide the following format for your response:\n" + scratchpad_options[0]
            else:
                # fixed, simple instruction: mimic examples, end with FORECAST:
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
                scratchpad_method = f"""{rating_text_dist}\nProvide the following format for your response:\n1. Carefully reason about the likely outcome{', using the style of reasoning shown in the example forecasts above' if few_shot_block else ''}.
2. Explicitly weigh reasons for lower outcomes versus reasons for higher outcomes on this rating scale: {options_text}.
3. At the very end, on the last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale, with no extra words.""".strip()
        if is_rag_variant:
            # No rating-scale needed in the output, but keeping the earlier rating_scale load is fine.
            system_msg = f"""You are an experienced international aid decision maker with a quantitative mindset. "
Your job is to identify missing-but-important information from activity information documents and produce search phrases to find them in the documents. {'Only target facts that would be knowable at the start of the activity; ignore later implementation results. ' if early_only else ""}You format the final lines of your response with exactly 5 phrases, with one line each per phrase, e.g.,\n\nPHRASE 1: <query>\nPHRASE 2: <query>\nPHRASE 3: <query>\nPHRASE 4: <query>\nPHRASE 5: <query>"""

            prompt_lines = []

            prompt_lines.append(f"First, consider what information is available, and what is generally unavailable but would be useful to know, in order to forecast the activity outcome on the following scale: {options_text}.")
            prompt_lines.append("Second, generate five short search phrases to look up in the activity's documents. selected from the list below or similar to it, and customized to fill key informational gaps for forecasting the current activity's eventual evaluation rating.")
            prompt_lines.append("")
            prompt_lines.append("EXAMPLE QUERY PHRASES:")
            questions = [
                "Readiness of bidding documents for the biggest contracts",
                "Land needed to be acquired for works and feasibility",
                "Quality of plan to manage environmental and social harms",
                "Budget for operations and maintenance after closing",
                "Number of approvals needed and approval risks",
                "Risk of policy reversal after funds are disbursed",
                "Procurement delays for recent, similar projects ",
                "Contractor market interest and realistic competition",
                "Credibility of cost estimates against recent local prices",
                "Budget release reliability and arrears history",
                "Strength of political support and likely blockers",
                "Dependence on other donors or parallel projects",
                "Credible plan to measure specific outcomes",
                "How strongly benefits may be affected by delays and overruns",
            ]

            random.shuffle(questions)
            prompt_lines.extend(f"- {q}" for q in questions)
            prompt_lines.append("")

            prompt_lines.append("### ACTIVITY TO FORECAST ###")

            # prompt_text = "\n".join(prompt_lines)

            # prompts[aid] = {
            #     "system_msg": system_msg,
            #     "prompt": prompt_text,
            #     "prompt_type": f"fewshot_k{few_shot_k}_variant_{variant}",
            # }
            # continue  # IMPORTANT: skip the rest of the forecast-building logic
        elif is_knn_summary:
            system_msg = (
                "You are an experienced international aid decision maker with a quantitative mindset. "
                "Provide a balanced and thoughtful assessment of how similar activities were rated. Do not attempt to forecast the current activity outcome."
            )
            if early_only:
                system_msg += " Only include information that would be knowable at the start of the activity; ignore later implementation results. Under no circumstances reveal actual ex-post information."

            prompt_lines = []
            prompt_lines.append(f"""You are extracting and analyzing information from the examples below as applies to the current activity.""")
            if not few_shot_block:
                print("WARNING: NO FEW SHOT BLOCK IN KNN SUMMARY. SKIPPING")
                continue
            #     prompt_lines.append("### EXAMPLE ACTIVITIES ###")
            #     prompt_lines.append(few_shot_block)
            #     prompt_lines.append("### END EXAMPLE ACTIVITIES ###\n")
            # else:
            # prompt_lines.append("### NEW ACTIVITY TO FORECAST ###")

        else:
            if stage in ("final","s3"):
                prompt_lines: List[str] = []
                system_msg = (
                    "You are an experienced international aid decision maker with a quantitative mindset. "
                    f"Respond with a comprehensive, thorough forecast of what the overall evaluation rating of the activity will be, from the options of {options_text}."
                )
                prompt_lines.append(
                    "Forecast what the outcome will be for this activity."
                )
            elif stage in ("s1","s2"):
                system_msg = (
                    "You are an experienced international aid decision maker with a quantitative mindset. "
                    f"Provide a thorough, thoughtful response without coming to any premature conclusions."
                )
                prompt_lines: List[str] = []
                prompt_lines.append(
                    "You are considering the likely outcomes of this activity."
                )
            # elif stage in ("s3",):
            #     system_msg = (
            #         "You are an experienced international aid decision maker with a quantitative mindset. "
            #         f"Provide a thorough, thoughtful response."
            #     )
            #     prompt_lines: List[str] = []
            #     prompt_lines.append(
            #         "You are considering the likely outcomes of this activity."
            #     )

        if variant_base == "summarize_knn": # (previously we skipped if no neighbor ids)
            prompt_lines.append("### EXAMPLE ACTIVITIES ###")
            prompt_lines.append(few_shot_block)
            prompt_lines.append("### END EXAMPLE ACTIVITIES ###\n")
        elif knn_summary_provided and "_no_knn" not in variant_base:  # ADD CHECK HERE:
            if knn_txt:
                prompt_lines.append("\n### Lessons from similar activities ###")
                prompt_lines.append(knn_txt)
                prompt_lines.append("### End lessons ###\n")
            elif few_shot_block:
                prompt_lines.append("\n### EXAMPLE ACTIVITIES ###")
                prompt_lines.append(few_shot_block)
                prompt_lines.append("### END EXAMPLE ACTIVITIES ###\n")

        # if knn_txt:
        #     prompt_lines.append("\n### Lessons from similar activities ###")
        #     prompt_lines.append(knn_txt)
        #     prompt_lines.append("### End lessons ###\n")
        # elif few_shot_block:
        #     prompt_lines.append("\n### EXAMPLE ACTIVITIES ###")
        #     prompt_lines.append(few_shot_block)
        #     prompt_lines.append("### END EXAMPLE ACTIVITIES ###\n")


        # prompt_lines.append(
        #     "Your response will be balanced and comprehensive, including consideration of the "
        #     "information from the uploaded activity documents and the example forecasts for "
        #     "similar activities shown below."
        # )
        prompt_lines.append("")
        # prompt_lines.append(
        #     f"First, study the following {len(neighbor_ids)} forecast(s), each paired with its final evaluation outcome:"
        # )
        # prompt_lines.append("")
        prompt_lines.append(f"ACTIVITY ID: {aid}")
        if variant_base != "onlysummary_no_knn_no_rag":
            if title:
                prompt_lines.append(f"ACTIVITY TITLE: {title}")
            if planned_start_date:
                prompt_lines.append(f"ORIGINAL PLANNED START DATE: {planned_start_date}")
            if planned_end_date:
                prompt_lines.append(f"ORIGINAL PLANNED END DATE: {planned_end_date}")
        # if scope:
        #     prompt_lines.append(f"ACTIVITY SCOPE: {scope}")
        # if locations:
        #     prompt_lines.append(f"ACTIVITY LOCATION(S): {locations}")
        # if gdp_percap:
        #     try:
        #         prompt_lines.append(
        #             f"LOCATION GDP PER CAPITA, USD: {int(float(gdp_percap))}"
        #         )
        #     except Exception:
        #         prompt_lines.append(f"LOCATION GDP PER CAPITA, USD: {gdp_percap}")
        # if orgs:
        #     prompt_lines.append(f"PARTICIPATING ORGANIZATIONS: {orgs}")
        # if implementing_org_type:
        #     prompt_lines.append(
        #         f"IMPLEMENTING ORGANIZATION CATEGORY: {implementing_org_type}"
        #     )

        # if variant_base == "A" or variant == "B":

        #     if chatgpt_description:
        #         prompt_lines.append(f"ACTIVITY DESCRIPTION: {chatgpt_description}")

        #     if risks_summary:
        #         prompt_lines.append(f"ACTIVITY RISKS: {risks_summary}")


        if variant_base in LIST_OF_ALL_RECENT_VARIANTS:

            title = (meta.get("activity_title") or "").strip()
            orgs = (meta.get("reporting_orgs") or "").strip()
            locations = (meta.get("country_location") or "").strip()
            gdp_percap = (meta.get("gdp_percap") or "").strip()
            implementing_org_type = (meta.get("implementing_org_type") or "").strip()
            planned_start_date = meta.get("original_planned_start_date") if meta.get("original_planned_start_date") != "" else  meta.get("actual_start_date")
            planned_end_date = meta.get("original_planned_close_date")
            activity_scope = (meta.get("activity_scope") or "").strip()

            activity_context = (meta.get("activity_context") or "").strip()

            is_RCT = (bundle.get('is_RCT') or "")

            disbursement_total = (bundle.get('disbursement_total') or "")
            loan_total = (bundle.get('loan_total') or "")
            disbursement_units = (bundle.get('disbursement_units') or "")
            loan_units = (bundle.get('loan_units') or "")

            activity_context = (bundle.get('activity_context') or "").strip()
            complexity_details = (bundle.get('complexity_details') or "").strip()
            how_integrated_description = (bundle.get('how_integrated_description') or "").strip()
            finance_summary = (bundle.get('finance_summary') or "").strip()
            implementer_performance_text = (bundle.get('implementer_performance_text') or "").strip()
            risks_summary = (bundle.get('risks_summary') or "").strip()
            possibilities_summary = (bundle.get('possibilities_summary') or "").strip()
            targets_summary = (bundle.get('targets_summary') or "").strip()

            context_grade_text = (bundle.get('context_grade_text') or "").strip()
            complexity_grade_text = (bundle.get('complexity_grade_text') or "").strip()
            integratedness_grade_text = (bundle.get('integratedness_grade_text') or "").strip()
            finance_grade_text = (bundle.get('finance_grade_text') or "").strip()
            implementer_performance_grade_text = (bundle.get('implementer_performance_grade_text') or "").strip()
            targets_grade_text = (bundle.get('targets_grade_text') or "").strip()
            risks_grade_text = (bundle.get('risks_grade_text') or "").strip()

            # pprint.pprint("bundle")
            # pprint.pprint(bundle)

            # prompt_lines.append(f"""\nACTIVITY TITLE: {title}""")
            # if planned_start_date != "" and planned_start_date is not None and planned_start_date != "NO RESPONSE":
            #     prompt_lines.append(f"\nPLANNED START DATE: {planned_start_date}")
            # if planned_end_date != "" and planned_end_date is not None and planned_end_date != "NO RESPONSE":
            #     prompt_lines.append(f"\nPLANNED END DATE: {planned_end_date}")
            if activity_scope != "" and activity_scope is not None and activity_scope != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY SCOPE: {activity_scope}")
            planned_exp = planned_by_aid.get(aid)
            if planned_exp is not None:
                planned_usd = float(np.exp(planned_exp))
                prompt_lines.append(f"\nPLANNED TOTAL DISBURSEMENT (USD): {planned_usd:,.0f}")
                # prompt_lines.append(f"\nPLANNED TOTAL DISBURSEMENT: {disbursement_total} {disbursement_units}")
            # if disbursement_total != "" and (disbursement_total != "NO RESPONSE"):
            # if loan_total != "" and (loan_total != "NO RESPONSE"):
            #     prompt_lines.append(f"\nPLANNED TOTAL LOANS AND CREDIT: {loan_total} {loan_units}")
            if locations != "" and locations is not None and locations != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY LOCATION(S): {locations}")
            if gdp_percap != "" and gdp_percap is not None and gdp_percap != "NO RESPONSE":
                try:
                    gdp_int = int(float(gdp_percap))
                    prompt_lines.append(f"\nLOCATION GDP PER CAPITA, USD: {gdp_int}")
                except ValueError:
                    prompt_lines.append(f"\nLOCATION GDP PER CAPITA, USD: {gdp_percap}")
                # prompt_lines.append(f"\nLOCATION GDP PER CAPITA, USD: {int(float(gdp_percap))}")
            if orgs != "" and orgs is not None and orgs != "NO RESPONSE":
                prompt_lines.append(f"\nPARTICIPATING ORGANIZATIONS: {orgs}")
            if implementing_org_type != "" and implementing_org_type is not None and implementing_org_type != "NO RESPONSE" and implementing_org_type.lower() != "other":
                prompt_lines.append(f"\nIMPLEMENTING ORGANIZATION CATEGORY: {implementing_org_type}")

            if chatgpt_description and chatgpt_description != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY DESCRIPTION: {chatgpt_description}")
            if targets_summary and targets_summary != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY TARGETS: {targets_summary}")
            if activity_context and activity_context != "NO RESPONSE":
                formatted = format_risks_if_listlike(activity_context)
                prompt_lines.append(f"\nACTIVITY CONTEXT: {formatted}")


            if complexity_details != "" and complexity_details is not None and complexity_details != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY COMPLEXITY: {complexity_details}")
            if how_integrated_description != "" and how_integrated_description is not None and how_integrated_description != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY INTEGRATEDNESS: {how_integrated_description}")
            # if finance_summary != "" and finance_summary is not None and finance_summary != "NO RESPONSE":
            #     prompt_lines.append(f"\nFINANCING DETAILS: {finance_summary}")
            if finance_summary and finance_summary != "NO RESPONSE":
                formatted = format_risks_if_listlike(finance_summary)
                prompt_lines.append("\nFINANCING DETAILS: " + formatted)
            if implementer_performance_text != "" and implementer_performance_text is not None and implementer_performance_text != "NO RESPONSE":
                prompt_lines.append(f"\nIMPLEMENTER PERFORMANCE CONTEXT: {implementer_performance_text}")
            # if risks_summary != "" and risks_summary is not None and risks_summary != "NO RESPONSE":
            #     prompt_lines.append(f"\nACTIVITY RISKS: {risks_summary}")
            if risks_summary and risks_summary != "NO RESPONSE":
                formatted = format_risks_if_listlike(risks_summary)
                if "\n" in formatted:
                    prompt_lines.append("ACTIVITY RISKS:\n" + formatted)
                else:
                    prompt_lines.append(f"ACTIVITY RISKS: {formatted}")

            if possibilities_summary and possibilities_summary != "NO RESPONSE":
                formatted = format_risks_if_listlike(possibilities_summary)
                if "\n" in formatted:
                    prompt_lines.append(f"\nACTIVITY POSSIBILITIES: {formatted}")
                else:
                    prompt_lines.append(f"\nACTIVITY POSSIBILITIES: {formatted}")

            # if possibilities_summary != "" and possibilities_summary is not None and possibilities_summary != "NO RESPONSE":
            #     prompt_lines.append(f"\nACTIVITY POSSIBILITIES: {possibilities_summary}")
            if variant_base == "short_well_or_badly_add_grades":
                # Pull grade-style fields from the enriched bundle (rows)
                context_grade_text = (bundle.get("context_grade_text") or "").strip()
                complexity_grade_text = (bundle.get("complexity_grade_text") or "").strip()
                integratedness_grade_text = (bundle.get("integratedness_grade_text") or "").strip()
                finance_grade_text = (bundle.get("finance_grade_text") or "").strip()
                implementer_performance_grade_text = (bundle.get("implementer_performance_grade_text") or "").strip()
                targets_grade_text = (bundle.get("targets_grade_text") or "").strip()
                risks_grade_text = (bundle.get("risks_grade_text") or "").strip()

                # Only append non-empty, non-“NO RESPONSE” grades
                def ok_grade(txt: str) -> bool:
                    return (
                        txt
                        and txt != "GRADE: NO RESPONSE"
                        and txt != "NO RESPONSE"
                    )

                if ok_grade(context_grade_text):
                    prompt_lines.append(f"CONTEXT {context_grade_text}")
                if ok_grade(complexity_grade_text):
                    prompt_lines.append(f"COMPLEXITY {complexity_grade_text}")
                if ok_grade(integratedness_grade_text):
                    prompt_lines.append(f"INTEGRATEDNESS {integratedness_grade_text}")
                if ok_grade(finance_grade_text):
                    prompt_lines.append(f"FINANCE {finance_grade_text}")
                if ok_grade(implementer_performance_grade_text):
                    prompt_lines.append(f"IMPLEMENTER PERFORMANCE {implementer_performance_grade_text}")
                if ok_grade(targets_grade_text):
                    prompt_lines.append(f"TARGETS {targets_grade_text}")
                # if ok_grade(risks_grade_text):
                #     prompt_lines.append(f"RISKS {risks_grade_text}")
            # --- Inject KNN-summary (if available) into forecast prompts ---
            # print("WARNING:only knn summaries are getting fewshot, might want to alter later")
            # if knn_summary_by_aid:
            #     knn_txt = (knn_summary_by_aid.get(str(aid)) or "").strip()
            #     if knn_txt and knn_txt != "NO RESPONSE":
            #         prompt_lines.append("\n### Lessons from similar activities ###")
            #         prompt_lines.append(knn_txt)
            #         prompt_lines.append("### End KNN summary ###\n")

        if variant_base == "onlysummary_no_knn_no_rag":
            if chatgpt_description and chatgpt_description != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY DESCRIPTION: {chatgpt_description}")

        if ("rag_added" in variant_base and "_no_rag" not in variant_base):  # SIMPLIFIED CHECK
            if not rag_answers_by_aid:
                    print(f"WARNING: there was no rag info injected for {aid}! skipping")
                    # continue 
            else:
                inserted_context = (rag_answers_by_aid.get(str(aid)) or "").strip()
                if not inserted_context:
                    print(f"WARNING: skipping as rag did not return any additional information {aid}")
                else:
                    if inserted_context != "NO RESPONSE":
                        prompt_lines.append("\n### Additional specific information about the activity that you summarized ###")
                        prompt_lines.append(inserted_context)
                        prompt_lines.append("\n### End of additional information you summarized ###")


            # target_text = collapse_activity_items_to_text(
            #     bundle,
            #     max_chars=MAX_TARGET_CHARS,
            #     allow_ocr=True,
            # )

            # prompt_lines.append("")
            # prompt_lines.append("### START DETAILED INFORMATION ON ACTIVITY TO FORECAST ###")
            # prompt_lines.append("")
            # prompt_lines.append(target_text.strip())
            # prompt_lines.append("")
            # prompt_lines.append("### END DETAILED INFORMATION ON ACTIVITY TO FORECAST ###")
            # prompt_lines.append("")

        # Convert 6-category scale to numeric
        RATING_MAP = {
            'Highly Unsatisfactory': 0,
            'Unsatisfactory': 1,
            'Moderately Unsatisfactory': 2,
            'Moderately Satisfactory': 3,
            'Satisfactory': 4,
            'Highly Satisfactory': 5
        }
        INV_RATING_MAP = {v:k for k,v in RATING_MAP.items()}

        if variant_base == "adjust_based_on_random_forest":
            v = rf_preds.get(str(aid))
            if v is not None:
                prompt_lines.append(f"\nMODEL PREDICTION (random forest, average R^2 of ~0.2, based on planned duration, total planned disbursement, and the other quantitative or categorical information listed above): {INV_RATING_MAP.get(int(np.round(float(v))))}")

        if "forced_rf" in variant_base:
            v = rf_preds.get(str(aid))
            if v is not None:
                rf_label, rf_detail = rf_pred_label_and_number(v)
                prompt_lines.append(f"\nSTATISTICAL MODEL PREDICTION: {rf_label} {rf_detail}")
                if str(aid) in stat_interpretations:
                    prompt_lines.append(f"STATISTICAL MODEL EXPLANATION: {stat_interpretations[str(aid)]}")

        if is_rag_variant:
            prompt_lines.append(
                "\nProvide the following format for your response:"
            )
            prompt_lines.append("")
            prompt_lines.append("COMPREHENSIVE REASONING ABOUT GOOD PHRASES: <extensive reasoning>")
            prompt_lines.append("PHRASE 1: ...")
            prompt_lines.append("PHRASE 2: ...")
            prompt_lines.append("PHRASE 3: ...")
            prompt_lines.append("PHRASE 4: ...")
            prompt_lines.append("PHRASE 5: ...")
        elif is_knn_summary:
            prompt_lines.append(
                "\nPlease respond as follows:"
            )
            prompt_lines.append(f"""List all the separate reasons that the example forecasts described above went well or badly, given their evaluations, if such considerations could apply to this activity. Rate the applicability of each consideration. What are the relevant lessons that can be learned as could apply to forecasting the outcome of this activity? Describe the key reasons each example was given the rating they were.""")
            if early_only:
                prompt_lines.append(f"""Ensure the only information given is that which could be known or reasonably forecasted at the start of the activity.""")
        else:
            prompt_lines.append("")
            prompt_lines.append(scratchpad_method)
            prompt_lines.append("")



        prompt_lines.append(
            "Respond only in English."
        )

        prompt_text = "\n".join(prompt_lines)

        prompts[aid] = {
            "system_msg": system_msg,
            "prompt": prompt_text,
            "prompt_type": f"fewshot_k{few_shot_k}_variant_{variant}",
            "knn_neighbor_ids": list(neighbor_ids),
        }

        # print("\n\nprompts[aid]")
        # print(prompts[aid])
        # input()

        # break
    print("n_knn_empty:")
    print(n_knn_empty)
    print("")
    print("")
    print("DONE CONSTRUCTING PROMPTS")
    print("")
    return prompts
