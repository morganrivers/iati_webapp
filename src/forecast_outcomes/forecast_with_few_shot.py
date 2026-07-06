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

import ast
import re

LIST_OF_ALL_RECENT_VARIANTS = ("short_well_or_badly", "exactly_like_halawi_et_al","choose_your_own_adventure","consider_the_knn", "adjust_based_on_random_forest","generate_rag_queries","short_well_or_badly_rag_added", "exactly_like_halawi_et_al_rag_added", "summarize_knn", "exactly_like_halawi_et_al_rag_added_no_knn_no_rag", "exactly_like_halawi_et_al_rag_added_forced_rf", "exactly_like_halawi_et_al_rag_added_no_knn_no_rag_forced_rf", "exactly_like_halawi_et_al_rag_added_forced_rf_with_explanation", "exactly_like_halawi_et_al_better_model_rag_added_forced_rf")

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from extracting_and_grading_helper_functions import (
    consolidate_rows_by_activity,
    loop_over_rows_to_call_model,
)
from extract_pdfs_as_txt import (
    normalized_basename,
    process_one,
)
from helpers_for_ratings_and_final_activity_features import get_ratings_text, get_rating_scale_info_from_rating_object, get_rating_scale_info, get_text_to_describe_rating_distribution, pick_start_date, compute_training_distribution_by_prefix, load_good_overall_ids

DATA_DIR = Path("../../data")
ACTIVITY_INFO_CSV = DATA_DIR / "info_for_activity_forecasting.csv"
MERGED_OVERALL_RATINGS = DATA_DIR / "merged_overall_ratings.jsonl"
RETROSPECTIVE_FORECAST_JSONL = DATA_DIR / "outputs_retrospective_forecast.jsonl"
CHATGPT_SUMMARIES_JSONL = DATA_DIR / "outputs_summaries.jsonl"
RISKS_JSONL = DATA_DIR / "outputs_risks.jsonl"
INFO_FOR_ACTIVITY_FORECASTING = '../../data/info_for_activity_forecasting_with_cpia_imputed.csv'
OUT_MISC = Path("../../data/outputs_misc.jsonl")

TXT_OUTPUT_DIR = DATA_DIR / "iati_all_pdfs_txt_format"

FEW_SHOT_KS = [1, 7, 20]
MAX_TARGET_CHARS = 999999999
MAX_NEIGHBOR_CHARS = 999999999

SIMILARITY_TOP_N = 1000
MODEL_NAME = "gemini"


def format_risks_if_listlike(risks_summary: str) -> str:
    if risks_summary is None:
        return risks_summary

    t = str(risks_summary).strip()
    if not t or t == "NO RESPONSE":
        return risks_summary

    if not (t.startswith("[") and t.endswith("]")):
        return risks_summary

    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        pass

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


def load_ml_model_preds_for_prompts(col="pred_rf_llm_modded"):
    path = DATA_DIR / "best_model_predictions.csv"
    df = pd.read_csv(path, dtype={"activity_id": str})
    if "activity_id" not in df.columns:
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str)
        return df[col].astype(float).to_dict()

    return pd.Series(df[col].astype(float).values, index=df["activity_id"].astype(str)).to_dict()


def load_stat_model_interpretations(path="../../data/stat_model_interpretations.csv"):
    df = pd.read_csv(path, dtype={"activity_id": str})
    return dict(zip(df["activity_id"].astype(str), df["interpretation"].astype(str)))



def rf_pred_label_and_number(v: float) -> tuple[str, str]:
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



def _load_risks_summaries() -> Dict[str, str]:
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

            resp = data.get("response")
            text = ""
            if isinstance(resp, dict):
                text = resp.get("content") or resp.get("text") or ""

            if not text:
                raw = (data.get("response_text") or "").strip()
                if raw:
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = raw

                    if isinstance(parsed, dict):
                        text = parsed.get("risks_summary") or json.dumps(parsed)
                    else:
                        text = str(parsed)

            text = str(text).strip()
            if not text:
                continue

            out[aid] = text

    return out

def _load_chatgpt_descriptions() -> Dict[str, str]:
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
    out: Dict[str, Dict[str, str]] = {}
    import csv
    with ACTIVITY_INFO_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            aid = (r.get("activity_id") or "").strip()
            if aid:
                out[aid] = r
    chatgpt_descriptions = _load_chatgpt_descriptions()
    for aid, desc in chatgpt_descriptions.items():
        row = out.setdefault(aid, {})
        row["chatgpt_description"] = desc

    risks_summaries = _load_risks_summaries()
    for aid, risks_text in risks_summaries.items():
        row = out.setdefault(aid, {})
        row["risks_summary"] = risks_text

    return out



def load_mock_forecasts(path: Path) -> Dict[str, str]:
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


def ensure_txt_for_pdf(pdf_path: Path, allow_ocr: bool = True) -> Path:
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
    target_aid_str = str(target_aid)

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

    primary_stats = None
    overall_stats = None

    if isinstance(rating_stats, dict) and "aid_fraction" in rating_stats:
        primary_stats = rating_stats
        overall_stats = rating_stats
        print(f"[KNN] {target_aid_str}: rating_stats is flat (no by_prefix).")
    elif isinstance(rating_stats, dict):
        overall_stats = rating_stats.get("overall")
        byp = rating_stats.get("by_prefix")

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

    if (not primary_stats or "aid_fraction" not in primary_stats) and overall_stats and "aid_fraction" in overall_stats:
        print(f"[KNN] {target_aid_str}: no prefix stats; using overall stats as primary")
        primary_stats = overall_stats

    if not overall_stats or not isinstance(overall_stats, dict) or "aid_fraction" not in overall_stats:
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

    def rating_int_from_fraction(f: float) -> int:
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

    cands_primary = build_candidates(use_overall=False)
    print(f"[KNN] {target_aid_str}: prefix candidates={len(cands_primary)}")

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


def build_few_shot_block(
    neighbor_ids: List[str],
    baseline_map: Dict[str, Dict[str, Any]],
    activity_info: Dict[str, Dict[str, str]],
    ratings: Dict[str, Dict[str, Any]],
    mock_forecasts: Dict[str, str],
    include_mock_forecast: bool = True,
) -> str:
    lines: List[str] = []
    few_shot_actual = 0
    for idx, aid in enumerate(neighbor_ids, start=1):
        meta = activity_info.get(aid, {})
        title = (meta.get("activity_title") or "").strip()
        locations = (meta.get("country_location") or "").strip()
        scope = (meta.get("activity_scope") or "").strip()
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
    return "\n".join(lines).strip()


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
    variant = variant_base + "_idx_" + str(call_idx)
    baseline_map = {str(b["activity_id"]): b for b in baseline_bundles}
    prompts: Dict[str, Dict[str, str]] = {}


    include_mock_forecast = variant_base in LIST_OF_ALL_RECENT_VARIANTS
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
        rf_preds = load_ml_model_preds_for_prompts(col="pred_rf_llm_modded")

    if "forced_rf" in variant_base:
        rf_preds = load_ml_model_preds_for_prompts(col="pred_rf_llm_modded")
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

        is_rag_variant = (variant_base == "generate_rag_queries")
        is_knn_summary = (variant_base == "summarize_knn")
        early_only = (str(aid)[:5] == "DE-1-") or (str(aid)[:4] == "DE-1")  # robust-ish

        prev_by_stage = prev_by_stage or {}
        s1_text = (prev_by_stage.get("s1", {}).get(aid) or "").strip()
        s2_text = (prev_by_stage.get("s2", {}).get(aid) or "").strip()
        s3_text = (prev_by_stage.get("s3", {}).get(aid) or "").strip()

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

        get_fewshot_directly_instead_of_knn_summary = False
        if knn_summary_by_aid and (knn_summary_by_aid.get(str(aid)) or "").strip() == "":
            # in this case, we're trying to insert knn summary by aid, but this one didn't have any few-shot to get.
            get_fewshot_directly_instead_of_knn_summary = True
            if few_shot_k > 0:
                print("WARNING: knn_summary had empty text for this aid (but attempted few-shot). Falling back to direct few-shot.")
                n_knn_empty += 1

        add_in_fewshot_examples = False
        if few_shot_k == 0:
            add_in_fewshot_examples = False
            if is_knn_summary:
                print("WARNING: could not insert knn summary: few shot k was set to zero.")
                continue
        elif get_fewshot_directly_instead_of_knn_summary:
            add_in_fewshot_examples = True
        elif is_knn_summary:
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
                if "forced_rf" in variant_base:
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
                    scratchpad_method = f"""{rating_text_dist}Here are a few reasons that you said the answer might be \"Moderately Satisfactory\" or worse:\n{s1_text}\nHere are a few reasons that you said the answer might be \"Satisfactory\" or better:\n{s2_text}\n\nYOUR TASK:\n Aggregate the your considerations above. Think like a superforecaster (e.g. Nate Silver). On the very last line of your response, write 'FORECAST: ' followed by exactly one option from this rating scale with no extra words: {options_text}.""".strip()
            elif stage == "final":
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
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
            if use_random_scratchpad:
                scratchpad_options = make_scratchpad_methods(
                    num_options=num_options,
                    midpoint_low_text=midpoint_low_text,
                    midpoint_high_text=midpoint_high_text,
                    final_result_for_prompt=final_result_for_prompt,
                )
                rating_text_dist = '\n'.join(text_to_describe_rating_distribution)
                scratchpad_method = rating_text_dist + "\nProvide the following format for your response:\n" + scratchpad_options[0]
            else:
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

        if variant_base == "summarize_knn":
            prompt_lines.append("### EXAMPLE ACTIVITIES ###")
            prompt_lines.append(few_shot_block)
            prompt_lines.append("### END EXAMPLE ACTIVITIES ###\n")
        elif knn_summary_provided and "_no_knn" not in variant_base:
            if knn_txt:
                prompt_lines.append("\n### Lessons from similar activities ###")
                prompt_lines.append(knn_txt)
                prompt_lines.append("### End lessons ###\n")
            elif few_shot_block:
                prompt_lines.append("\n### EXAMPLE ACTIVITIES ###")
                prompt_lines.append(few_shot_block)
                prompt_lines.append("### END EXAMPLE ACTIVITIES ###\n")

        prompt_lines.append("")
        prompt_lines.append(f"ACTIVITY ID: {aid}")
        if variant_base != "onlysummary_no_knn_no_rag":
            if title:
                prompt_lines.append(f"ACTIVITY TITLE: {title}")
            if planned_start_date:
                prompt_lines.append(f"ORIGINAL PLANNED START DATE: {planned_start_date}")
            if planned_end_date:
                prompt_lines.append(f"ORIGINAL PLANNED END DATE: {planned_end_date}")

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

            if activity_scope != "" and activity_scope is not None and activity_scope != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY SCOPE: {activity_scope}")
            planned_exp = planned_by_aid.get(aid)
            if planned_exp is not None:
                planned_usd = float(np.exp(planned_exp))
                prompt_lines.append(f"\nPLANNED TOTAL DISBURSEMENT (USD): {planned_usd:,.0f}")
            if locations != "" and locations is not None and locations != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY LOCATION(S): {locations}")
            if gdp_percap != "" and gdp_percap is not None and gdp_percap != "NO RESPONSE":
                try:
                    gdp_int = int(float(gdp_percap))
                    prompt_lines.append(f"\nLOCATION GDP PER CAPITA, USD: {gdp_int}")
                except ValueError:
                    prompt_lines.append(f"\nLOCATION GDP PER CAPITA, USD: {gdp_percap}")
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

        if variant_base == "onlysummary_no_knn_no_rag":
            if chatgpt_description and chatgpt_description != "NO RESPONSE":
                prompt_lines.append(f"\nACTIVITY DESCRIPTION: {chatgpt_description}")

        if ("rag_added" in variant_base and "_no_rag" not in variant_base):
            if not rag_answers_by_aid:
                    print(f"WARNING: there was no rag info injected for {aid}! skipping")
            else:
                inserted_context = (rag_answers_by_aid.get(str(aid)) or "").strip()
                if not inserted_context:
                    print(f"WARNING: skipping as rag did not return any additional information {aid}")
                else:
                    if inserted_context != "NO RESPONSE":
                        prompt_lines.append("\n### Additional specific information about the activity that you summarized ###")
                        prompt_lines.append(inserted_context)
                        prompt_lines.append("\n### End of additional information you summarized ###")

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

    print("n_knn_empty:")
    print(n_knn_empty)
    print("")
    print("")
    print("DONE CONSTRUCTING PROMPTS")
    print("")
    return prompts
