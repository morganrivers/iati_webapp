#!/usr/bin/env python3
"""
Standalone script: run the full LLM forecast pipeline for a single webapp-extracted activity.

Usage:
    python run_rag_forecast.py [path/to/projects/webapp_XXXXXXXX]

If no argument given, uses the most recently modified subdirectory of projects/.
"""

# ── KNN similarity backend ────────────────────────────────────────────────────
# False → BM25 over outputs_summaries.jsonl text (no embeddings file, low RAM).
# True  → Gemini vector embeddings from activity_text_embeddings_gemini.jsonl.
USE_VECTOR_KNN: bool = False
import os
import sys
import json
import asyncio
import pickle
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


# ── Path setup ────────────────────────────────────────────────────────────────
WEBAPP_DIR = Path(__file__).resolve().parent
C_FORECAST_DIR = WEBAPP_DIR.parent / "src" / "forecast_outcomes"
UTILS_DIR = WEBAPP_DIR.parent / "src" / "utils"
MODULES_DIR = WEBAPP_DIR / "modules"

for p in [str(C_FORECAST_DIR), str(UTILS_DIR), str(MODULES_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Core imports (same as B_generate_rag_forecasts.py) ───────────────────────

import forecast_with_few_shot as knn
from rag_bm25 import build_synthesis_prompt_from_phrasegen_text
from location_features import extract_features_from_location, get_org_dummies
from rf_predictor import impute_and_run_statistical_model
import get_similar_activities as _gsa
# forecast_with_few_shot uses relative paths (designed to run from src/forecast_outcomes/).
# Override them here so they resolve correctly when run_rag_forecast.py is the entry point.
from webapp_paths import DATA_DIR as _DATA_DIR, MODEL_DIR
knn.DATA_DIR = _DATA_DIR
knn.ACTIVITY_INFO_CSV = _DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"
knn.MERGED_OVERALL_RATINGS = _DATA_DIR / "merged_overall_ratings.jsonl"
knn.RETROSPECTIVE_FORECAST_JSONL = _DATA_DIR / "outputs_retrospective_forecast.jsonl"
knn.CHATGPT_SUMMARIES_JSONL = _DATA_DIR / "outputs_summaries.jsonl"
knn.RISKS_JSONL = _DATA_DIR / "outputs_risks.jsonl"
knn.OUT_MISC = _DATA_DIR / "outputs_misc.jsonl"
knn.INFO_FOR_ACTIVITY_FORECASTING = str(_DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv")

# ── SQLite loaders: replace CSV+JSONL loaders with lightweight SQLite loaders ─
_DB_PATH = _DATA_DIR / "webapp.db"
assert _DB_PATH.exists(), f"webapp.db not found at {_DB_PATH}"

from sqlite_loaders import (
    load_activity_info_sqlite,
    make_lazy_mock_forecasts,
    load_bm25_corpus_sqlite,
    load_activity_dataframe_sqlite,
)

# get_similar_activities also uses relative paths; patch them too.
_gsa.EMBEDDINGS_PATH = _DATA_DIR / "activity_text_embeddings_gemini.jsonl"
_gsa.BM25_SUMMARIES_PATH = _DATA_DIR / "outputs_summaries.jsonl"
_gsa.CSV_PATH = str(knn.ACTIVITY_INFO_CSV)




def _compute_gemini_embedding(text: str) -> np.ndarray:
    """Return an L2-normalised Gemini embedding vector for text (768-dim)."""
    from google import genai as _genai
    from google.genai import types as _genai_types
    from llm_tracing import wrap_genai_client
    client = wrap_genai_client(_genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=_genai_types.EmbedContentConfig(output_dimensionality=768),
    )
    emb = np.array(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def _build_activity_dataframe(activity_id: str, app_state: dict) -> pd.DataFrame:
    """SQLite activity dataframe (add_derived_columns) plus a synthetic row for
    the webapp activity so it can act as the similarity query."""
    df = _gsa.add_derived_columns(load_activity_dataframe_sqlite(_DB_PATH))
    if activity_id in set(df["activity_id"].astype(str)):
        return df

    confirmed = app_state.get("confirmed_metadata", {})
    widget = app_state.get("widget_state", {})
    start_str = widget.get("input_start_date", "2020-01-01")
    end_str = widget.get("input_planned_end_date", "2025-01-01")
    try:
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
    except Exception:
        start_dt, end_dt = pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-01")

    synthetic = {col: None for col in df.columns}
    synthetic["activity_id"] = activity_id
    synthetic["start_date"] = start_dt
    synthetic["end_date"] = end_dt
    synthetic["activity_title"] = confirmed.get("title", "")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat([df, pd.DataFrame([synthetic])], ignore_index=True)


def _load_embeddings_with_webapp(activity_id: str, activity_text: str) -> dict:
    """Load Gemini embeddings and ensure the webapp activity has a vector."""
    import sqlite3 as _sqlite3
    with _sqlite3.connect(str(_DB_PATH)) as _c:
        _db_has_embeddings = bool(
            _c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='embeddings'"
            ).fetchone()
        )
    if _db_has_embeddings:
        embs = _gsa.load_activity_embeddings_sqlite(_DB_PATH)
        print(f"[KNN] Loaded {len(embs)} embeddings from SQLite.")
    else:
        embs = _gsa.load_activity_embeddings(_gsa.EMBEDDINGS_PATH)
        print(f"[KNN] Loaded {len(embs)} embeddings from JSONL.")

    if activity_id not in embs:
        if activity_text:
            try:
                embs[activity_id] = _compute_gemini_embedding(activity_text)
                print(f"[KNN] Computed Gemini embedding for {activity_id}.")
            except Exception as e:
                print(f"[KNN] Gemini embedding failed ({e}); falling back to mean vector.")
                activity_text = ""
        if not activity_text:
            vecs = np.stack(list(embs.values()), axis=0)
            mean_vec = vecs.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            embs[activity_id] = mean_vec
            print(f"[KNN] No text for {activity_id}; using mean vector fallback.")
    return embs


def _build_similarity_fn(activity_id: str, app_state: dict, activity_dir: Path):
    """
    Build a bound similarity function for the webapp activity via dependency
    injection (no monkeypatching). Returns a callable with signature
    similarity_fn(target_aid, top_n, allowed_ids) -> (result_df, query_row),
    matching what get_knn_neighbors expects.

    USE_VECTOR_KNN=False (default): BM25 over the SQLite summaries corpus.
    USE_VECTOR_KNN=True: Gemini vector embeddings.
    """
    df = _build_activity_dataframe(activity_id, app_state)

    activity_text = _read_jsonl_field(activity_dir, "summary.jsonl", "chatgpt_description")
    if not activity_text:
        confirmed = app_state.get("confirmed_metadata", {})
        activity_text = confirmed.get("title", "") or ""

    if not USE_VECTOR_KNN:
        corpus = load_bm25_corpus_sqlite(_DB_PATH)
        print(f"[KNN] BM25 mode: corpus {len(corpus)} activities, query {len(activity_text)} chars.")
        return functools.partial(
            _gsa.find_similar_activities_bm25,
            df=df,
            corpus=corpus,
            query_text=activity_text,
        )

    embs = _load_embeddings_with_webapp(activity_id, activity_text)
    return functools.partial(
        _gsa.find_similar_activities_semantic,
        df=df,
        embeddings=embs,
    )

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "variant": "exactly_like_halawi_et_al_better_model_rag_added_forced_rf",
    "tag": "deepseek_val",
    "tag_for_s3": "deepseek_val_forced_rf",
    "stages_to_run": ["s1", "s2", "s3"],
}

ENSEMBLE_CALLS = 1
FEWSHOT_K = 3

# Standard 6-option WB-style rating scale used as fallback for the webapp activity
# (which has no ground-truth rating yet). Constructed to match what
# get_rating_scale_info_from_rating_object returns for a midpoint WB rating.
DEFAULT_RATING_SCALE = {
    "num_options": 6,
    "midpoint_low_text": "Moderately Unsatisfactory",
    "midpoint_high_text": "Moderately Satisfactory",
    "options_text": (
        "'Highly Unsatisfactory', 'Unsatisfactory', 'Moderately Unsatisfactory', "
        "'Moderately Satisfactory', 'Satisfactory', 'Highly Satisfactory'"
    ),
    "final_result_for_prompt": "Moderately Satisfactory",
    "rating_value_raw": None,
    "rating_min": None,
    "rating_max": None,
    "numeric_rating": None,
    "fraction": None,
}

# ── Output dirs ───────────────────────────────────────────────────────────────
LLM_FORECASTS_DIR = WEBAPP_DIR / "llm_forecasts"
LLM_FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

# RAG index persist root (within llm_forecasts so everything stays together)
RAG_PERSIST_ROOT = LLM_FORECASTS_DIR / "rag_indexes"
RAG_PERSIST_ROOT.mkdir(parents=True, exist_ok=True)

# TXT output dir for PDF→text conversion
TXT_OUTPUT_DIR = LLM_FORECASTS_DIR / "pdf_txt"
TXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers (mirrors B_generate_rag_forecasts.py) ─────────────────────────────

def dump_prompts_jsonl(path: Path, prompts: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for aid in sorted(prompts.keys()):
            obj = {"activity_id": aid, **prompts[aid]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_outputs_jsonl(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            aid = str(data.get("activity_id") or "")
            if not aid:
                continue
            resp = data.get("response")
            txt = ""
            if isinstance(resp, dict):
                txt = resp.get("content") or resp.get("text") or ""
            if not txt:
                txt = data.get("response_text") or ""
            if txt:
                out[aid] = str(txt)
    return out


# ── Bundle builder ────────────────────────────────────────────────────────────

def _read_jsonl_field(activity_dir: Path, filename: str, key: str) -> str:
    filepath = activity_dir / filename
    if not filepath.exists():
        return ""
    with filepath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            return data.get(key, "")
    return ""


def load_webapp_bundle(activity_dir: Path, activity_id: str) -> dict:
    """Build a bundle dict from webapp extracted JSONL files."""
    bundle = {"activity_id": activity_id, "section": "Outcome"}

    bundle["activity_summary"] = _read_jsonl_field(activity_dir, "summary.jsonl", "chatgpt_description")
    bundle["activity_context"] = _read_jsonl_field(activity_dir, "context.jsonl", "response_text")
    bundle["implementer_performance_text"] = _read_jsonl_field(activity_dir, "implementer_performance.jsonl", "response_text")
    bundle["targets_summary"] = _read_jsonl_field(activity_dir, "targets.jsonl", "response_text")
    bundle["risks_summary"] = _read_jsonl_field(activity_dir, "risks.jsonl", "response_text")
    bundle["finance_summary"] = _read_jsonl_field(activity_dir, "finance_qualitative.jsonl", "response_text")

    # misc.jsonl has structured JSON in response_text
    misc_path = activity_dir / "misc.jsonl"
    if misc_path.exists():
        with misc_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rt = data.get("response_text", "")
                try:
                    parsed = json.loads(rt)
                    bundle["complexity_details"] = parsed.get("complexity_details", "")
                    bundle["how_integrated_description"] = parsed.get("how_integrated_description", "")
                    bundle["disbursement_total"] = parsed.get("disbursement_total", "")
                    bundle["disbursement_units"] = parsed.get("disbursement_units", "")
                    bundle["loan_total"] = parsed.get("loan_total", "")
                    bundle["loan_units"] = parsed.get("loan_units", "")
                    bundle["is_RCT"] = parsed.get("is_RCT", "")
                except (json.JSONDecodeError, TypeError):
                    bundle["complexity_details"] = rt
                    bundle["how_integrated_description"] = rt
                break

    return bundle


# ── Activity info entry ───────────────────────────────────────────────────────

def build_webapp_activity_info(activity_dir: Path, activity_id: str, app_state: dict) -> dict:
    """Build an activity_info dict entry for the webapp activity."""
    metadata = {}
    metadata_path = activity_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

    confirmed = app_state.get("confirmed_metadata", {})
    widget = app_state.get("widget_state", {})

    chatgpt_description = _read_jsonl_field(activity_dir, "summary.jsonl", "chatgpt_description")
    risks_text = _read_jsonl_field(activity_dir, "risks.jsonl", "response_text")

    return {
        "activity_title": confirmed.get("title") or metadata.get("title", ""),
        "reporting_orgs": widget.get("select_reporting_org", ""),
        "country_location": confirmed.get("country_location") or metadata.get("country_location", ""),
        "activity_scope": str(widget.get("input_activity_scope", "")),
        "gdp_percap": "",  # will be computed by RF pred; not needed for prompt text
        "implementing_org_type": confirmed.get("implementing_org_type") or metadata.get("implementing_org_type", ""),
        "original_planned_start_date": confirmed.get("planned_start_date") or metadata.get("planned_start_date", ""),
        "original_planned_close_date": confirmed.get("planned_end_date") or metadata.get("planned_end_date", ""),
        "chatgpt_description": chatgpt_description,
        "risks_summary": risks_text,
    }




# ── Main pipeline ─────────────────────────────────────────────────────────────
# After imports load (before main() body starts)

def main(activity_dir_override=None):
    """Run the full LLM forecast pipeline.

    Parameters
    ----------
    activity_dir_override:
        When called in-process from the webapp, pass the Path to the activity
        directory directly instead of relying on sys.argv.
    """

    print("="*80, flush=True)
    print("NARRATIVE FORECAST PIPELINE STARTING", flush=True)
    print("="*80, flush=True)

    # Resolve activity directory
    if activity_dir_override is not None:
        activity_dir = Path(activity_dir_override).resolve()
    elif len(sys.argv) >= 2:
        activity_dir = Path(sys.argv[1]).resolve()
    else:
        from webapp_paths import PROJECTS_DIR
        base = PROJECTS_DIR
        candidates = [p for p in base.iterdir() if p.is_dir()]
        if not candidates:
            raise RuntimeError(f"no extracted activity directories found under {base}")
        activity_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"No argument given, using most recent: {activity_dir}", flush=True)

    activity_id = activity_dir.name
    pdf_path = activity_dir / "uploaded.pdf"
    print(f"\n[INIT] Activity ID: {activity_id}", flush=True)
    print(f"[INIT] PDF path: {pdf_path}", flush=True)
    print(f"[INIT] PDF exists: {pdf_path.exists()}", flush=True)

    app_state_path = activity_dir / "app_state.json"
    if not app_state_path.exists():
        raise RuntimeError(f"app_state.json not found at {app_state_path}")
    with app_state_path.open(encoding="utf-8") as f:
        app_state = json.load(f)
    print(f"[INIT] ✓ Loaded app_state.json", flush=True)

    # ── Build bundle and row list ─────────────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"[STAGE 0/7] BUILDING ACTIVITY BUNDLE", flush=True)
    print(f"{'='*80}", flush=True)
    bundle = load_webapp_bundle(activity_dir, activity_id)
    rows = [bundle]
    print(f"[STAGE 0/7] ✓ Bundle created with {len(bundle)} fields", flush=True)

    # ── Load core data (same as B_generate_rag_forecasts.py) ─────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"[STAGE 1/7] LOADING REFERENCE DATA", flush=True)
    print(f"[STAGE 1/7] Estimated time: ~10 seconds", flush=True)
    print(f"{'='*80}", flush=True)
    activity_info = load_activity_info_sqlite(_DB_PATH)
    print(f"[STAGE 1/7] ✓ Loaded {len(activity_info)} activities", flush=True)
    ratings = knn.load_good_overall_ids(knn.MERGED_OVERALL_RATINGS)
    print(f"[STAGE 1/7] ✓ Loaded {len(ratings)} ratings", flush=True)
    mock_forecasts = make_lazy_mock_forecasts(_DB_PATH)
    print(f"[STAGE 1/7] ✓ Loaded {len(mock_forecasts)} mock forecasts", flush=True)
    rating_stats = knn.compute_training_distribution_by_prefix(ratings)
    print(f"[STAGE 1/7] ✓ Computed rating statistics", flush=True)

    # ── Inject webapp activity into activity_info ─────────────────────────────
    activity_info[activity_id] = build_webapp_activity_info(activity_dir, activity_id, app_state)
    print(f"[STAGE 1/7] ✓ Injected webapp activity into reference data", flush=True)

    # ── Build injected similarity function for the webapp activity ────────────
    similarity_fn = _build_similarity_fn(activity_id, app_state, activity_dir)
    print(f"[STAGE 1/7] ✓ Built similarity function for webapp activity", flush=True)

    # ── Compute RF prediction and prepare extra_rf_preds ─────────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"[STAGE 2/7] COMPUTING RANDOM FOREST PREDICTION", flush=True)
    print(f"[STAGE 2/7] Estimated time: ~2 seconds", flush=True)
    print(f"{'='*80}", flush=True)
    _base_path = MODEL_DIR
    with open(_base_path / "model.pkl", "rb") as f:
        _rf_model = pickle.load(f)
    with open(_base_path / "extra_model.pkl", "rb") as f:
        _extra_model = pickle.load(f)
    with open(_base_path / "per_org_baseline.json") as f:
        _per_org_baseline = json.load(f)
    with open(_base_path / "start_year_correction.json") as f:
        _start_year_correction = json.load(f)
    with open(_base_path / "train_medians.json") as f:
        _train_medians = json.load(f)
    with open(_base_path / "feature_names.json") as f:
        _feature_names = json.load(f)
    _model_metadata = {"train_medians": _train_medians, "feature_names": _feature_names}
    _widget = app_state.get("widget_state", {})
    _session_state_values = {
        **_widget,
        "sector_percentages": app_state.get("sector_percentages", {}),
        "embedding_results": app_state.get("embedding_results", {}),
    }
    _location_str = (app_state.get("confirmed_metadata", {}).get("country_location")
                     or app_state.get("extracted_values", {}).get("location")) or ""
    if not _location_str:
        _lc = app_state.get("location_countries", [])
        if _lc:
            _location_str = "|".join(f"{c['code']}|{int(c['pct'])}" for c in _lc)
    _start_date_str = _widget.get("input_start_date", "2020-01-01")
    _location_features = extract_features_from_location(_location_str, start_date=_start_date_str) if _location_str else {}
    _reporting_org = _widget.get("select_reporting_org", "World Bank")
    _gdp_percap_input = _widget.get("input_gdp_percap", 0)
    _, _, _, _, rf_pred, _ = impute_and_run_statistical_model(
        _rf_model, _extra_model, _per_org_baseline, _start_year_correction, _model_metadata,
        _reporting_org, _gdp_percap_input,
        app_state.get("field_edited", {}), _session_state_values, _location_features)
    print(f"[STAGE 2/7] ✓ RF prediction: {rf_pred:.3f}", flush=True)
    extra_rf_preds = {activity_id: rf_pred}

    # ── Pipeline ──────────────────────────────────────────────────────────────
    execpool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="genai")
    variant = CFG["variant"]

    try:
        for call_idx in range(ENSEMBLE_CALLS):
            tag = CFG["tag"]
            tag_for_s3 = CFG["tag_for_s3"]

            # ── KNN SUMMARY ──────────────────────────────────────────────────
            print(f"\n{'='*80}", flush=True)
            print(f"[STAGE 3/7] KNN SUMMARY (call {call_idx+1}/{ENSEMBLE_CALLS})", flush=True)
            print(f"[STAGE 3/7] Finding {FEWSHOT_K} similar activities and generating summary", flush=True)
            print(f"[STAGE 3/7] Estimated time: ~15-30 seconds", flush=True)
            print(f"{'='*80}", flush=True)
            out_knn_summary = LLM_FORECASTS_DIR / f"outputs_knn_summary_{tag}_call_1.jsonl"
            knn_summary_prompts = knn.build_prompts_with_few_shot(
                baseline_bundles=rows,
                activity_info=activity_info,
                ratings=ratings,
                mock_forecasts=mock_forecasts,
                few_shot_k=FEWSHOT_K,
                variant_base="summarize_knn",
                similarity_fn=similarity_fn,
                rating_stats=rating_stats,
                call_idx=call_idx + 1,
                default_rating_scale=DEFAULT_RATING_SCALE,
            )
            dump_prompts_jsonl(
                LLM_FORECASTS_DIR / f"dryrun_knn_summary_{tag}_call_{call_idx+1}.jsonl",
                knn_summary_prompts,
            )
            rows_for_knn = [b for b in rows if b["activity_id"] in knn_summary_prompts]
            for attempt in range(2):
                print(f"[STAGE 3/7] 🌐 API CALL {attempt+1}/2: Calling DeepSeek Reasoner for KNN summary...", flush=True)
                asyncio.run(knn.loop_over_rows_to_call_model(
                    str(out_knn_summary), rows_for_knn, knn_summary_prompts,
                    response_schema=None, execpool=execpool, model="deepseek-reasoner",
                ))
                print(f"[STAGE 3/7] ✓ API call {attempt+1}/2 completed", flush=True)

            knn_summary_by_aid = read_outputs_jsonl(out_knn_summary)
            print(f"[STAGE 3/7] ✓ KNN summary complete: {len(knn_summary_by_aid)} activities", flush=True)

            # ── PHRASEGEN ────────────────────────────────────────────────────
            print(f"\n{'='*80}", flush=True)
            print(f"[STAGE 4/7] PHRASE GENERATION (call {call_idx+1}/{ENSEMBLE_CALLS})", flush=True)
            print(f"[STAGE 4/7] Generating search queries for RAG retrieval", flush=True)
            print(f"[STAGE 4/7] Estimated time: ~15-30 seconds", flush=True)
            print(f"{'='*80}", flush=True)
            out_phrasegen = LLM_FORECASTS_DIR / f"outputs_phrasegen_{tag}_call_1.jsonl"

            phrasegen_prompts = knn.build_prompts_with_few_shot(
                baseline_bundles=rows,
                activity_info=activity_info,
                ratings=ratings,
                mock_forecasts=mock_forecasts,
                few_shot_k=FEWSHOT_K,
                variant_base="generate_rag_queries",
                similarity_fn=similarity_fn,
                rating_stats=rating_stats,
                call_idx=call_idx + 1,
                knn_summary_by_aid=knn_summary_by_aid,
                default_rating_scale=DEFAULT_RATING_SCALE,
            )
            dump_prompts_jsonl(
                LLM_FORECASTS_DIR / f"dryrun_phrasegen_{variant}_{tag}_call_{call_idx+1}.jsonl",
                phrasegen_prompts,
            )
            rows_for_phrasegen = [b for b in rows if b["activity_id"] in phrasegen_prompts]
            for attempt in range(2):
                print(f"[STAGE 4/7] 🌐 API CALL {attempt+1}/2: Calling DeepSeek Reasoner for phrase generation...", flush=True)
                asyncio.run(knn.loop_over_rows_to_call_model(
                    str(out_phrasegen), rows_for_phrasegen, phrasegen_prompts,
                    response_schema=None, execpool=execpool, model="deepseek-reasoner",
                ))
                print(f"[STAGE 4/7] ✓ API call {attempt+1}/2 completed", flush=True)

            phrasegen_by_aid = read_outputs_jsonl(out_phrasegen)
            print(f"[STAGE 4/7] ✓ Phrase generation complete: {len(phrasegen_by_aid)} activities", flush=True)

            # ── RAG SYNTHESIS ─────────────────────────────────────────────────
            print(f"\n{'='*80}", flush=True)
            print(f"[STAGE 5/7] RAG SYNTHESIS (call {call_idx+1}/{ENSEMBLE_CALLS})", flush=True)
            print(f"[STAGE 5/7] Retrieving relevant PDF evidence and generating synthesis", flush=True)
            print(f"[STAGE 5/7] Estimated time: ~30-60 seconds (includes PDF indexing)", flush=True)
            print(f"{'='*80}", flush=True)
            out_raganswers = LLM_FORECASTS_DIR / f"outputs_raganswers_variant_{variant}_{tag}_call_1.jsonl"
            out_synth_prompts = LLM_FORECASTS_DIR / f"dryrun_prompts_rag_synthesis_{variant}_call_{call_idx+1}.jsonl"

            synth_prompts = {}
            synth_rows = []
            for b in rows:
                aid = str(b["activity_id"])
                if aid not in phrasegen_by_aid:
                    print(f"WARNING: no phrasegen output for {aid}, skipping RAG synthesis")
                    continue
                phrasegen_text = phrasegen_by_aid[aid]
                print(f"[STAGE 5/7] Building RAG synthesis prompt for {aid}...", flush=True)
                print(f"[STAGE 5/7] ├─ Indexing PDF and retrieving relevant passages...", flush=True)
                diag, (sysmsg, usermsg), phrases, phrase_to_snips = build_synthesis_prompt_from_phrasegen_text(
                    activity_id=aid,
                    phrasegen_text=phrasegen_text,
                    docs_log_csv=knn.DATA_DIR / "activity_docs_log_final_restrictive.csv",
                    txt_output_dir=TXT_OUTPUT_DIR,
                    persist_root=RAG_PERSIST_ROOT,
                    allow_ocr=True,
                    top_k=6,
                    override_pdf_paths=[pdf_path],
                )
                if not sysmsg or not usermsg:
                    print(f"WARNING: RAG synthesis prompt empty for {aid} (no docs or parse fail); skipping")
                    continue
                synth_prompts[aid] = {
                    "system_msg": sysmsg,
                    "prompt": usermsg,
                    "prompt_type": f"rag_synthesis_{variant}_call_{call_idx+1}",
                }
                print(f"[STAGE 5/7] ✓ RAG prompt built successfully", flush=True)
                synth_rows.append(b)

            dump_prompts_jsonl(out_synth_prompts, synth_prompts)
            print(f"[STAGE 5/7] ✓ Saved {len(synth_prompts)} synthesis prompts", flush=True)
            for attempt in range(2):
                print(f"[STAGE 5/7] 🌐 API CALL {attempt+1}/2: Calling DeepSeek Reasoner for RAG synthesis...", flush=True)
                asyncio.run(knn.loop_over_rows_to_call_model(
                    str(out_raganswers), synth_rows, synth_prompts,
                    response_schema=None, execpool=execpool, model="deepseek-reasoner",
                ))
                print(f"[STAGE 5/7] ✓ API call {attempt+1}/2 completed", flush=True)

            rag_answers_by_aid = read_outputs_jsonl(out_raganswers)
            print(f"[STAGE 5/7] ✓ RAG synthesis complete: {len(rag_answers_by_aid)} activities", flush=True)

            # ── MULTI-STAGE FORECAST: s1, s2, s3 ─────────────────────────────
            prev_by_stage = {}
            rows_current = rows

            for stage in CFG["stages_to_run"]:
                stage_num = {"s1": 6, "s2": 7, "s3": 8}.get(stage, 6)
                stage_name = {"s1": "Why it might go badly", "s2": "Why it might go well", "s3": "Final forecast"}.get(stage, stage)
                print(f"\n{'='*80}", flush=True)
                print(f"[STAGE {stage_num}/8] {stage.upper()}: {stage_name}", flush=True)
                print(f"[STAGE {stage_num}/8] Generating narrative forecast (call {call_idx+1}/{ENSEMBLE_CALLS})", flush=True)
                print(f"[STAGE {stage_num}/8] Estimated time: ~20-40 seconds", flush=True)
                print(f"{'='*80}", flush=True)

                stage_prompts = knn.build_prompts_with_few_shot(
                    baseline_bundles=rows_current,
                    activity_info=activity_info,
                    ratings=ratings,
                    mock_forecasts=mock_forecasts,
                    few_shot_k=FEWSHOT_K,
                    variant_base=variant,
                    similarity_fn=similarity_fn,
                    rating_stats=rating_stats,
                    call_idx=call_idx + 1,
                    rag_answers_by_aid=rag_answers_by_aid,
                    knn_summary_by_aid=knn_summary_by_aid,
                    stage=stage,
                    prev_by_stage=prev_by_stage,
                    extra_rf_preds=extra_rf_preds,
                    default_rating_scale=DEFAULT_RATING_SCALE,
                )

                out_prompts_path = (
                    LLM_FORECASTS_DIR
                    / f"dryrun_prompts_{variant}_{tag_for_s3}_{stage}_call_{call_idx+1}.jsonl"
                )
                dump_prompts_jsonl(out_prompts_path, stage_prompts)

                rows_for_stage = [b for b in rows_current if b["activity_id"] in stage_prompts]

                out_stage = (
                    LLM_FORECASTS_DIR
                    / f"outputs_{variant}_{tag_for_s3}_{stage}_call_{call_idx+1}.jsonl"
                )

                model_for_stage = "deepseek-reasoner"
                stage_num = {"s1": 6, "s2": 7, "s3": 8}.get(stage, 6)
                print(f"[STAGE {stage_num}/8] Processing {len(rows_for_stage)} activities with model={model_for_stage}", flush=True)
                for attempt in range(2):
                    print(f"[STAGE {stage_num}/8] 🌐 API CALL {attempt+1}/2: Calling DeepSeek Reasoner for {stage}...", flush=True)
                    asyncio.run(knn.loop_over_rows_to_call_model(
                        str(out_stage), rows_for_stage, stage_prompts,
                        response_schema=None, execpool=execpool, model=model_for_stage,
                    ))
                    print(f"[STAGE {stage_num}/8] ✓ API call {attempt+1}/2 completed", flush=True)
                print(f"[STAGE {stage_num}/8] ✓ Wrote {stage} outputs: {out_stage.name}", flush=True)

                prev_by_stage[stage] = read_outputs_jsonl(out_stage)
                rows_current = rows_for_stage

            print(f"\n{'='*80}", flush=True)
            print(f"✅ PIPELINE COMPLETE!", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"All outputs saved to: {LLM_FORECASTS_DIR}", flush=True)
            if activity_id in prev_by_stage.get("s3", {}):
                print(f"\n{'='*80}", flush=True)
                print(f"FINAL FORECAST for {activity_id}:", flush=True)
                print(f"{'='*80}", flush=True)
                print(prev_by_stage["s3"][activity_id], flush=True)
                print(f"{'='*80}", flush=True)

    finally:
        execpool.shutdown(wait=False, cancel_futures=True)

