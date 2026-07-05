#!/usr/bin/env python3
"""
PDF batch scorer with Gemini structured output (file-only uploads).

- Loads activity docs from FINAL_CSV and activity objects from SUBSET_JSON.
- For each of 10 random activities (seed=42) that have at least one usable
  Baseline and one usable Outcome PDF, picks the first 3 PDFs per section
  by _doc_index_int DESC and scores them in page-batches.

Batching rules (per PDF length):
  Always 3 pages per batch (last batch may be 1–2 pages)

Scoring:
  - Baseline PDFs: score on all 8 Intervention criteria.
  - Outcome PDFs:  score on all 8 Evaluation criteria.
  - Uses Gemini structured output with a response schema (top-level scratchpad + per-page items).
  - File-only upload: NO extracted page text included.
  - Prints the prompt being sent to the model.
  - Pauses with input("hit enter to add csv row") before writing each row.

CSV:
  - Appends one row per page to ../../data/pdf_relevance_scores.csv
  - Always includes all 16 possible score columns (8 Intervention + 8 Evaluation).
  - Adds a "scratchpad" column.
  - Crashes if JSON parsing fails (structured output must succeed).

Token counts:
  - input_token_count computed via count_tokens on [prompt, file].
  - output_token_count taken from response.usage_metadata if available; otherwise
    attempts count_tokens on response.text; if unavailable, set to -1.

Requirements:
  micromamba install google-genai pypdf pydantic

Environment:
  GOOGLE_API_KEY (or GOOGLE_API_KEY_GEMINI) must be set.

Notes:
  - No fancy flags. Hardcoded file locations to match builder layout.
  - Minimal reuse from prior example; this is a fresh script.
"""

import os
import csv
import sys

import json
import html
import tempfile
import shutil
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
import asyncio

from pypdf import PdfReader, PdfWriter

# --- Google GenAI (Gemini) ---
from google import genai
import pprint
# ---------- Fixed paths / constants ----------
FINAL_CSV         = "../../data/activity_docs_log_final_restrictive.csv"
SUBSET_JSON       = "../../data/subset_results.json"
LOCATION_PDFS     = "../../data/iati_all_pdfs"
OUTPUT_CSV        = "../../data/pdf_relevance_scores.csv"

SEED              = 42
N_ACTIVITIES      = 200
MAX_PICK_PER_SEC  = 3
MODEL_NAME        = "gemini-2.5-flash"
TIMEOUT_SECONDS = 300

# Activity resume/skip helper (kept simple and crash-safe)
ACT_COUNTS_CSV    = "../../data/activity_counts.csv"
MAX_PER_ACTIVITY  = 1  # threshold per activity across runs


UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from prompt_bundle_pdf import open_with_evince, make_string

# ---------- Scoring criteria ----------
# INTERVENTION_DICT = {
#     "description_of_key_activities": "The description of the key activities and context of the project.",
#     "how_invested_are_partners": "How invested in the project are the partners in the developing countries? Do they feel ownership over the project?",
#     "implementer_skill_level": "How capable is the partner government or implementing organization at delivering successful projects?",
#     "larger_project_integration": "Whether the project is integrated into a larger program",
#     "riskiness_and_risks": "How risky the project is and what the risks are",
#     "expected_outcomes_future": "Prediction of expected or targeted outcomes of intervention",
#     "theory_of_change": "The theory of change or strategy of the project",
# }

INTERVENTION_DICT = {
    "predicting_outcomes": "Whether the description is helpful for predicting future activity outcomes",
    "description_of_key_activities": "Whether key activities and context of the project are described",
    "implementer_skill_level": "Information pertaining to the skill or experience level of the partner government or implementing organization at delivering successful projects",
    "how_invested_are_partners": "Indications or direct statements about how emotionally invested and the degree of ownership the partners in the developing countries have.",
    "larger_project_integration": "Information on whether the project is integrated into a larger program",
    "riskiness_and_risks": "Descriptions of how high or low risk the project is and what the risks are",
    "expected_outcomes_future": "Explicit predictions or targets for outcomes of the future activities",
    "theory_of_change": "The theory of change or strategy of the project",
}

# INTERVENTION_DICT_STRING = json.dumps(INTERVENTION_DICT, indent=2, ensure_ascii=False)
INTERVENTION_DICT_STRING = make_string(INTERVENTION_DICT)#json.dumps(INTERVENTION_DICT, indent=2, ensure_ascii=False)

EVALUATION_DICT = {
    "evaluation": "Whether what happened during the activity is being described",
    "quantitative_outcomes": "Whether the page contains quantitative outcomes of the intervention",
    "qualitative_outcomes": "Whether the page contains qualitative description of intervention outcomes",
    "qualitative_degree_of_success": "Whether the page contains descriptions of the degree of success of the project",
    "quantitative_degree_of_success": "Whether a quantitative score or intercomparable category for how successful or unsuccessful the project was is present",
    "reasons_for_failure_or_issues": "Whether reasons that the project went wrong are described",
    "deviation_from_expectations": "Whether ways that outcomes differed from expectations are described",
    "expected_outcomes_original": "Whether the original targets for outcomes of the activity are described",
}

# EVALUATION_DICT_STRING = json.dumps(EVALUATION_DICT, indent=2, ensure_ascii=False)
EVALUATION_DICT_STRING = make_string(EVALUATION_DICT)


# Always include all 13 columns in the CSV (unused ones blank per row)
INTERVENTION_KEYS_DESCRIPTION = [key + "_description" for key in INTERVENTION_DICT.keys()]
INTERVENTION_KEYS_SCORES = [key + "_informativeness" for key in INTERVENTION_DICT.keys()]
EVALUATION_KEYS_DESCRIPTION   = [key + "_description" for key in EVALUATION_DICT.keys()]
EVALUATION_KEYS_SCORES   = [key + "_informativeness" for key in EVALUATION_DICT.keys()]
INTERVENTION_KEYS = list(INTERVENTION_DICT.keys())
INTERVENTION_KEYS = list(INTERVENTION_DICT.keys())
EVALUATION_KEYS   = list(EVALUATION_DICT.keys())
EVALUATION_KEYS   = list(EVALUATION_DICT.keys())

# ---------- Page batching rules ----------
def batch_size_for_pages(pages_total: int) -> int:
    # Always score 3 pages per batch; the last batch may be a partial (1-2 pages)
    return 3

# ---------- Basic PDF usability checks ----------
def is_real_pdf_header(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(1024)
    except Exception:
        return False
    if len(head) < 5:
        return False
    return head[:5] == b"%PDF-"

def has_usable_pdf(cached_rel: str) -> bool:
    if not cached_rel:
        return False
    path = os.path.join(LOCATION_PDFS, cached_rel)
    if not (os.path.exists(path)):
        return False
    if not is_real_pdf_header(path):
        return False
    try:
        PdfReader(path)
        return True
    except Exception:
        return False

# ---------- Load docs & acts ----------
def _activity_id(act) -> Optional[str]:
    # Prefer the hyphenated key used by IATI dumps; fall back to underscore variant
    return (
        _first_text(act, "iati-identifier")
        or _first_text(act, "iati_identifier")
        or None
    )

def load_docs(csv_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns:
      activities: dict[activity_id] -> {
        'title': str,
        'sections': {'Baseline': [rows...], 'Outcome': [rows...]}}
      order: list of activity_ids in first-seen order
    Each row includes _doc_index_int.
    """
    activities: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            aid = r.get("activity_id", "") or r.get("iati_identifier", "")
            if not aid:
                continue
            if aid not in activities:
                activities[aid] = {
                    "title": r.get("activity_title", "") or "",
                    "sections": {"Baseline": [], "Outcome": []},
                }
                order.append(aid)
            sec = r.get("section", "Baseline")
            if sec not in ("Baseline", "Outcome"):
                activities[aid]["sections"].setdefault(sec, [])
            try:
                r["_doc_index_int"] = int(r.get("doc_index", "0"))
            except Exception:
                r["_doc_index_int"] = 0
            activities[aid]["sections"].setdefault(sec, []).append(r)

    # pre-sort ascending; we'll select by DESC later
    for aid, obj in activities.items():
        for sec, rows in obj["sections"].items():
            rows.sort(key=lambda x: x["_doc_index_int"])
    return activities, order

def filter_usable(rows: List[dict]) -> List[dict]:
    out = []
    for d in rows:
        try:
            pages = int((d.get("pages", "") or "0").strip())
        except Exception:
            pages = 0
        if pages <= 0:
            continue
        if not has_usable_pdf(d.get("cached_file", "").strip()):
            continue
        out.append(d)
    return out

# ---------- Activity metadata helpers (from your snippet) ----------
def _first_text(obj, *keys):
    cur = obj
    for k in keys:
        cur = (cur or {}).get(k, [])
        if isinstance(cur, list) and cur:
            cur = cur[0]
    if isinstance(cur, dict):
        return cur.get("text()")
    return cur if isinstance(cur, str) else None

def activity_title(act): 
    return _first_text(act, "title", "narrative") or ""

def desc(act):
    return html.unescape(_first_text(act, "description", "narrative") or "")

def load_acts_map(subset_json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a map: activity_id -> activity object, from a subset_results.json
    that is a list of rows, each containing an "iati_json" string.
    """
    if not os.path.exists(subset_json_path):
        return {}

    with open(subset_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        iati_blob = row.get("iati_json")
        if not iati_blob:
            continue
        try:
            iati = json.loads(iati_blob)
        except Exception:
            continue

        acts = iati.get("iati-activity", [])
        if isinstance(acts, dict):
            acts = [acts]

        for act in acts:
            aid = _activity_id(act) or row.get("iati_identifier", "")
            if not aid:
                continue
            # Index by exact id and a lowercase alias (helps if CSV uses different casing/whitespace)
            out[aid] = act
            out[aid.strip().lower()] = act

    return out
# # ---------- Temp PDF slicing ----------
# def write_pdf_slice(src_path: str, start_idx: int, end_idx_excl: int, tmpdir: str) -> str:
#     """
#     Create a temp PDF containing pages [start_idx, end_idx_excl) from src_path.
#     Returns the path to the sliced file.
#     """
#     reader = PdfReader(src_path)
#     writer = PdfWriter()
#     for i in range(start_idx, min(end_idx_excl, len(reader.pages))):
#         writer.add_page(reader.pages[i])
#     base = os.path.basename(src_path)
#     out_path = os.path.join(tmpdir, f"{Path(base).stem}_p{start_idx+1}-{end_idx_excl}.pdf")
#     with open(out_path, "wb") as f:
#         writer.write(f)
#     return out_path

def write_pdf_slice(src_path: str, start_idx: int, end_idx_excl: int, tmpdir: str) -> Tuple[str, int]:
    """
    Create a temp PDF containing pages [start_idx, end_idx_excl) from src_path.
    Returns (out_path, num_pages_written).

    Notes:
    - start_idx/end_idx_excl are zero-based, end exclusive.
    - Indices are clamped to the document's bounds.
    - If the (clamped) range is empty, an empty PDF is written and num_pages_written == 0.
    """
    reader = PdfReader(src_path)
    writer = PdfWriter()

    total = len(reader.pages)
    start = max(0, start_idx)
    end = min(end_idx_excl, total)
    if end < start:
        end = start  # empty range

    for i in range(start, end):
        writer.add_page(reader.pages[i])

    base = os.path.basename(src_path)
    # Filename uses 1-based inclusive end for readability; for empty, show start==end.
    page_tag = f"p{start+1}-{end}" if end > start else f"p{start+1}-{start}"
    out_path = os.path.join(tmpdir, f"{Path(base).stem}_{page_tag}.pdf")

    with open(out_path, "wb") as f:
        writer.write(f)

    num_pages_written = end - start
    print("out_path")
    print(out_path)
    return out_path, num_pages_written


def iter_page_batches(pages_total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    """
    Yields (start_idx, end_idx_excl) zero-based half-open ranges.
    """
    i = 0
    while i < pages_total:
        j = min(i + batch_size, pages_total)
        yield (i, j)
        i = j

# ---------- Gemini client & structured schemas ----------
def make_genai_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY_GEMINI")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GOOGLE_API_KEY_GEMINI).")
    return genai.Client(api_key=api_key)

# ---------- Structured JSON schema builders (top-level with scratchpad + pages[1..3]) ----------
def make_page_schema_for_baseline() -> dict:
    # Each item = Intervention scores for a single page
    return {
        "type": "object",
        "properties": {
            "predicting_outcomes_description": {"type": "string", "maxLength":300},
            "predicting_outcomes_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "description_of_key_activities_description": {"type": "string", "maxLength":300},
            "description_of_key_activities_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "how_invested_are_partners_description": {"type": "string", "maxLength":300},
            "how_invested_are_partners_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "implementer_skill_level_description": {"type": "string", "maxLength":300},
            "implementer_skill_level_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "larger_project_integration_description": {"type": "string", "maxLength":300},
            "larger_project_integration_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "riskiness_and_risks_description": {"type": "string", "maxLength":300},
            "riskiness_and_risks_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "expected_outcomes_future_description": {"type": "string", "maxLength":300},
            "expected_outcomes_future_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "theory_of_change_description": {"type": "string", "maxLength":300},
            "theory_of_change_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
        },
        "required": [
            "predicting_outcomes_informativeness",
            "description_of_key_activities_informativeness",
            "how_invested_are_partners_informativeness",
            "implementer_skill_level_informativeness",
            "larger_project_integration_informativeness",
            "riskiness_and_risks_informativeness",
            "expected_outcomes_future_informativeness",
            "theory_of_change_informativeness",
        ],
        "propertyOrdering": [
            "predicting_outcomes_description",
            "predicting_outcomes_informativeness",
            "description_of_key_activities_description",
            "description_of_key_activities_informativeness",
            "how_invested_are_partners_description",
            "how_invested_are_partners_informativeness",
            "implementer_skill_level_description",
            "implementer_skill_level_informativeness",
            "larger_project_integration_description",
            "larger_project_integration_informativeness",
            "riskiness_and_risks_description",
            "riskiness_and_risks_informativeness",
            "expected_outcomes_future_description",
            "expected_outcomes_future_informativeness",
            "theory_of_change_description",
            "theory_of_change_informativeness",
        ]
    }

def make_page_schema_for_outcome() -> dict:
    # Each item = Evaluation scores for a single page
    return {
        "type": "object",
        "properties": {
            "evaluation_description": {"type": "string", "maxLength": 300},
            "evaluation_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "quantitative_outcomes_description": {"type": "string", "maxLength": 300},
            "quantitative_outcomes_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "qualitative_outcomes_description": {"type": "string", "maxLength": 300},
            "qualitative_outcomes_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "qualitative_degree_of_success_description": {"type": "string", "maxLength": 300},
            "qualitative_degree_of_success_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "quantitative_degree_of_success_description": {"type": "string", "maxLength": 300},
            "quantitative_degree_of_success_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "reasons_for_failure_or_issues_description": {"type": "string", "maxLength": 300},
            "reasons_for_failure_or_issues_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "deviation_from_expectations_description": {"type": "string", "maxLength": 300},
            "deviation_from_expectations_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
            "expected_outcomes_original_description": {"type": "string", "maxLength": 300},
            "expected_outcomes_original_informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
        },
        "required": [
            "evaluation_informativeness",
            "quantitative_outcomes_informativeness",
            "qualitative_outcomes_informativeness",
            "qualitative_degree_of_success_informativeness",
            "quantitative_degree_of_success_informativeness",
            "reasons_for_failure_or_issues_informativeness",
            "deviation_from_expectations_informativeness",
            "expected_outcomes_original_informativeness",
        ],
        "propertyOrdering": [
            "evaluation_description",
            "evaluation_informativeness",
            "quantitative_outcomes_description",
            "quantitative_outcomes_informativeness",
            "qualitative_outcomes_description",
            "qualitative_outcomes_informativeness",
            "qualitative_degree_of_success_description",
            "qualitative_degree_of_success_informativeness",
            "quantitative_degree_of_success_description",
            "quantitative_degree_of_success_informativeness",
            "reasons_for_failure_or_issues_description",
            "reasons_for_failure_or_issues_informativeness",
            "deviation_from_expectations_description",
            "deviation_from_expectations_informativeness",
            "expected_outcomes_original_description",
            "expected_outcomes_original_informativeness",
        ],

    }

def make_top_schema(section: str, n_items: int) -> dict:
    # Allow 1..3 pages so the final partial batch can be scored when we hit the 500-page cap.
    page_schema = make_page_schema_for_baseline() if section == "Baseline" else make_page_schema_for_outcome()
    return {
        "type": "object",
        "properties": {
            "scratchpad": {"type": "string", "maxLength": 1000},
            "pages": {
                "type": "array",
                "minItems": n_items,
                "maxItems": n_items,
                "items": page_schema,
            },
        },
        "required": ["pages"],
        "propertyOrdering": ["scratchpad", "pages"],
    }

# ---------- Token counting helpers ----------
# def count_input_tokens(client: genai.Client, model: str, prompt_text: str, file_path: str) -> int:
#     uploaded = client.files.upload(file=file_path)
#     resp = client.models.count_tokens(model=model, contents=[prompt_text, uploaded])
#     return int(resp.total_tokens)

# def try_get_output_tokens(client: genai.Client, model: str, response_text: str) -> int:
#     """
#     Prefer usage metadata if available; otherwise attempt count_tokens on text.
#     If unavailable, return -1.
#     """
#     # The caller should pass response.usage_metadata if available; here we only have text fallback.
#     try:
#         resp = client.models.count_tokens(model=model, contents=[response_text])
#         return int(resp.total_tokens)
#     except Exception:
#         return -1

# ---------- CSV I/O ----------
CSV_FIELDNAMES = [
    "run_timestamp_iso",
    "activity_id",
    "activity_title",
    "activity_description_truncated_3000",
    "section",
    "doc_index_int",
    "doc_title",
    "cached_file",
    "pdf_pages_total",
    "page_start",
    "page_end",
    "model_name",
    "input_token_count",
    "output_token_count",
    "scratchpad",
] + INTERVENTION_KEYS_DESCRIPTION + INTERVENTION_KEYS_SCORES + EVALUATION_KEYS_DESCRIPTION + EVALUATION_KEYS_SCORES

# print("CSV_FIELDNAMES")
# print(CSV_FIELDNAMES)
# input("fieldnames okay?")
def ensure_csv_header(path: str):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not exists:
            w.writeheader()

def append_csv_row(row: Dict[str, Any]):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writerow(row)

# ---------- Activity counts crash-safe tracker ----------
def load_activity_counts() -> Dict[str, int]:
    if not os.path.exists(ACT_COUNTS_CSV):
        return {}
    out: Dict[str, int] = {}
    with open(ACT_COUNTS_CSV, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            aid = r.get("activity_id", "")
            try:
                cnt = int(r.get("count", "0"))
            except Exception:
                cnt = 0
            if aid:
                out[aid] = cnt
    return out

def persist_activity_counts(activity_counts: Dict[str, int]):
    with open(ACT_COUNTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["activity_id", "count"])
        w.writeheader()
        for aid, cnt in sorted(activity_counts.items()):
            w.writerow({"activity_id": aid, "count": cnt})

# ---------- Prompt builder ----------
PROMPT_TEMPLATE = """Provide a brief summary and then score each page provided from the PDF document on how much information each page provides on each of the following questions:

{string_describing_rubric}

CONTEXT:
- ACTIVITY TITLE: {activity_title}
- ACTIVITY DESCRIPTION: {activity_desc_3000}
- SECTION: {section}
- PDF TITLE: {pdf_title}
- PAGE RANGE: {page_range} out of {pages_total} pages

INSTRUCTIONS:
- Use integers from 0 to 10, where:
  0 = not mentioned at all in the document excerpt; 10 = explicit, clear, and complete answer in the document excerpt
- Return JSON with a top-level "scratchpad" (brief reasoning, optional) and "pages": a brief, non-repetitive summary of the content for each question on that page, and scores for each individual page.
- The description is ideally a direct copy of the most relevant text on the page. Otherwise summarize the content contained there (if not repeating other responses).
- For the grade for each field, score exclusively the level of informativeness contained in that page for each field.
- Do not repeat information between the pages in the brief summaries. If there is no new relevant information to summarize, leave it blank.
- If the page is blank or contains little substantive information (e.g. title page, table of contents, references), leave out all descriptions and score 0 on every item for that page.
- Ensure all answers to the prompts are in English regardless of the source document language
"""

def build_prompt(activity_title: str, activity_desc_3000: str, section: str,
                 pdf_title: str, pdf_path: str, page_start_1based: int, page_end_1based: int, pages_total: int) -> str:
    if section == "Baseline":
        int_or_eval_string = INTERVENTION_DICT_STRING
    else:
        int_or_eval_string = EVALUATION_DICT_STRING

    if page_start_1based == page_end_1based:
        page_range = str(page_start_1based)
    else:
        page_range = "pages "+str(page_start_1based) + "-" + str(page_end_1based)
    print("int_or_eval_string")
    print(int_or_eval_string)
    return PROMPT_TEMPLATE.format(
        string_describing_rubric=int_or_eval_string,
        activity_title=activity_title,
        activity_desc_3000=(activity_desc_3000 or "")[:1500],
        section=section,
        pdf_title=pdf_title or "",
        pdf_path=pdf_path,
        page_range=page_range,
        pages_total=pages_total,
    )

def get_promising_baseline_and_outcomes_ranked(usable_baseline, usable_outcome):
    """
    Select docs from ranked_documents.csv:
      - excluded_flag == False
      - assigned_grade >= 'c-' (c-/c/c+/.../a+)
      - pages <= 300
      - order by assigned_rank asc
      - limit to first 5 docs per section
    Then join back to usable_* rows by a stable key (title, pages, language).
    """
    RANKS_CSV = "../../data/ranked_documents.csv"

    def key_from_row(r: dict) -> tuple:
        title = (r.get("doc_title", "") or "").strip()
        try:
            pages = int((r.get("pages", "") or r.get("page_count", "") or "0"))
        except Exception:
            pages = 0
        lang = (r.get("language", "") or "").strip().lower()
        return (title, pages, lang)

    # Build quick lookup from usable docs
    base_map = {key_from_row(r): r for r in usable_baseline}
    outc_map = {key_from_row(r): r for r in usable_outcome}

    # Acceptable grades (c- or better)
    grade_order = ["a+", "a", "a-", "b+", "b", "b-", "c+", "c", "c-"]
    allowed = set(grade_order)

    ranked_baseline: List[dict] = []
    ranked_outcome: List[dict] = []

    if not os.path.exists(RANKS_CSV):
        return [], []

    with open(RANKS_CSV, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            section = (r.get("section") or "").strip()
            title = (r.get("doc_title") or "").strip()
            try:
                pages = int((r.get("page_count") or "0"))
            except Exception:
                pages = 0
            lang = (r.get("language") or "").strip().lower()
            try:
                excluded = str(r.get("excluded_flag", "")).strip().lower()
                excluded_flag = excluded in ("True", "true", "1", "yes")
            except Exception:
                excluded_flag = False
            grade = (r.get("assigned_grade") or "").strip().lower()
            # Skip rows that aren't doc rows
            if not title or section not in ("Baseline", "Outcome"):
                continue
            if excluded_flag:
                continue
            if grade not in allowed:
                continue
            if pages <= 0 or pages > 300:
                continue
            try:
                rank = int(r.get("assigned_rank", "-1"))
            except Exception:
                rank = -1
            if rank < 1:
                continue

            key = (title, pages, lang)
            if section == "Baseline" and key in base_map:
                ranked_baseline.append((rank, base_map[key]))
            elif section == "Outcome" and key in outc_map:
                ranked_outcome.append((rank, outc_map[key]))

    ranked_baseline.sort(key=lambda x: x[0])
    ranked_outcome.sort(key=lambda x: x[0])
    def _dedupe(rows): # remove duplicate rows
        seen, out = set(), []
        for r in rows:
            key = r.get("cached_file") or (r.get("doc_title"), r.get("pages"), r.get("language"))
            if key in seen: 
                continue
            seen.add(key)
            out.append(r)
        return out[:5]

    chosen_baselines = _dedupe([d for _, d in ranked_baseline])
    chosen_outcomes  = _dedupe([d for _, d in ranked_outcome])

    # chosen_baselines = [d for _, d in ranked_baseline[:5]]
    # chosen_outcomes  = [d for _, d in ranked_outcome[:5]]

    return chosen_baselines, chosen_outcomes

def append_error_row(aid, act_title, section, start_idx, end_idx):
    row: Dict[str, Any] = {
        "run_timestamp_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "activity_id": aid,
        "activity_title": act_title,
        "activity_description_truncated_3000": -1,
        "section": section,
        "doc_index_int": -1,
        "doc_title": -1,
        "cached_file": -1,
        "pdf_pages_total": -1,
        "page_start": start_idx,
        "page_end": end_idx,
        "model_name": MODEL_NAME,
        "input_token_count": -1,
        "output_token_count": -1,
        "scratchpad": -1,
    }



# ---------- Main processing ----------
async def main():
    # Basic path checks
    if not os.path.exists(FINAL_CSV):
        raise SystemExit(f"Missing {FINAL_CSV}")
    if not os.path.exists(LOCATION_PDFS):
        raise SystemExit(f"Missing {LOCATION_PDFS}")
    if not os.path.exists(SUBSET_JSON):
        print(f"Warning: missing {SUBSET_JSON}; activity descriptions will be empty.")

    # Load data
    activities, order = load_docs(FINAL_CSV)
    acts_map = load_acts_map(SUBSET_JSON)
    activity_counts = load_activity_counts()

    # Shuffle activities (seed=42) and pick N with both Baseline and Outcome usable
    rng = random.Random(SEED)
    rng.shuffle(order)
    picked: List[str] = []
    for aid in order:
        secmap = activities[aid]["sections"]
        if filter_usable(secmap.get("Baseline", [])) and filter_usable(secmap.get("Outcome", [])):
            picked.append(aid)
        if len(picked) >= N_ACTIVITIES:
            break
    if not picked:
        print("No qualifying activities found.")
        return

    # CSV header
    ensure_csv_header(OUTPUT_CSV)

    # Client and tmpdir
    client = make_genai_client()
    tmpdir = tempfile.mkdtemp(prefix="pdf_batches_")

    try:
        for idx, aid in enumerate(picked, start=1):
            # Crash-safe counter update (skip if beyond threshold)
            count_so_far = activity_counts.get(aid, 0) + 1
            if count_so_far > MAX_PER_ACTIVITY:
                print(f"[skip: activity seen {count_so_far-1} time(s) already] {aid}\n")
                continue
            activity_counts[aid] = count_so_far
            persist_activity_counts(activity_counts)

            secmap = activities[aid]["sections"]
            title_from_csv = activities[aid]["title"] or ""
            act_obj = acts_map.get(aid, {})
            act_title = activity_title(act_obj) or title_from_csv
            act_desc = desc(act_obj) or ""

            # Prepare selections
            usable_baseline = filter_usable(secmap.get("Baseline", []))
            usable_outcome  = filter_usable(secmap.get("Outcome", []))
            if (len(usable_baseline) == 0) or (len(usable_outcome) == 0):
                print("\nWARNING: missing a baseline or outcome, skipping\n")
                continue
            # HERE, WE NEED TO SORT BY THE ORDER FROM ../../data/ranked_documents.csv
            sel_baseline, sel_outcome = get_promising_baseline_and_outcomes_ranked(usable_baseline, usable_outcome)
            # sel_baseline = pick_desc_first_n(usable_baseline, MAX_PICK_PER_SEC)
            # sel_outcome  = pick_desc_first_n(usable_outcome,  MAX_PICK_PER_SEC)

            print(f"\n\n\n[{idx:02d}] New Activity {aid} — {act_title}")
            print("sel_baseline")
            print(sel_baseline)
            print("sel_outcome")
            print(sel_outcome)

            total_qs = 0

            for section, rows in (("Baseline", sel_baseline), ("Outcome", sel_outcome)):
                docs_used = 0
                pages_used = 0

                print(f"\n NEW SECTION: {section}")

                for d in rows:
                    if docs_used >= 5 or pages_used >= 500:
                        break

                    rel_path = (d.get("cached_file", "") or "").strip()
                    abs_path = os.path.join(LOCATION_PDFS, rel_path)
                    if not os.path.exists(abs_path):
                        continue
                    try:
                        pages_total = int(d.get("pages", 0))
                    except Exception:
                        pages_total = 0
                    if pages_total <= 0:
                        continue
                    if pages_total > 300:
                        continue

                    remaining = 500 - pages_used
                    if remaining <= 0:
                        break

                    effective_pages = min(pages_total, remaining)
                    if effective_pages <= 0:
                        continue

                    bsz = batch_size_for_pages(effective_pages)
                    doc_index = d.get("_doc_index_int", 0)
                    doc_title = d.get("doc_title", "") or "(untitled)"

                    # Process batches (limited to effective_pages, supports partial last batch)
                    for start_idx, end_idx in iter_page_batches(effective_pages, bsz):
                        # Make a sliced temp PDF for this batch (file-only upload)
                        slice_path, n_pdf_pages = write_pdf_slice(abs_path, start_idx, end_idx, tmpdir)

                        # Build prompt (no extracted text, only metadata + instructions)
                        prompt_text = build_prompt(
                            activity_title=act_title,
                            activity_desc_3000=act_desc,
                            section=section,
                            pdf_title=doc_title,
                            pdf_path=rel_path,
                            page_start_1based=start_idx + 1,
                            page_end_1based=start_idx + n_pdf_pages,
                            pages_total=pages_total
                        )

                        # Print the prompt being sent
                        print("\n--- PROMPT TO MODEL ---")
                        print(prompt_text)
                        print("-----------------------\n")

                        # Upload file again for generation (separate lifecycle from counting)
                        uploaded = client.files.upload(file=slice_path)
                        print(f"n_pages {n_pdf_pages}")
                        print("slice_path")
                        print(slice_path)
                        print("")
                        print("")
                        open_with_evince(slice_path)
                        # Configure structured output schema per section (Baseline vs Outcome)
                        top_schema = make_top_schema(section, n_pdf_pages)

                        total_qs += 1
                        print("done uploading, now doing the prompt")
                        try:
                            # Run the synchronous call in a thread; preserve your overall timeout
                            async def _call():
                                return await asyncio.to_thread(
                                    client.models.generate_content,
                                    model=MODEL_NAME,
                                    contents=[prompt_text, uploaded],
                                    config={
                                        "response_mime_type": "application/json",
                                        "response_schema": top_schema,
                                    },
                                )
                            response = await asyncio.wait_for(_call(), timeout=TIMEOUT_SECONDS)
                        
                        except asyncio.exceptions.TimeoutError:
                            append_error_row(aid, act_title, section, start_idx, end_idx)
                            print("timeout error 5 minutes. continuing.\n\n")
                            # append_error_row(csv_path, activity_id, section, -1, "5 minute timer failure.")
                            print('asyncio.exceptions.TimeoutError')
                            continue
                        except asyncio.CancelledError:
                            append_error_row(aid, act_title, section, start_idx, end_idx)
                            # If the task was cancelled due to global shutdown or outer timeout
                            # append_error_row(csv_path, activity_id, section, -1, "Cancelled")
                            print('asyncio.CancelledError')
                            continue

                        # # Generate structured output (top-level: scratchpad + pages[1..3])
                        # response = client.models.generate_content(
                        #     model=MODEL_NAME,
                        #     contents=[prompt_text, uploaded],
                        #     # contents=[prompt_text],
                        #     config={
                        #         "response_mime_type": "application/json",
                        #         "response_schema": top_schema,
                        #     },
                        # )
                        # r = client.models.generate_content(model=MODEL, contents=["Return {}", uploaded],
                        #     config={"response_mime_type":"application/json","response_schema":schema},
                        #     request_options={"timeout":60})
                        print("uploaded!")

                        # Parse strictly; crash if parsing fails
                        parsed = getattr(response, "parsed", None)
                        if parsed is None:
                            try:
                                parsed = json.loads(response.text)
                            except Exception:
                                append_error_row(aid, act_title, section, start_idx, end_idx)
                                print('RuntimeError("Structured output parsing failed (no parsed object and JSON load failed).")')
                                continue

                        if not isinstance(parsed, dict) or "pages" not in parsed or not isinstance(parsed["pages"], list):
                            append_error_row(aid, act_title, section, start_idx, end_idx)
                            print('RuntimeError("Structured output parsing failed (missing "pages" array).")')
                            continue

                        pages_items = parsed["pages"]
                        if not pages_items:
                            append_error_row(aid, act_title, section, start_idx, end_idx)
                            print('RuntimeError("Structured output parsing failed (empty "pages" array).")')
                            continue

                        # Collect output token count if available
                        output_tokens = -1
                        usage = getattr(response, "usage_metadata", None)
                        # if usage and isinstance(usage, dict):
                        #     output_tokens = int(usage.get("output_tokens", -1))
                        # else:
                        #     output_tokens = -1
                        output_tokens = -1
                        usage = getattr(response, "usage_metadata", None)
                        if usage is not None:
                            # handles both dict-like and attribute objects
                            output_tokens = (
                                usage.get("output_tokens", -1) if hasattr(usage, "get")
                                else getattr(usage, "output_tokens", -1)
                            )

                        # Emit one CSV row per page in this batch.
                        page_offset_1based = start_idx + 1
                        # pprint.pprint("\n\npages_items")
                        # pprint.pprint(pages_items)
                        print("printing outputs")
                        for i, page in enumerate(pages_items):
                            actual_page_num = page_offset_1based + i
                            if actual_page_num > pages_total:
                                break

                            row: Dict[str, Any] = {
                                "run_timestamp_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "activity_id": aid,
                                "activity_title": act_title,
                                "activity_description_truncated_3000": (act_desc or "")[:3000],
                                "section": section,
                                "doc_index_int": doc_index,
                                "doc_title": doc_title,
                                "cached_file": rel_path,
                                "pdf_pages_total": pages_total,
                                "page_start": actual_page_num,
                                "page_end": actual_page_num + n_pdf_pages,
                                "model_name": MODEL_NAME,
                                "input_token_count": -1,
                                "output_token_count": output_tokens,
                                "scratchpad": parsed.get("scratchpad", ""),
                            }


                            for k in INTERVENTION_KEYS + EVALUATION_KEYS:
                                row[k+"_informativeness"] = ""
                                row[k+"_description"] = ""
                            pprint.pprint("\n\npage")
                            pprint.pprint(page)

                            if section == "Baseline":
                                for k in INTERVENTION_KEYS:
                                    row[k+"_informativeness"] = int(page[f"{k}_informativeness"])
                                    if f"{k}_description" in page:
                                        row[k+"_description"] = page[f"{k}_description"]
                                    else:
                                        row[k+"_description"] = ""
                            else:
                                for k in EVALUATION_KEYS:
                                    row[k+"_informativeness"] = int(page[f"{k}_informativeness"])
                                    if f"{k}_description" in page:
                                        row[k+"_description"] = page[f"{k}_description"]
                                    else:
                                        row[k+"_description"] = ""

                            print("")
                            pprint.pprint(row)
                            print("")
                            # input("hit enter to add csv row")
                            # time.sleep(1)
                            append_csv_row(row)

                    # After finishing this doc (whether full or truncated), update counters
                    pages_used += effective_pages
                    docs_used += 1

                if total_qs >= 100:
                    print('quitting ("we reached 100")')
                    quit()
            print()  # spacing after activity

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    _program_start = datetime.now()  # <-- add this
    asyncio.run(main())
    print(f"\n(END PROGRAM): took {(datetime.now() - _program_start).total_seconds():.2f}s\n\n\n\n")  # <-- add this
