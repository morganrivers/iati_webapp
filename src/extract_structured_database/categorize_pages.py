#!/usr/bin/env python3
"""
COST: batch mode for about 20k separate queries, with 258 tokens per pdf page input, and about 1000 tokens thinking + response (in USD)
calc "20000*0.15*258*3/1e6+20000*1.25*1000/1e6"
    27.322

so roughly 27 dollars for the full set for 2.5 flash.


PDF batch scorer with Gemini structured output (file-only uploads).

- Loads activity docs from FINAL_CSV and activity objects from SUBSET_JSON.

CSV:
  - Appends one row per page to ../../data/pdf_relevance_scores.csv
  - Adds a "scratchpad" column.
  - Crashes if JSON parsing fails (structured output must succeed).

Requirements:
  micromamba install google-genai pypdf pydantic

Environment:
  GOOGLE_API_KEY (or GOOGLE_API_KEY_GEMINI) must be set.
"""

import os
import csv
import sys
import pandas as pd

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

# --- add near other imports ---
from typing import Set

from pypdf import PdfReader, PdfWriter

# --- Google GenAI (Gemini) ---
from google import genai
import pprint

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from repo_paths import DATA_DIR

# ---------- Fixed paths / constants ----------
FINAL_CSV         = str(DATA_DIR / "activity_docs_log_final_restrictive.csv")
SUBSET_JSON       = str(DATA_DIR / "subset_results.json")
LOCATION_PDFS     = str(DATA_DIR / "iati_all_pdfs")
# OUTPUT_CSV        = "../../data/pdf_categories_scores.csv"
OUTPUT_CSV        = str(DATA_DIR / "pdf_categories_scores_expenditure_breakdown.csv")

EXISTING_CATEGORIES_CSV = str(DATA_DIR / "pdf_categories_scores.csv")

# Restrict to activity IDs from merged ratings file
RESTRICT_TO_MERGED_RATINGS = True
MERGED_RATINGS_JSONL = str(DATA_DIR / "merged_overall_ratings.jsonl")

BATCH_MODE = True
OUTPUT_JSONL_BATCH = str(DATA_DIR / "batch_requests" / "pdf_categories_scores_D.jsonl")

RANKS_CSV         = str(DATA_DIR / "ranked_documents.csv")

SEED              = 42
# N_ACTIVITIES      = 200
MAX_PICK_PER_SEC  = 3
# MODEL_NAME        = "gemini-2.5-flash"
# MODEL_NAME        = "gemini-2.0-flash"
MODEL_NAME        = "gemini-2.5-flash-lite"
TIMEOUT_SECONDS = 300

CONCURRENCY = 10  # run up to 5 activities at once

NUMBER_PAGES_BATCH = 3
AVG_BYTES_PER_PAGE = 60_000  # ~60kB/page

from score_page_relevance import load_docs, load_acts_map, load_activity_counts, filter_usable, activity_title, iter_page_batches, write_pdf_slice, desc, activity_title
from prompt_bundle_pdf import open_with_evince
from extracting_and_grading_helper_functions import make_genai_client
from get_codes_we_like import get_activity_to_codes, get_good_bad_and_target_codes

# ---------- Structured JSON schema builders (top-level with scratchpad + pages[1..3]) ----------
def make_page_schema_for_baseline() -> dict:
    """
    A page = array (0..3) of enum categories. No summary, no detail strings.
    """
    baseline_enum_1 = [
        "table_of_contents",
        "blank_page",
        "glossary",
        "references",
        "core_activities",
        "theory_of_change",
        "targets",
        "broader_context",
        "preliminary_results",
        "other",
    ]

    baseline_enum_2 = [
        "condensed_summary",
        "sub_activities_outlined",
        "detailed_implementation_plans",
        "broad_objectives",
        "possible_outcomes",
        "quantitative_targets",
        "qualitative_targets",
        "risks_as_word_or_numeric",
        "risks_or_dangers_generally",
        "plans_to_address_key_risks",
        "positive_indicators",
        "progress_reports",
        "similar_cases_outcomes",
        "implementation_context_country",
        "contextual_challenges",
        "financing_details",
        "budget_and_legal",
        "who_implements",
        "whether_part_of_larger_program",
        "partner_identity_or_skill",
        "whether_skin_in_the_game",
        "other_stakeholder_engagement",
        "activity_monitoring_details",
    ]

    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": baseline_enum_1},
            "subcategory_A": {"type": "string", "enum": baseline_enum_2},
            "subcategory_B": {"type": "string", "enum": baseline_enum_2},
            "informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
        },
        "required": ["category", "informativeness"],
        # "propertyOrdering": ["category_1", "category_2", "evaluation_informativeness"],
    }


def make_page_schema_for_outcome() -> dict:
    """
    A page = array (0..3) of enum categories. No summary, no detail strings.
    """
    evaluation_enum_1 = [
        "glossary",
        "blank_page",
        "table_of_contents",
        "outcome_evaluation",
        "activity_description",
        "references",
        "other",
    ]
    evaluation_enum_2 = [
        "expenditures_breakdown",
        "expected_outcomes",
        "deviation_from_plans",
        "preliminary_results",
        "final_outcomes",
        "delays_or_early_completion",
        "over_or_under_spending",
        "overview_as_was_planned",
        "unrelated_to_evaluation",
    ]

    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": evaluation_enum_1},
            "subcategory_A": {"type": "string", "enum": evaluation_enum_2},
            "subcategory_B": {"type": "string", "enum": evaluation_enum_2},
            "has_quantitative_targets": {"type": "boolean"},
            "has_quantitative_outcomes": {"type": "boolean"},
            "has_overall_ratings": {"type": "boolean"},
            "informativeness": {"type": "integer", "minimum": 0, "maximum": 10},
        },
        "required": ["category", "informativeness","has_quantitative_targets","has_quantitative_outcomes","has_overall_ratings"],
        # "propertyOrdering": ["category_1", "category_2", "evaluation_informativeness"],
    }


def make_top_schema(section: str, n_items: int) -> dict:
    """
    Top-level: optional scratchpad + pages[] where each page is an array of up to 3 enum strings.
    """
    page_schema = make_page_schema_for_baseline() if section == "Baseline" else make_page_schema_for_outcome()
    # pprint.pprint("page_schema")
    # pprint.pprint(page_schema)
    # quit()
    return {
        "type": "object",
        "properties": {
            "scratchpad": {"type": "string", "maxLength": 800},
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
    "baseline_category_1",
    "baseline_category_2",
    "baseline_category_3",
    "outcome_category_1",
    "outcome_category_2",
    "outcome_category_3",
    "has_quantitative_targets",
    "has_quantitative_outcomes",
    "has_overall_ratings",
    "score",
]

# ---------- Prompt builder ----------
PROMPT_TEMPLATE = """Categorize each of the attached PDF pages from an {document_description} document with the categories best reflecting the content contained in those pages, then provide a score from 0-10 for how informative the content in each page is for {informativeness_text}.

CONTEXT:
- ACTIVITY TITLE: {activity_title}
- ACTIVITY DESCRIPTION: {activity_desc_3000}
- SECTION: {section}
- PDF TITLE: {pdf_title}
- PAGE RANGE: {page_range} out of {pages_total} pages

INSTRUCTIONS:
- Return JSON with a top-level "scratchpad" (brief reasoning, optional) and "pages": an array of page categories, with up to three different categorizations per page as is needed to properly reflect the content of the page, and an informativeness score from 0-10 for {informativeness_text}.
"""

def build_prompt(activity_title: str, activity_desc_3000: str, section: str,
                 pdf_title: str, pdf_path: str, page_start_1based: int, page_end_1based: int, pages_total: int, document_description: str, informativeness_text: str) -> str:
    if page_start_1based == page_end_1based:
        page_range = str(page_start_1based)
    else:
        page_range = "pages "+str(page_start_1based) + "-" + str(page_end_1based)
    return PROMPT_TEMPLATE.format(
        activity_title=activity_title,
        activity_desc_3000=(activity_desc_3000 or "")[:1500],
        section=section,
        pdf_title=pdf_title or "",
        pdf_path=pdf_path,
        page_range=page_range,
        pages_total=pages_total,
        document_description=document_description,
        informativeness_text=informativeness_text,
    )

def get_promising_baseline_and_outcomes_ranked(usable_baseline, usable_outcome):
    def norm(s): return (s or "").strip().lower()

    def key_doc(d):
        return (norm(d.get("doc_title")), norm(d.get("language")))

    base_map = {}
    for d in usable_baseline:
        base_map.setdefault(key_doc(d), []).append(d)

    outc_map = {}
    for d in usable_outcome:
        outc_map.setdefault(key_doc(d), []).append(d)

    allowed = {"a+","a","a-","b+","b","b-","c+","c","c-"}
    ranked_baseline = []
    ranked_outcome  = []

    with open(RANKS_CSV, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            section = (r.get("section") or "").strip()
            if section not in ("Baseline", "Outcome"):
                continue

            excluded = str(r.get("excluded_flag", "")).strip().lower()
            if excluded in ("true","1","yes"):
                continue

            grade = (r.get("assigned_grade") or "").strip().lower()
            if grade not in allowed:
                continue

            try:
                rank = int(r.get("assigned_rank") or -1)
            except Exception:
                rank = -1
            if rank < 1:
                continue

            # IMPORTANT: accept whatever the ranks file calls the page count
            try:
                pages = int(r.get("page_count") or r.get("pages") or 0)
            except Exception:
                pages = 0
            if pages <= 0 or pages > 300:
                continue

            k = (norm(r.get("doc_title")), norm(r.get("language")))

            # If multiple activity docs share same title/lang, pick closest page count
            if section == "Baseline" and k in base_map:
                best = min(base_map[k], key=lambda d: abs(int(d.get("pages") or 0) - pages))
                ranked_baseline.append((rank, best))
            elif section == "Outcome" and k in outc_map:
                best = min(outc_map[k], key=lambda d: abs(int(d.get("pages") or 0) - pages))
                ranked_outcome.append((rank, best))

    ranked_baseline.sort(key=lambda x: x[0])
    ranked_outcome.sort(key=lambda x: x[0])

    # dedupe by cached_file
    def dedupe(rows):
        seen, out = set(), []
        for _, d in rows:
            cf = (d.get("cached_file") or "").strip()
            if not cf or cf in seen:
                continue
            seen.add(cf)
            out.append(d)
        return out

    return dedupe(ranked_baseline)[:2], dedupe(ranked_outcome)[:2]




def build_batch_request(
    prompt: str | dict,
    uploaded_files: List[Tuple[Optional[str], Any]],
    response_schema: dict | None = None,
) -> dict:
    """
    Build a Gemini Batch API request object.

    uploaded_files: list of (label_text_or_None, uploaded_file_obj),
    where uploaded_file_obj has .uri and .mime_type.
    """
    system_instruction = None
    text_prompt = prompt

    if isinstance(prompt, dict):
        system_instruction = prompt.get("system_msg")
        text_prompt = prompt.get("prompt")

    parts: List[Dict[str, Any]] = []

    if system_instruction:
        parts.append({
            "text": f"SYSTEM INSTRUCTION:\n{system_instruction}"
        })

    parts.append({"text": text_prompt})

    for label, uploaded in uploaded_files:
        if label:
            parts.append({"text": label})
        parts.append({
            "fileData": {
                "fileUri": uploaded.uri,
                "mimeType": uploaded.mime_type,
            }
        })

    request_obj: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
    }

    if response_schema is not None:
        request_obj["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema,
        }

    return request_obj


def write_batch_request_line(
    output_jsonl: str,
    batch_key: str,
    request_obj: dict,
) -> Path:
    """
    Append a single JSONL line for Gemini Batch.

    Uses the same naming convention as run_one_row:
      <../../data/batch_requests>/<stem>_batch<suffix>
    """
    out_path = Path(output_jsonl)
    batch_dir = DATA_DIR / "batch_requests"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_jsonl #batch_dir / f"{out_path.stem}_batch{out_path.suffix}"

    line = {"key": batch_key, "request": request_obj}
    with open(batch_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line))
        f.write("\n")

    return batch_path


def one_line(s) -> str:
    if s is None:
        return ""
    s = str(s)
    # normalize newlines first, then escape
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", "\\t")
    return s.replace("\n", "\\n")

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
    append_csv_row(row)

def ensure_csv_header(path: str):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not exists:
            w.writeheader()
def load_completed_activity_pages(csv_path: str) -> Set[Tuple[str, str, int]]:
    """
    Return {(activity_id, cached_file, page_start_1based)} for rows already written.
    """
    completed: Set[Tuple[str, str, int]] = set()
    if not os.path.exists(csv_path):
        return completed

    # quick total rows (approx = lines - header)
    with open(csv_path, "rb") as f:
        total_lines = sum(1 for _ in f)
    total_rows = max(0, total_lines - 1)
    print(f"CSV rows (excluding header): {total_rows:,}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        count = 0
        for row in csv.DictReader(f):
            count += 1
            if count % 1000 < 10:
                print(f"at row {count}...")
            aid = (row.get("activity_id") or "").strip()
            rel = (row.get("cached_file") or "").strip()
            try:
                p = int(row.get("page_start", "0"))
            except ValueError:
                continue
            if aid and rel and p > 0:
                completed.add((aid, rel, p))
    return completed

def append_csv_row(row: Dict[str, Any]) -> None:
    """
    Append a row to OUTPUT_CSV using this module's CSV_FIELDNAMES.
    This avoids fieldname drift with score_page_relevance.append_csv_row.
    """
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDNAMES,
        )
        # Make sure every declared field exists (write empty string if missing)
        safe_row = {k: row.get(k, "") for k in CSV_FIELDNAMES}
        w.writerow(safe_row)


def filter_usable_fast(rows: List[dict]) -> List[dict]:
    out = []
    for d in rows:
        try:
            pages = int((d.get("pages", "") or "0").strip())
        except Exception:
            pages = 0
        if pages <= 0 or pages == "":
            continue
        out.append(d)
    return out


def allowed_activity_ids_from_ranks(ranks_csv: str) -> set[str]:
    pass_grades = {"a+","a","a-","b+","b","b-","c+","c","c-"}
    df = pd.read_csv(ranks_csv)

    excluded = (
        df.get("excluded_flag")
          .astype(str).str.strip().str.lower()
          .isin({"true","1","yes"})
    )
    page_count = pd.to_numeric(df.get("page_count"), errors="coerce").fillna(0).astype(int)
    assigned_rank = pd.to_numeric(df.get("assigned_rank"), errors="coerce").fillna(-1).astype(int)

    restrict = (~excluded) & page_count.between(1, 300) & (assigned_rank >= 1) & df["assigned_grade"].str.lower().isin(pass_grades)

    base_ok = set(df[(df["section"].str.lower()=="baseline") & restrict]["activity_id"])
    out_ok  = set(df[(df["section"].str.lower()=="outcome")  & restrict]["activity_id"])
    return base_ok & out_ok  # this is your 271

def allowed_activity_ids_with_good_dac_codes(activity_ids) -> set[str]:
    activity_to_codes = get_activity_to_codes()
    GOOD_CODES, BAD_CODES, TARGET_CODES = get_good_bad_and_target_codes()
    return_aids = set()
    for aid in activity_ids:
        codes = activity_to_codes.get(aid)
        if not codes:
            # no DAC 5-digit codes for this activity; treat as not-good
            continue
        if len(activity_to_codes[aid] & GOOD_CODES) > 0:
            return_aids.add(aid)
    return return_aids

def load_merged_ratings_activity_ids(jsonl_path: str) -> set[str]:
    """Load activity IDs from merged ratings JSONL file."""
    if not os.path.exists(jsonl_path):
        print(f"Warning: Merged ratings file not found: {jsonl_path}")
        return set()

    activity_ids = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                aid = obj.get("activity_id")
                if aid:
                    activity_ids.add(aid)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(activity_ids)} activity IDs from {jsonl_path}")
    return activity_ids

from collections import defaultdict

def load_excluded_pages_from_categories(csv_path: str) -> Set[Tuple[str, str, int]]:
    """
    Return {(activity_id, cached_file, page_start_1based)} for pages already categorized
    as glossary, blank_page, table_of_contents, or references in the Outcome section.
    Only used when RESTRICT_TO_MERGED_RATINGS is True.
    """
    excluded: Set[Tuple[str, str, int]] = set()
    if not os.path.exists(csv_path):
        return excluded

    exclude_categories = {"glossary", "blank_page", "table_of_contents", "references"}
    
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            aid = (row.get("activity_id") or "").strip()
            rel = (row.get("cached_file") or "").strip()
            try:
                p = int(row.get("page_start", "0"))
            except ValueError:
                continue
            
            if not (aid and rel and p > 0):
                continue
            
            # Check outcome categories only
            for col in ["outcome_category_1", "outcome_category_2", "outcome_category_3"]:
                cat = (row.get(col) or "").strip().lower()
                if cat in exclude_categories:
                    excluded.add((aid, rel, p))
                    break
    
    print(f"Loaded {len(excluded)} excluded pages from {csv_path}")
    return excluded

def build_todo_work_items(activities, picked, completed_pages, excluded_pages=None):
    todo = []
    for aid in picked:
        # if not aid.startswith("XM-DAC"):
        #     continue

        secmap = activities[aid]["sections"]

        def rows_for_section(section):
            rows = secmap.get(section, []) or []
            # DE-1: no ranks; just cached_file-present docs
            if aid.startswith("DE-1"):
                return [r for r in rows if (r.get("cached_file","") or "").strip()]
            return rows

        base_rows = rows_for_section("Baseline")
        out_rows  = rows_for_section("Outcome")

        if aid.startswith("DE-1"):
            # if only outcome exists, treat it as baseline (same as your runtime logic)
            if not base_rows and out_rows:
                base_rows, out_rows = out_rows, []

            sel_baseline, sel_outcome = base_rows, out_rows
        else:
            sel_baseline, sel_outcome = get_promising_baseline_and_outcomes_ranked(base_rows, out_rows)

        DEBUG_AID = "XM-DAC-41114-PROJECT-00113842"

        if aid == DEBUG_AID:
            print("\n=== DEBUG build_todo_work_items for", aid, "===")
            print("base_rows:", [(d.get("doc_index"), d.get("pages"), d.get("cached_file"), d.get("doc_title")) for d in base_rows[:10]])
            print("out_rows :", [(d.get("doc_index"), d.get("pages"), d.get("cached_file"), d.get("doc_title")) for d in out_rows[:10]])
            print("sel_baseline:", [(d.get("doc_index"), d.get("pages"), d.get("cached_file"), d.get("doc_title")) for d in sel_baseline])
            print("sel_outcome :", [(d.get("doc_index"), d.get("pages"), d.get("cached_file"), d.get("doc_title")) for d in sel_outcome])
        if RESTRICT_TO_MERGED_RATINGS:
            work_list = [("Outcome", sel_outcome)]  # Note the brackets!
        else:
            work_list = [("Baseline", sel_baseline), ("Outcome", sel_outcome)]
        # print("work_list")
        # print(work_list)
        for section, rows in work_list:
            docs_used = 0
            pages_used = 0
            for d in rows:
                if docs_used >= 5 or pages_used >= 500:
                    break

                rel_path = (d.get("cached_file", "") or "").strip()
                if not rel_path:
                    continue
                abs_path = os.path.join(LOCATION_PDFS, rel_path)
                if not os.path.exists(abs_path):
                    continue

                pages_total = int(d.get("pages", 0) or 0)
                if pages_total <= 0 or pages_total > 300:
                    continue

                effective_pages = min(pages_total, 500 - pages_used)
                if effective_pages <= 0:
                    continue

                for start_idx, end_idx in iter_page_batches(effective_pages, NUMBER_PAGES_BATCH):
                    if start_idx >= 50:
                        break
                    if (aid, rel_path, start_idx + 1) in completed_pages:
                        if aid == "XM-DAC-41114-PROJECT-00113842":
                            print("SKIP completed:", (aid, rel_path, start_idx + 1))
                        continue
                    # ADD THIS NEW CHECK HERE:
                    if excluded_pages and (aid, rel_path, start_idx + 1) in excluded_pages:
                        continue
                    
                    todo.append((aid, section, rel_path, start_idx))  # store just what we need for gating

                pages_used += effective_pages
                docs_used += 1
    return todo

# ---------- Main processing ----------
async def main():
    # Basic path checks
    if not os.path.exists(FINAL_CSV):
        raise SystemExit(f"Missing {FINAL_CSV}")
    if not os.path.exists(LOCATION_PDFS):
        raise SystemExit(f"Missing {LOCATION_PDFS}")
    if not os.path.exists(SUBSET_JSON):
        print(f"Missing {SUBSET_JSON}; activity descriptions will be empty.")
        raise SystemExit(f"Missing SUBSET_JSON")

    # Load data
    activities, order = load_docs(FINAL_CSV)
    acts_map = load_acts_map(SUBSET_JSON)
    # activity_counts = load_activity_counts()
    print("loaded docs and acts_map")
    # Shuffle activities (seed=42) and pick N with both Baseline and Outcome usable
    rng = random.Random(SEED)
    rng.shuffle(order)
    
    allowed_aids = allowed_activity_ids_from_ranks(RANKS_CSV)
    print("restricting to the nice codes we like...")
    allowed_aids = allowed_activity_ids_with_good_dac_codes(allowed_aids)
    print("done restricting to the nice codes we like...")

    # Optionally restrict to merged ratings activity IDs
    if RESTRICT_TO_MERGED_RATINGS:
        merged_aids = load_merged_ratings_activity_ids(MERGED_RATINGS_JSONL)
        if merged_aids:
            allowed_aids = allowed_aids & merged_aids
            print(f"Restricted to {len(allowed_aids)} activities from merged ratings")

    picked = []
    for aid in order:
        secmap = activities[aid]["sections"]

        # --- SPECIAL HANDLING FOR DE-1* ACTIVITIES ---
        if aid.startswith("DE-1"):
            # don't filter on ranks / DAC codes; just require at least one doc with a cached file
            has_any_doc = False
            for section in ("Baseline", "Outcome"):
                for d in secmap.get(section, []) or []:
                    rel_path = (d.get("cached_file", "") or "").strip()
                    if rel_path:
                        has_any_doc = True
                        break
                if has_any_doc:
                    break

            if has_any_doc:
                picked.append(aid)
            # skip the normal filtering logic for DE-1
            continue

        # --- ORIGINAL LOGIC FOR NON-DE-1 ACTIVITIES ---
        if aid not in allowed_aids:
            continue
        if filter_usable_fast(secmap.get("Baseline", [])) and filter_usable_fast(secmap.get("Outcome", [])):
            picked.append(aid)


    # picked: List[str] = []
    # for aid in order:
    #     secmap = activities[aid]["sections"]
    #     if filter_usable_fast(secmap.get("Baseline", [])) and filter_usable_fast(secmap.get("Outcome", [])):

    #         picked.append(aid)
    #     # if len(picked) >= N_ACTIVITIES:
    #     #     break
    if not picked:
        print("No qualifying activities found.")
        return
    print("added all the usable activity ids")
    DEBUG_AID = "XM-DAC-41114-PROJECT-00113842"

    print("\n=== DEBUG picked ===")
    print("picked count:", len(picked))
    print("DEBUG_AID in picked?", DEBUG_AID in picked)
    if DEBUG_AID not in picked:
        secmap = activities.get(DEBUG_AID, {}).get("sections", {})
        print("\n=== DEBUG why not picked ===")
        print("has baseline usable?", bool(filter_usable_fast(secmap.get("Baseline", []))))
        print("has outcome  usable?", bool(filter_usable_fast(secmap.get("Outcome",  []))))

        allowed_aids_raw = allowed_activity_ids_from_ranks(RANKS_CSV)
        allowed_aids_good = allowed_activity_ids_with_good_dac_codes(allowed_aids_raw)

        # print("in allowed_aids_raw (rank gate)?", DEBUG_AID in allowed_aids_raw)
        # print("in allowed_aids_good (dac gate)?", DEBUG_AID in allowed_aids_good)

    # CSV header
    ensure_csv_header(OUTPUT_CSV)
    # completed_ids = load_completed_activity_ids(OUTPUT_CSV)
    print("load_completd_pages...")
    completed_pages = load_completed_activity_pages(OUTPUT_CSV)
    # ADD THIS:
    excluded_pages = set()
    if RESTRICT_TO_MERGED_RATINGS:
        excluded_pages = load_excluded_pages_from_categories(EXISTING_CATEGORIES_CSV)

    print("build todo work items...")
    todo = build_todo_work_items(activities, picked, completed_pages, excluded_pages)



    # START = ("44000-P171059", "Outcome", "2f243b653d4a__p171059-521f0849-914a-45e9-806c-a229a1667f9a.pdf.pdf", 3)  # page_start=4 -> start_idx=3
    # skip_n = todo.index(START); print(f"[resume] skipping {skip_n} todo batches; starting at {START}")
    # todo = todo[skip_n:]





    print("todo batches:", len(todo))
    if not todo:
        print("Nothing to do.")
        return

    todo_map = defaultdict(set)
    for aid, section, rel_path, start_idx in todo:
        todo_map[aid].add((section, rel_path, start_idx))

    todo_aids = sorted(todo_map.keys())
    print("\n=== DEBUG planned activity ids ===")
    print("planned count:", len(todo_aids))
    print("first 50 planned:", todo_aids[:50])

    print("DEBUG_AID planned?", DEBUG_AID in todo_map)

    aid = "XM-DAC-41114-PROJECT-00113842"
    print("\n=== DEBUG todo_map for", aid, "===")
    print("todo_map.keys()")
    print(todo_map.keys())
    # items = sorted(todo_map.get(aid, []))
    # print("total todo items:", len(items))
    # print("baseline todo items:", sum(1 for x in items if x[0] == "Baseline"))
    # print("outcome  todo items:", sum(1 for x in items if x[0] == "Outcome"))
    # quit()
    total_planned_pages = 0
    for aid, section, rel_path, start_idx in todo:
        end_idx = min(start_idx + NUMBER_PAGES_BATCH, 50)
        total_planned_pages += max(0, end_idx - start_idx)

    est_bytes = total_planned_pages * AVG_BYTES_PER_PAGE
    est_mb = est_bytes / (1024 * 1024)
    print(f"Planned upload: {total_planned_pages} pages (~{est_mb:.1f} MB).")

    # total_planned_pages = len(todo) * NUMBER_PAGES_BATCH  # rough

    # # # --- NEW: pre-pass to estimate total pages + upload size ---
    # # total_planned_pages = compute_total_pages_to_upload(
    # #     activities=activities,
    # #     acts_map=acts_map,
    # #     picked=picked,
    # #     completed_pages=completed_pages,
    # # )
    # if total_planned_pages <= 0:
    #     print("Nothing new to upload (all relevant pages already processed or skipped).")
    # else:
    #     est_bytes = total_planned_pages * AVG_BYTES_PER_PAGE
    #     est_mb = est_bytes / (1024 * 1024)
    #     print(
    #         f"Planned upload: {total_planned_pages} pages "
    #         f"(~{est_mb:.1f} MB at {AVG_BYTES_PER_PAGE/1024:.1f} kB/page)."
    #     )

    # shared progress counter
    pages_uploaded = 0

    # Client and tmpdir
    client = make_genai_client()
    tmpdir = tempfile.mkdtemp(prefix="pdf_batches_")

    try:
        async def _process_activity(aid: str, idx: int):
            if aid not in todo_map:
                return

            nonlocal pages_uploaded, total_planned_pages
            tmpdir = tempfile.mkdtemp(prefix=f"pdf_batches_{aid}_")
            try:
                # Crash-safe counter update (skip if beyond threshold)
                # count_so_far = activity_counts.get(aid, 0) + 1
                # if count_so_far > MAX_PER_ACTIVITY:
                #     print(f"[skip: activity seen {count_so_far-1} time(s) already] {aid}\n")
                #     continue
                # activity_counts[aid] = count_so_far
                # persist_activity_counts(activity_counts)

                secmap = activities[aid]["sections"]
                title_from_csv = activities[aid]["title"] or ""
                act_obj = acts_map.get(aid, {})
                act_title = activity_title(act_obj) or title_from_csv
                act_desc = desc(act_obj) or ""

                is_de1 = aid.startswith("DE-1")

                if is_de1:
                    # --- SPECIAL HANDLING FOR DE-1* ACTIVITIES ---
                    raw_base = secmap.get("Baseline", []) or []
                    raw_out  = secmap.get("Outcome", []) or []

                    def _has_cached(r: dict) -> bool:
                        return bool((r.get("cached_file", "") or "").strip())

                    # "don't filter, just inject the documents directly (only test is that the cached file exists)"
                    usable_baseline = [r for r in raw_base if _has_cached(r)]
                    usable_outcome  = [r for r in raw_out  if _has_cached(r)]

                    # If there is only outcome document, treat it as baseline and run baseline prompt
                    treat_outcome_as_baseline = False
                    if not usable_baseline and usable_outcome:
                        treat_outcome_as_baseline = True
                        usable_baseline = usable_outcome
                        usable_outcome = []

                    # For DE-1 we skip the ranked-doc selection
                    sel_baseline, sel_outcome = usable_baseline, usable_outcome

                else:
                    # --- ORIGINAL LOGIC FOR NON-DE-1 ACTIVITIES ---
                    usable_baseline = filter_usable_fast(secmap.get("Baseline", []))
                    usable_outcome  = filter_usable_fast(secmap.get("Outcome", []))

                    if (len(usable_baseline) == 0) or (len(usable_outcome) == 0):
                        print("\nWARNING: missing a baseline or outcome, skipping\n")
                        return

                    sel_baseline, sel_outcome = get_promising_baseline_and_outcomes_ranked(
                        usable_baseline,
                        usable_outcome,
                    )
                    treat_outcome_as_baseline = False

                # print("\n\n\nsel_baseline[0]")
                # print(sel_baseline[0])
                # print("sel_baseline")
                # print(sel_baseline)
                # print("sel_outcome[0]")
                # print(sel_outcome[0])
                # print("sel_outcome")
                # print(sel_outcome)
                # quit()
                # sel_baseline = pick_desc_first_n(usable_baseline, MAX_PICK_PER_SEC)
                # sel_outcome  = pick_desc_first_n(usable_outcome,  MAX_PICK_PER_SEC)
                # print("len(sel_baseline)")
                # print(len(sel_baseline))
                # print("len(sel_outcome)")
                # print(len(sel_outcome))
                # quit()
                print(f"\n\n\n[{idx:02d}] New Activity {aid} — {act_title}")
                # print("sel_baseline")
                # print(sel_baseline)
                # print("sel_outcome")
                # print(sel_outcome)
                total_qs = 0

                if is_de1 and treat_outcome_as_baseline:
                    # only run the baseline prompt, using the (former) outcome docs as baseline
                    section_rows = [("Baseline", sel_baseline)]
                else:
                    section_rows = [("Baseline", sel_baseline), ("Outcome", sel_outcome)]

                for section, rows in section_rows:

                    docs_used = 0
                    pages_used = 0

                    print(f"\n NEW SECTION: {section}")
                    if section == "Baseline":
                        document_description = "activity description"
                        informativeness_text = "forecasting future activity outcomes"
                    else:
                        document_description = "activity evaluation"
                        informativeness_text = "evaluating past activity outcomes"
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

                        doc_index = d.get("_doc_index_int", 0)
                        doc_title = d.get("doc_title", "") or "(untitled)"

                        # Process batches (limited to effective_pages, supports partial last batch)
                        for start_idx, end_idx in iter_page_batches(effective_pages, NUMBER_PAGES_BATCH):
                            if (section, rel_path, start_idx) not in todo_map[aid]:
                                continue

                            # for idx, aid in enumerate(picked, start=1):
                            if (aid, rel_path, start_idx+1) in completed_pages:
                                # print(f"[skip: already recorded: {aid} {rel_path}, pages {start_idx+1}")
                                continue
                            if start_idx >= 50:
                                # print("past page 50, we will skip.")
                                continue
                            # if aid in completed_ids:
                            #     # actual_page_num
                            #     continue
                            # Make a sliced temp PDF for this batch (file-only upload)
                            try:
                                print("writing pdf slice!")
                                slice_path, n_pdf_pages = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        write_pdf_slice,
                                        abs_path,
                                        start_idx,
                                        end_idx,
                                        tmpdir,
                                    ),
                                    timeout=10,  # seconds
                                )
                            except asyncio.TimeoutError:
                                append_error_row(aid, act_title, section, start_idx, end_idx)
                                print("write_pdf_slice timed out after 10 seconds; recorded error row and continuing.\n")
                                continue
                            print("done writing pdf slice.")
                            # slice_path, n_pdf_pages = "",0#write_pdf_slice(abs_path, start_idx, end_idx, tmpdir)

                            # Build prompt (no extracted text, only metadata + instructions)
                            prompt_text = build_prompt(
                                activity_title=act_title,
                                activity_desc_3000=act_desc,
                                section=section,
                                pdf_title=doc_title,
                                pdf_path=rel_path,
                                page_start_1based=start_idx + 1,
                                page_end_1based=start_idx + n_pdf_pages,
                                pages_total=pages_total,
                                document_description=document_description,
                                informativeness_text=informativeness_text,
                            )

                            # Print the prompt being sent
                            # print("\n--- PROMPT TO MODEL ---")
                            # print(prompt_text)
                            # print("-----------------------\n")

                            # Upload file again for generation (separate lifecycle from counting)
                            try:
                                uploaded = await asyncio.to_thread(client.files.upload, file=slice_path)
                            except Exception:
                                append_error_row(aid, act_title, section, start_idx, end_idx)
                                print("upload failed; recorded error row and continuing.\n")
                                continue
                            # --- NEW: progress accounting ---
                            pages_uploaded += n_pdf_pages
                            if total_planned_pages > 0:
                                pct = (pages_uploaded / total_planned_pages) * 100.0
                                print(
                                    f"[progress] Uploaded {pages_uploaded}/"
                                    f"{total_planned_pages} pages ({pct:.1f}%)"
                                )
                            print(f"n_pages {n_pdf_pages}")
                            print("slice_path")
                            print(slice_path)
                            print("")
                            # open_with_evince(slice_path)
                            # Configure structured output schema per section (Baseline vs Outcome)
                            top_schema = make_top_schema(section, n_pdf_pages)

                            total_qs += 1
                            print("done uploading, now doing the prompt")
                            if BATCH_MODE:
                                # ---- Batch mode: just enqueue a request, no live model call ----
                                # One line per (activity, section, pdf, starting page)
                                batch_key = f"{aid}::{section}::{rel_path}::{start_idx+1}"

                                request_obj = build_batch_request(
                                    prompt_text,
                                    uploaded_files=[("PAGES:", uploaded)],
                                    response_schema=top_schema,
                                )

                                write_batch_request_line(OUTPUT_JSONL_BATCH, batch_key, request_obj)
                                print(f"Queued batch request {batch_key}")
                                # In batch mode we don't parse or write CSV here; continue to next batch
                                continue

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
                            print("uploaded!")

                            # Parse strictly; crash if parsing fails
                            parsed = getattr(response, "parsed", None)
                            if parsed is None:
                                # try:
                                #     parsed = json.loads(response.text)
                                # except Exception:
                                #     append_error_row(aid, act_title, section, start_idx, end_idx)
                                #     print('RuntimeError("Structured output parsing failed (no parsed object and JSON load failed).")')
                                #     continue
                                try:
                                    parsed = json.loads(json_text)
                                except Exception as e:
                                    print(
                                        f"[line {line_no}] JSON parse error for key={key}: {e}\n"
                                        f"--- JSON head ---\n{json_text[:1000]}\n"
                                        f"--- JSON tail ---\n{json_text[-1000:]}\n"
                                        f"--- skipping ---"
                                    )
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
                            # output_tokens = -1
                            # usage = getattr(response, "usage_metadata", None)
                            # # if usage and isinstance(usage, dict):
                            # #     output_tokens = int(usage.get("output_tokens", -1))
                            # # else:
                            # #     output_tokens = -1
                            # output_tokens = -1
                            # usage = getattr(response, "usage_metadata", None)
                            # if usage is not None:
                            #     # handles both dict-like and attribute objects
                            #     output_tokens = (
                            #         usage.get("output_tokens", -1) if hasattr(usage, "get")
                            #         else getattr(usage, "output_tokens", -1)
                            #     )

                            # Emit one CSV row per page in this batch.
                            page_offset_1based = start_idx + 1
                            # pprint.pprint("\n\npages_items")
                            # pprint.pprint(pages_items)
                            # print("printing outputs")
                            for i, page in enumerate(pages_items):
                                actual_page_num = page_offset_1based + i
                                if actual_page_num > pages_total:
                                    break

                                score_val = int(page.get("informativeness"))
                                if score_val is None:
                                    append_error_row(aid, act_title, section, start_idx, end_idx)
                                    print('RuntimeError("Structured output parsing failed (informativeness not a valid 0-10 integer).")')
                                    continue

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
                                    "page_end": actual_page_num,
                                    "model_name": MODEL_NAME,
                                    "input_token_count": -1,
                                    "output_token_count": -1,
                                    "scratchpad": one_line(parsed.get("scratchpad", "")),
                                    "score": score_val,
                                }

                                # # --- assign row fields from the page's structured output ---
                                # # page is the current element from parsed["pages"]
                                # cats = page.get("categories", [])
                                # if not isinstance(cats, list):
                                #     cats = []

                                # # take up to 3, pad with empty strings for missing slots
                                # c1, c2, c3 = (cats + ["", "", ""])[:3]

                                if section == "Baseline":
                                    c1 = page.get("category")
                                    c2 = page.get("subcategory_A","")
                                    c3 = page.get("subcategory_B","")
                                    row["outcome_category_1"] = ""
                                    row["outcome_category_2"] = ""
                                    row["outcome_category_3"] = ""

                                    row["baseline_category_1"] = c1
                                    row["baseline_category_2"] = c2
                                    row["baseline_category_3"] = c3


                                    row["has_quantitative_targets"] = ""
                                    row["has_quantitative_outcomes"] = ""
                                    row["has_overall_ratings"] = ""


                                    print(f"baseline_category_1: {c1}")
                                    print(f"baseline_category_2: {c2}")
                                    print(f"baseline_category_3: {c3}")


                                else:
                                    c1 = page.get("category")
                                    c2 = page.get("subcategory_A","")
                                    c3 = page.get("subcategory_B","")
                                    hqt = page.get("has_quantitative_targets")
                                    hqo = page.get("has_quantitative_outcomes")
                                    hor = page.get("has_overall_ratings")

                                    row["baseline_category_1"] = ""
                                    row["baseline_category_2"] = ""
                                    row["baseline_category_3"] = ""

                                    row["outcome_category_1"] = c1
                                    row["outcome_category_2"] = c2
                                    row["outcome_category_3"] = c3

                                    row["has_quantitative_targets"] = hqt
                                    row["has_quantitative_outcomes"] = hqo
                                    row["has_overall_ratings"] = hor

                                    print(f"outcome_category_1: {c1}")
                                    print(f"outcome_category_2: {c2}")
                                    print(f"outcome_category_3: {c3}")
                                    print(f"has_quantitative_targets: {hqt}")
                                    print(f"has_quantitative_outcomes: {hqo}")
                                    print(f"has_overall_ratings: {hor}")
                                print(f"score {row['score']}")
                                # print()
                                # print("")
                                # pprint.pprint(row)
                                # print("")
                                # input("hit enter to add csv row")
                                # time.sleep(1)
                                append_csv_row(row)

                        # After finishing this doc (whether full or truncated), update counters
                        pages_used += effective_pages
                        docs_used += 1
                    print("total_qs")
                    print(total_qs)
                    # if total_qs >= 100:
                    #     print('quitting ("we reached 100")')
                    #     return
                # print()  # spacing after activity
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        sem = asyncio.Semaphore(CONCURRENCY)

        tasks = []

        async def _guard(aid, idx):
            async with sem:
                await _process_activity(aid, idx)

        todo_aids = list(todo_map.keys())
        for idx, aid in enumerate(todo_aids, start=1):
            tasks.append(asyncio.create_task(_guard(aid, idx)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                print("Task error:", r)

        # for idx, aid in enumerate(picked, start=1):
        #     all_aids.add(aid)
        #     tasks.append(asyncio.create_task(_guard(aid, idx)))
        # print("len(all_aids)")
        # print(len(all_aids))
        # if tasks:
        #     results = await asyncio.gather(*tasks, return_exceptions=True)
        #     for r in results:
        #         if isinstance(r, Exception):
        #             print("Task error:", r)


    finally:
        pass
        # shutil.rmtree(tmpdir, ignore_errors=True)


# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    _program_start = datetime.now()  # <-- add this
    asyncio.run(main())
    print(f"\n(END PROGRAM): took {(datetime.now() - _program_start).total_seconds():.2f}s\n\n\n\n")  # <-- add this
