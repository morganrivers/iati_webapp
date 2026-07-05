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
from typing import Set

from pypdf import PdfReader, PdfWriter

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
OUTPUT_CSV        = str(DATA_DIR / "pdf_categories_scores_expenditure_breakdown.csv")

EXISTING_CATEGORIES_CSV = str(DATA_DIR / "pdf_categories_scores.csv")

RESTRICT_TO_MERGED_RATINGS = True
MERGED_RATINGS_JSONL = str(DATA_DIR / "merged_overall_ratings.jsonl")

BATCH_MODE = True
OUTPUT_JSONL_BATCH = str(DATA_DIR / "batch_requests" / "pdf_categories_scores_D.jsonl")

RANKS_CSV         = str(DATA_DIR / "ranked_documents.csv")

SEED              = 42
MAX_PICK_PER_SEC  = 3
MODEL_NAME        = "gemini-2.5-flash-lite"
TIMEOUT_SECONDS = 300

CONCURRENCY = 10  # run up to 5 activities at once

NUMBER_PAGES_BATCH = 3
AVG_BYTES_PER_PAGE = 60_000  # ~60kB/page

from score_page_relevance import activity_title, iter_page_batches, desc
from prompt_bundle_pdf import open_with_evince

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
    }


def make_top_schema(section: str, n_items: int) -> dict:
    """
    Top-level: optional scratchpad + pages[] where each page is an array of up to 3 enum strings.
    """
    page_schema = make_page_schema_for_baseline() if section == "Baseline" else make_page_schema_for_outcome()
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
