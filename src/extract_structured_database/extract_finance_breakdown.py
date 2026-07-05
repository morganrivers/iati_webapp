#!/usr/bin/env python3
"""
Extract finance + sector breakdowns + planned disbursements + planned duration
+ intended allocation by quantitative outcome from evaluation PDFs using structured JSON.

This follows the same "pages -> consolidate_rows_by_activity -> prompts -> loop_over_rows_to_call_model"
pattern you already use.

Outputs: JSONL (1 line per activity bundle) to OUTPUT_JSONL.
"""

import json
import pprint
import sys
import asyncio
from datetime import datetime
from typing import Optional, Set, Dict, Any, Tuple
from pathlib import Path

import csv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


from extracting_and_grading_helper_functions import (
    consolidate_rows_by_activity,
    loop_over_rows_to_call_model,
)
from repo_paths import DATA_DIR


# --------------------
# Paths / constants
# --------------------

MERGED_OVERALL_RATINGS = str(DATA_DIR / "merged_overall_ratings.jsonl")
CHATGPT_SUMMARIES_JSONL = str(DATA_DIR / "outputs_summaries.jsonl")  # adjust if different in your repo

PROCESS_OUTCOMES = False
if PROCESS_OUTCOMES:
    OUTPUT_JSONL = str(DATA_DIR / "outputs_finance_sectors_disbursements_outcomes_gemini2p5flash.jsonl")
else:
    OUTPUT_JSONL = str(DATA_DIR / "outputs_finance_sectors_disbursements_baseline_gemini2p5flash.jsonl")

# Choose which CSV to use for page categorization scores
# Option 1: Original categorization (for baseline)

MODEL_NAME = "gemini-2.5-flash"
CONCURRENCY = 6

if not PROCESS_OUTCOMES:
    PDF_SCORES_CSV = DATA_DIR / "pdf_categories_scores.csv"
else:
    # Option 2: Expenditure breakdown-focused categorization
    PDF_SCORES_CSV = DATA_DIR / "pdf_categories_scores_expenditure_breakdown.csv"

OUTCOME_ENUM = [
        "reduced rates of open defecation",
        "more people with drinking water services",
        "more people with access to electricity",
        "increased managed forest land",
        "reduced CO2 emissions",
        "reduced PM2.5 air pollution",
        "reduced electricity transmission losses",
        # "increased electricity production",
        "increased renewable electricity production",
        # "increased cereal yield",
        "increased food production",
        "other",
    ]
BASELINE_ENUM = [
        "reduced rates of open defecation", #
        "more people with drinking water services",
        "more people with access to electricity",
        "increased managed forest land",
        "reduced CO2 emissions",
        "reduced PM2.5 air pollution",
        "reduced electricity transmission losses", #
        # "increased electricity production",
        "increased renewable electricity production", #
        # "increased cereal yield",
        "increased food production",
        "other",
    ]

# --------------------
# Response schema
# --------------------
def get_finance_sector_outcomes_schema() -> dict:
    """
    Simplified schema:

    - planned_duration_years: float (can be decimal)
    - planned_disbursements: loan + grant only (either/both/none)
    - sector_breakdown: for HHI (shares and/or amounts)
    - quantitative_outcome_allocations: free-form sector + intended_amount
    - Nothing required. No totals. No notes. No dates.
    """

    # sectors_enum = [
    #     "",
        
    # ]
    
    # disb_item = {
    #     "type": "object",
    #     "properties": {
    #         "amount": {"type": "number"},
    #         "currency": {"type": "string", "maxLength": 16},
    #     },
    # }
    grant_or_loan = [
        "grant",
        "loan"
    ]


    return {
        "type": "object",
        "required": ["total_allocation"],
        "properties": {
            "total_allocation": {
                "type": "object",
                "required": ["amount", "currency"],
                "properties": {
                    "amount": {"type": "number"},
                    "currency": {"type": "string", "maxLength": 16},
                },
            },
            "quantitative_outcome_allocations": {
                # Free-form sector + intended_amount only
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # "description": {"type": "string", "maxLength": 400},
                        # "sector": {"type": "string", "enum": sectors_enum},
                        "custom_outcome": {"type": "string", "maxLength": 100},  # Add this                        
                        "outcome": {"type": "string", "enum": OUTCOME_ENUM if PROCESS_OUTCOMES else BASELINE_ENUM},
                        "grant_or_loan": {"type": "string", "enum": grant_or_loan},
                        "amount_allocated": {"type": "number"},
                        "currency": {"type": "string", "maxLength": 16},
                    },
                },
            },
        },
    }
def get_finance_sector_outcomes_prompt(chunked_by_activity_id: list) -> Dict[str, str]:
    """
    activity_id -> prompt string
    """
    prompts: Dict[str, str] = {}

    for bundle in chunked_by_activity_id:
        aid = bundle.get("activity_id")
        if not aid:
            continue
        title = (bundle.get("activity_title") or bundle.get("title") or "").strip()
        # pprint.pprint(bundle)
        # quit()
        desc = (bundle.get("chatgpt_description") or "").strip()#[:1200]
        # summary = (bundle.get("chatgpt_summary") or bundle.get("summary") or "").strip()[:1200]

        prompts[aid] = (
            "Return only valid JSON matching the provided schema.\n\n"
            "CONTEXT:\n"
            f"- ACTIVITY TITLE: {title}\n"
            f"- ACTIVITY DESCRIPTION: {desc}\n"
            # f"- SUMMARY: {summary[:2000]}\n\n"
            "TASK:\n"
            f"You are extracting the {'planned' if not PROCESS_OUTCOMES else ''} allocations of loans and grants for this international aid activity. You will respond with how much of the total loans and disbursements were allocated between different categories of intended funding."
             # and intended outcomes, as well as \"overhead\" which applies to both. For each portion of the total loans and/or disbursements, record: \n"
            # "1) planned_duration_years:\n"
            # "   - A number of years (float allowed). Examples: 3, 2.5, 6.0.\n"
            # "   - If the document gives start/end dates, convert to approximate years only if the duration is explicit or trivial.\n"
            # "2) planned_disbursements:\n"
            # "   - Capture the total planned amount split by type.\n"
            # "   - Only two allowed keys: loan and grant.\n"
            # "   - If the document uses another term (e.g. 'credit', 'IDA', 'concessional loan'), map it to the closest of loan/grant.\n"
            # "   - If you only find one type, include just that one.\n"
            # "3) sector_breakdown:\n"
            "      - Label for spend category among the following options:\n"
            f"{', '.join([label for label in OUTCOME_ENUM])}\n"
            # "      - Outcome category (optional) among the following options:\n" 
            "      - If you select 'other' you MUST fill in a very short label for the category in custom_outcome (max 100 chars)\n"
            "      - Whether the record is a grant or a loan\n" 
            "      - The amount of grant or loan allocated\n" 
            "      - The currency of the recorded allocation. Prefer \"million USD\" where it is one of the stated options.\n\n" 
            "RULES:\n"
            "- If there is no way to assess the amount of funding contributing to the outcome in the pages, return only a total grant of quantity zero and units \"Not Found\".\n"
            # "- Do not infer missing numbers.\n"
            "- If something is not clearly stated, omit it.\n"
            "- Prefer totals over yearly schedules; ignore per-year breakdowns unless they are the only thing shown (then sum only if explicitly totaled).\n"
            "- Only select 'other' if the allocation cannot be interpreted to be among the selected options\n"
            "- The sum of the allocations must approximately sum to the total loans and grants over the full duration of the program.\n"
            "- Before responding, verify that the sum of amount_allocated is approximately equal to total_allocation."
            "- If you cannot reconcile the numbers, return only total_allocation and omit quantitative_outcome_allocations entirely."
        )

    return prompts


# --------------------
# Main
# --------------------
