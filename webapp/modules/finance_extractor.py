#!/usr/bin/env python3
"""
Extract finance breakdown from baseline pages.
Uses extract_finance_breakdown.py and loop_over_rows_to_call_model().
"""

import sys
import json
import jsonlines
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Add batch pipeline and utils to path

import logging

logger = logging.getLogger(__name__)

BATCH_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "extract_structured_database"
UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "utils"
if str(BATCH_DIR) not in sys.path:
    sys.path.insert(0, str(BATCH_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

# Import from M
from extract_finance_breakdown import (
    get_finance_sector_outcomes_schema,
    get_finance_sector_outcomes_prompt,
)

from extracting_and_grading_helper_functions import (
    loop_over_rows_to_call_model,
    consolidate_rows_by_activity,
    read_last_success_row,
)


def extract_finance_breakdown(
    pdf_path: str,
    activity_id: str,
    title: str,
    chatgpt_description: str,
    page_categories: List[Dict[str, Any]],
    output_dir: Path,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Extract finance breakdown from baseline pages.

    Args:
        pdf_path: Path to PDF file
        activity_id: Activity identifier
        title: Activity title
        chatgpt_description: Activity summary from Phase 2
        page_categories: Output from page_categorizer
        output_dir: Directory to save finance_breakdown.jsonl
        model: Gemini model name

    Returns:
        Dict with finance information
    """
    # Filter to finance pages
    target_subcats = ["financing_details", "budget_and_legal", "quantitative_targets",
                     "possible_outcomes", "detailed_implementation_plans"]
    relevant_pages = [
        p for p in page_categories
        if p.get("section") == "Baseline" and (
            p.get("subcategory_a") in target_subcats or
            p.get("subcategory_b") in target_subcats
        )
    ]

    # Sort by score and take top 10
    relevant_pages = sorted(relevant_pages, key=lambda x: x.get("score", 0), reverse=True)[:10]

    if not relevant_pages:
        logger.warning("No finance pages found, using first 10 pages")
        relevant_pages = page_categories[:10]

    logger.info(f"Using {len(relevant_pages)} pages for finance extraction")

    # Use absolute path for cached_file (helper prepends LOCATION_PDFS otherwise)
    abs_pdf_path = str(Path(pdf_path).resolve())

    # Build rows
    rows = []
    for p in relevant_pages:
        rows.append({
            "activity_id": activity_id,
            "cached_file": abs_pdf_path,
            "doc_title": "uploaded",
            "section": "Baseline",
            "activity_title": title,
            "title": title,
            "page_start": p.get("page_start", 1),
            "chatgpt_description": chatgpt_description,
        })

    # Consolidate
    chunked = consolidate_rows_by_activity(rows)

    # Add chatgpt_description to bundle (M expects this)
    for bundle in chunked:
        bundle["chatgpt_description"] = chatgpt_description

    # Get prompts from M
    prompts = get_finance_sector_outcomes_prompt(chunked)

    # Get schema from M
    schema = get_finance_sector_outcomes_schema()

    # Output file
    output_jsonl = output_dir / "finance_breakdown.jsonl"

    # Create executor
    execpool = ThreadPoolExecutor(max_workers=1)

    try:
        # Call loop_over_rows
        asyncio.run(loop_over_rows_to_call_model(
            str(output_jsonl),
            chunked,
            prompts,
            schema,
            execpool,
            model,
        ))
    finally:
        execpool.shutdown(wait=False, cancel_futures=True)

    # Read result (helper wraps structured output in response_text)
    result = read_last_success_row(output_jsonl)

    # Parse response_text if structured output was used
    if 'response_text' in result:
        finance_data = json.loads(result['response_text'])
        finance_data['activity_id'] = activity_id  # Preserve activity_id
    else:
        finance_data = result

    logger.info(f"Finance breakdown saved to {output_jsonl}")

    return finance_data
