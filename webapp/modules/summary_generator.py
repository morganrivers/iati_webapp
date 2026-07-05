#!/usr/bin/env python3
"""
Generate activity summary from baseline pages.
Uses generate_preactivity_summaries.py prompts and loop_over_rows_to_call_model().
"""


import logging
import sys
import json
import jsonlines
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from webapp_paths import ensure_src_paths
ensure_src_paths()

logger = logging.getLogger(__name__)

from generate_preactivity_summaries import get_prompts_summary
from extracting_and_grading_helper_functions import (
    loop_over_rows_to_call_model,
    consolidate_rows_by_activity,
    read_last_success_row,
)


def generate_activity_summary(
    pdf_path: str,
    activity_id: str,
    title: str,
    page_categories: List[Dict[str, Any]],
    output_dir: Path,
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Generate full activity description from baseline pages.

    Args:
        pdf_path: Path to PDF file
        activity_id: Activity identifier
        title: Activity title
        page_categories: Output from page_categorizer
        output_dir: Directory to save summary.jsonl
        model: Gemini model name

    Returns:
        Activity description text (chatgpt_description)
    """
    # Filter to relevant pages
    target_subcats = ["condensed_summary", "broad_objectives"]
    relevant_pages = [
        p for p in page_categories
        if p.get("section") == "Baseline" and (
            p.get("subcategory_a") in target_subcats or
            p.get("subcategory_b") in target_subcats
        )
    ]

    # Sort by score and take top 5
    relevant_pages = sorted(relevant_pages, key=lambda x: x.get("score", 0), reverse=True)[:5]

    if not relevant_pages:
        logger.warning("No summary pages found, using first 5 pages")
        relevant_pages = page_categories[:5]

    logger.info(f"Using {len(relevant_pages)} pages for summary")

    # Use absolute path for cached_file (helper prepends LOCATION_PDFS otherwise)
    abs_pdf_path = str(Path(pdf_path).resolve())

    # Build rows in format for loop_over_rows
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
        })

    # Consolidate
    chunked = consolidate_rows_by_activity(rows)

    # Get prompts using G's function
    prompts = get_prompts_summary(chunked)

    # Output file
    output_jsonl = output_dir / "summary.jsonl"

    # Create executor
    execpool = ThreadPoolExecutor(max_workers=1)

    try:
        # Call loop_over_rows (no schema for text generation)
        logger.info(f"Calling model with {len(chunked)} activities...")
        logger.info(f"Output will be saved to: {output_jsonl}")

        asyncio.run(loop_over_rows_to_call_model(
            str(output_jsonl),
            chunked,
            prompts,
            None,  # No structured schema
            execpool,
            model,
        ))

        logger.info("Model call completed")
    except Exception as e:
        logger.error(f"ERROR in loop_over_rows_to_call_model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Model call failed: {str(e)}") from e
    finally:
        execpool.shutdown(wait=False, cancel_futures=True)

    # Read result
    logger.info(f"Reading result from {output_jsonl}")
    result = read_last_success_row(output_jsonl)

    # Check if we need to parse response_text (from loop_over_rows)
    # or if we already have the final format (from previous run)
    if 'response_text' in result:
        # Fresh from loop_over_rows - response_text contains the summary
        description = result.get('response_text', '')
    elif 'chatgpt_description' in result:
        # Already in final format from previous run
        description = result.get('chatgpt_description', '')
    elif 'response' in result:
        # Alternative format
        description = result.get('response', '')
    else:
        logger.warning(f"Unexpected result format. Keys: {result.keys()}")
        description = ''

    logger.info(f"Got description of length {len(description)}")

    # Also save in standard format with chatgpt_description
    with jsonlines.open(output_jsonl, 'w') as writer:
        writer.write({
            "activity_id": activity_id,
            "chatgpt_description": description,
            "response": description,
        })

    logger.info(f"Summary saved to {output_jsonl}")

    return description
