#!/usr/bin/env python3
"""
Extract baseline features using extract_baseline_features.py and loop_over_rows_to_call_model().
"""

import sys
import json
import jsonlines
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Add batch pipeline and utils to path
BATCH_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "extract_structured_database"
UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "utils"
if str(BATCH_DIR) not in sys.path:
    sys.path.insert(0, str(BATCH_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

# Import from H
from extract_baseline_features import (
    get_implementer_performance_prompt,
    get_target_outcomes_prompt,
    get_risks_prompt,
    get_context_prompt,
    get_finance_prompt,
    get_misc_prompt,
    get_few_grades_schema,
)

from extracting_and_grading_helper_functions import (
    loop_over_rows_to_call_model,
    consolidate_rows_by_activity,
    read_last_success_row,
)


def extract_single_feature(
    pdf_path: str,
    activity_id: str,
    metadata_dict: Dict[str, Any],
    chatgpt_description: str,
    page_categories: List[Dict[str, Any]],
    output_jsonl: Path,
    get_prompt_func: callable,
    model: str,
    schema: Dict[str, Any] = None,
) -> str:
    """
    Extract a single feature using loop_over_rows_to_call_model().

    Args:
        pdf_path: Path to PDF
        activity_id: Activity ID
        metadata_dict: Metadata
        chatgpt_description: Summary
        page_categories: Page categories
        output_jsonl: Output file
        get_prompt_func: Function to get prompts (from H)
        model: Model name

    Returns:
        Response text
    """
    # Filter to baseline pages
    baseline_pages = [
        p for p in page_categories
        if p.get("section") == "Baseline" and p.get("score", 0) >= 3
    ]
    baseline_pages = sorted(baseline_pages, key=lambda x: x.get("score", 0), reverse=True)[:10]

    if not baseline_pages:
        baseline_pages = page_categories[:10]

    # Use absolute path for cached_file (helper prepends LOCATION_PDFS otherwise)
    abs_pdf_path = str(Path(pdf_path).resolve())

    # Build rows with metadata
    rows = []
    for p in baseline_pages:
        rows.append({
            "activity_id": activity_id,
            "cached_file": abs_pdf_path,
            "doc_title": "uploaded",
            "section": "Baseline",
            "activity_title": metadata_dict.get("title", ""),
            "title": metadata_dict.get("title", ""),
            "page_start": p.get("page_start", 1),
            "chatgpt_description": chatgpt_description,
            "participating_orgs": metadata_dict.get("participating_orgs", ""),
            "country_location": metadata_dict.get("country_location", ""),
            "implementing_org_type": metadata_dict.get("implementing_org_type"),
            "original_planned_start_date": metadata_dict.get("planned_start_date"),
            "actual_start_date": metadata_dict.get("planned_start_date"),
            "original_planned_close_date": metadata_dict.get("planned_end_date"),
        })

    # Consolidate
    chunked = consolidate_rows_by_activity(rows)

    # Add metadata to bundle (H expects this)
    for bundle in chunked:
        bundle["chatgpt_description"] = chatgpt_description
        bundle["participating_orgs"] = metadata_dict.get("participating_orgs", "")
        bundle["country_location"] = metadata_dict.get("country_location", "")
        bundle["implementing_org_type"] = metadata_dict.get("implementing_org_type")
        bundle["original_planned_start_date"] = metadata_dict.get("planned_start_date")
        bundle["actual_start_date"] = metadata_dict.get("planned_start_date")
        bundle["original_planned_close_date"] = metadata_dict.get("planned_end_date")

    # Build info dict for prompt functions (updated H expects this)
    info = {
        activity_id: {
            "participating_orgs": metadata_dict.get("participating_orgs", ""),
            "country_location": metadata_dict.get("country_location", ""),
            "gdp_percap": metadata_dict.get("gdp_percap", ""),
            "implementing_org_type": metadata_dict.get("implementing_org_type", ""),
            "actual_start_date": metadata_dict.get("planned_start_date", ""),
            "original_planned_close_date": metadata_dict.get("planned_end_date", ""),
            "activity_scope": metadata_dict.get("activity_scope", ""),
        }
    }

    # Get prompts using H's function (now expects info parameter)
    prompts = get_prompt_func(chunked, info)

    # Create executor
    execpool = ThreadPoolExecutor(max_workers=1)

    try:
        # Call loop_over_rows
        asyncio.run(loop_over_rows_to_call_model(
            str(output_jsonl),
            chunked,
            prompts,
            schema,  # Pass schema for structured output (or None for plain text)
            execpool,
            model,
        ))
    finally:
        execpool.shutdown(wait=False, cancel_futures=True)

    # Read result
    result = read_last_success_row(output_jsonl)

    # Check for response_text (from loop_over_rows) or response (from cache)
    if 'response_text' in result:
        return result.get('response_text', '')
    else:
        return result.get('response', '')


def extract_baseline_features(
    pdf_path: str,
    activity_id: str,
    metadata_dict: Dict[str, Any],
    chatgpt_description: str,
    page_categories: List[Dict[str, Any]],
    output_dir: Path,
    model: str = "gemini-2.5-flash",
    log_callback=None,
) -> Dict[str, str]:
    """
    Extract all baseline features (implementer, targets, risks).

    Args:
        pdf_path: Path to PDF file
        activity_id: Activity identifier
        metadata_dict: Metadata from Phase 0
        chatgpt_description: Activity summary from Phase 2
        page_categories: Output from page_categorizer
        output_dir: Directory to save feature JSONL files
        model: Gemini model name
        log_callback: Optional callable for progress logging

    Returns:
        Dict with feature responses
    """
    def log(msg: str) -> None:
        print(msg)
        if log_callback:
            log_callback(msg)

    features = {}

    # Extract each feature
    log("  📄 Extracting implementer_performance...")
    impl_output = output_dir / "implementer_performance.jsonl"
    features['implementer_performance'] = extract_single_feature(
        pdf_path, activity_id, metadata_dict, chatgpt_description,
        page_categories, impl_output, get_implementer_performance_prompt, model
    )

    log("  📄 Extracting targets...")
    targets_output = output_dir / "targets.jsonl"
    features['targets'] = extract_single_feature(
        pdf_path, activity_id, metadata_dict, chatgpt_description,
        page_categories, targets_output, get_target_outcomes_prompt, model
    )

    log("  📄 Extracting risks...")
    risks_output = output_dir / "risks.jsonl"
    features['risks'] = extract_single_feature(
        pdf_path, activity_id, metadata_dict, chatgpt_description,
        page_categories, risks_output, get_risks_prompt, model
    )

    log("  📄 Extracting context...")
    context_output = output_dir / "context.jsonl"
    features['context'] = extract_single_feature(
        pdf_path, activity_id, metadata_dict, chatgpt_description,
        page_categories, context_output, get_context_prompt, model
    )

    log("  📄 Extracting finance (qualitative)...")
    finance_qual_output = output_dir / "finance_qualitative.jsonl"
    features['finance'] = extract_single_feature(
        pdf_path, activity_id, metadata_dict, chatgpt_description,
        page_categories, finance_qual_output, get_finance_prompt, model
    )

    log("  📄 Extracting complexity and integratedness...")
    misc_output = output_dir / "misc.jsonl"
    misc_text = extract_single_feature(
        pdf_path, activity_id, metadata_dict, chatgpt_description,
        page_categories, misc_output, get_misc_prompt, model,
        schema=get_few_grades_schema()
    )

    # Parse structured output for complexity and integratedness
    try:
        misc_data = json.loads(misc_text)
        features['complexity'] = misc_data.get('complexity_details', '')
        features['integratedness'] = misc_data.get('how_integrated_description', '')
    except (json.JSONDecodeError, TypeError):
        # Fallback if parsing fails
        features['complexity'] = misc_text
        features['integratedness'] = misc_text

    log("  ✓ All features extracted and saved")

    return features
