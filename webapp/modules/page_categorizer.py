#!/usr/bin/env python3
"""
Page categorization for single PDF.
Uses categorize_pages.py schema and loop_over_rows_to_call_model().
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

from webapp_paths import ensure_src_paths
ensure_src_paths()

# Import schema builders from categorize_pages.py
from categorize_pages import make_top_schema, NUMBER_PAGES_BATCH

from extracting_and_grading_helper_functions import loop_over_rows_to_call_model, read_last_success_row


def categorize_single_pdf(
    pdf_path: str,
    activity_id: str,
    title: str,
    output_dir: Path,
    model: str = "gemini-2.5-flash",
    section: str = "Baseline",
) -> List[Dict[str, Any]]:
    """
    Categorize all pages of a single PDF using loop_over_rows_to_call_model().

    Args:
        pdf_path: Path to PDF file
        activity_id: Activity identifier
        title: Activity title
        output_dir: Directory to save page_categories.jsonl
        model: Gemini model name
        section: "Baseline" or "Outcome" (webapp only uses Baseline)

    Returns:
        List of page category dicts, one per page
    """
    from pypdf import PdfReader

    # Read PDF to get page count
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    logger.info(f"PDF has {total_pages} pages, categorizing in batches of {NUMBER_PAGES_BATCH}...")

    # Use absolute path for cached_file (helper prepends LOCATION_PDFS otherwise)
    abs_pdf_path = str(Path(pdf_path).resolve())

    # Build rows for each page
    rows = []
    for page_num in range(1, total_pages + 1):
        rows.append({
            "activity_id": activity_id,
            "cached_file": abs_pdf_path,
            "doc_title": "uploaded",
            "section": section,
            "activity_title": title,
            "page_start": page_num,
        })

    # For categorization, we process in batches of 3 pages
    # The batch script handles this internally via iter_page_batches
    # But we need to consolidate by "batch groups"

    # Actually, let's just pass all pages and let loop_over_rows handle batching
    # NO - loop_over_rows doesn't batch, it processes whole bundles
    # We need to manually batch the pages

    from extracting_and_grading_helper_functions import consolidate_rows_by_activity

    # Process pages in batches
    all_results = []
    batch_size = NUMBER_PAGES_BATCH  # 3

    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        batch_rows = rows[batch_start:batch_end]

        # Create a unique "activity_id" for this batch
        batch_id = f"{activity_id}_batch_{batch_start}_{batch_end}"
        for r in batch_rows:
            r["activity_id"] = batch_id  # Temporarily change ID for batching

        # Consolidate this batch
        chunked = consolidate_rows_by_activity(batch_rows)

        # Build prompt for this batch
        n_pages = len(batch_rows)
        prompt = f"""You are categorizing pages from an international aid activity document.

ACTIVITY TITLE: {title}
SECTION: {section} (documents created at the start of the activity)

TASK:
For each of the {n_pages} page(s) in this PDF slice, provide:
1. category: Main page type from the first enum
2. subcategory_A: Primary detailed category from the second enum
3. subcategory_B: Secondary detailed category from the second enum (if applicable)
4. informativeness: 0-10 score for how useful this page is for forecasting activity success

RULES:
- All pages are from {section} documents (planning/design phase)
- Select the most specific categories that apply
- informativeness: 0 = useless/blank, 10 = extremely informative for forecasting
- Return valid JSON matching the schema
"""

        prompts = {batch_id: prompt}

        # Get schema for this batch size
        schema = make_top_schema(section=section, n_items=n_pages)

        # Output file for this batch
        output_jsonl = output_dir / f"page_categories_batch_{batch_start}.jsonl"

        # Create executor
        execpool = ThreadPoolExecutor(max_workers=1)

        try:
            # Call loop_over_rows_to_call_model
            logger.info(f"Calling model for batch {batch_start+1}-{batch_end}...")
            asyncio.run(loop_over_rows_to_call_model(
                str(output_jsonl),
                chunked,
                prompts,
                schema,
                execpool,
                model,
            ))
            logger.info(f"Model call completed for batch {batch_start+1}-{batch_end}")
        except Exception as e:
            logger.error(f"ERROR calling model for batch {batch_start+1}-{batch_end}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model call failed for batch {batch_start+1}-{batch_end}: {str(e)}") from e
        finally:
            execpool.shutdown(wait=False, cancel_futures=True)

        # Read batch result
        if output_jsonl.exists() and output_jsonl.stat().st_size > 0:
            logger.info(f"Reading batch result from {output_jsonl}")
            try:
                result_obj = read_last_success_row(output_jsonl)
                response_text = result_obj.get('response_text', '{}')
                batch_result = json.loads(response_text)

                # Parse pages from result
                pages_data = batch_result.get("pages", [])
                if not pages_data:
                    logger.warning(f"Warning: batch result has no 'pages' key or empty pages")
                    logger.info(f"Batch result keys: {batch_result.keys()}")
                    logger.info(f"Full result: {batch_result}")
                    logger.info(f"Raw response_text: {response_text}")

                scratchpad = batch_result.get("scratchpad", "")

                # Convert to row format
                for i, page_data in enumerate(pages_data):
                    page_num = batch_start + i + 1
                    row = {
                        "activity_id": activity_id,  # Restore original ID
                        "cached_file": abs_pdf_path,
                        "page_start": page_num,
                        "page_end": page_num,
                        "section": section,
                        "subcategory_a": page_data.get("subcategory_A", page_data.get("category", "")),
                        "subcategory_b": page_data.get("subcategory_B", ""),
                        "score": page_data.get("informativeness", 0),
                        "scratchpad": scratchpad if i == 0 else "",
                        "title": title,
                    }
                    all_results.append(row)

                logger.info(f"Processed pages {batch_start+1}-{batch_end} ({len(pages_data)} pages extracted)")
            except Exception as e:
                logger.error(f"ERROR parsing batch result: {str(e)}")
                raise RuntimeError(f"Failed to parse batch result for pages {batch_start+1}-{batch_end}: {str(e)}") from e
        else:
            error_msg = f"No output generated for batch {batch_start+1}-{batch_end} at {output_jsonl}"
            if output_jsonl.exists():
                error_msg += f" (file exists but is empty)"
            logger.error(f"{error_msg}")
            raise RuntimeError(error_msg)

    # Validate we have results
    if not all_results:
        raise RuntimeError(f"Page categorization failed: 0 pages extracted from {total_pages} page PDF")

    # Save all results to single JSONL
    output_file = output_dir / "page_categories.jsonl"
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(all_results)

    logger.info(f"All {len(all_results)} pages categorized and saved to {output_file}")

    return all_results
