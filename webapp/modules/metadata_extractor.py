#!/usr/bin/env python3
"""
Extract activity metadata from uploaded PDF.
Uses loop_over_rows_to_call_model() from utils.
"""

from debug_utils import _print_ram


import logging

logger = logging.getLogger(__name__)

_print_ram("top of metadata extractor imports")
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
_print_ram("end of standard metadata extractor imports")

# Add utils to path
UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from extracting_and_grading_helper_functions import (
    loop_over_rows_to_call_model,
    make_executor,
    read_last_success_row,
)
_print_ram("end of extracting_and_grading_helper_functions imports")


def get_metadata_schema() -> dict:
    """JSON schema for activity metadata extraction."""
    return {
        "type": "object",
        "required": ["title"],
        "properties": {
            "title": {
                "type": "string",
                "maxLength": 500,
            },
            "participating_orgs": {
                "type": "array",
                "items": {"type": "string", "maxLength": 200},
                "maxItems": 20,
            },
            "country_locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "iso2_code": {"type": "string", "minLength": 2, "maxLength": 2},
                        "percentage": {"type": "number", "minimum": 0, "maximum": 100},
                    },
                    "required": ["iso2_code", "percentage"],
                },
            },
            "planned_start_date": {
                "type": "string",
                "maxLength": 50,
            },
            "planned_end_date": {
                "type": "string",
                "maxLength": 50,
            },
        },
    }


def get_metadata_prompt(activity_id: str) -> str:
    """Prompt for extracting basic activity metadata."""
    return """Return only valid JSON matching the provided schema.

TASK:
Extract basic information about this international aid activity from the uploaded PDF pages.

FIELDS TO EXTRACT:
1) title: The full activity/project title
2) participating_orgs: List of organizations involved (implementing partners, funders, stakeholders)
3) country_locations: Array of countries where activity takes place
   - iso2_code: Two-letter ISO country code (e.g., "KE" for Kenya, "UG" for Uganda)
   - percentage: Percentage of activity in this country (must sum to 100 across all countries)
4) planned_start_date: Project start date if mentioned (format: YYYY-MM-DD or as written)
5) planned_end_date: Project end date if mentioned (format: YYYY-MM-DD or as written)

RULES:
- title is required, all other fields are optional
- If multi-country, provide all countries with percentage splits
- If single country with no percentage mentioned, use 100%
- If information is not found, omit the field (except title)
- Do not infer or guess information not present in the document
"""


def extract_metadata_from_pdf(
    pdf_path: str,
    activity_id: str,
    output_dir: Path,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Extract metadata from a single PDF using loop_over_rows_to_call_model().

    Args:
        pdf_path: Path to uploaded PDF
        activity_id: Generated activity ID
        output_dir: Directory to save metadata.json
        model: Gemini model to use

    Returns:
        Dict with extracted metadata fields
    """
    # Build rows in format expected by loop_over_rows_to_call_model
    # For metadata, we just need the first few pages
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    total_pages = min(len(reader.pages), 5)  # Only first 5 pages for metadata

    # Use absolute path for cached_file (helper prepends LOCATION_PDFS otherwise)
    abs_pdf_path = str(Path(pdf_path).resolve())

    rows = [{
        "activity_id": activity_id,
        "cached_file": abs_pdf_path,
        "doc_title": "uploaded",
        "section": "Baseline",
        "activity_title": "uploaded",
        "page_start": i + 1,
    } for i in range(total_pages)]

    # Consolidate (will group all pages for this activity)
    from extracting_and_grading_helper_functions import consolidate_rows_by_activity
    chunked = consolidate_rows_by_activity(rows)

    # Build prompts dict
    prompts = {activity_id: get_metadata_prompt(activity_id)}

    # Get schema
    schema = get_metadata_schema()

    # Output file
    output_jsonl = output_dir / "metadata.jsonl"

    # Create executor
    execpool = ThreadPoolExecutor(max_workers=1)

    try:
        # Call loop_over_rows_to_call_model
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

    # Read result (helper wraps in container with response_text field)
    result = read_last_success_row(output_jsonl)

    # Parse response_text (contains the actual metadata as JSON string)
    if 'response_text' in result:
        _text = result['response_text'].strip()
        # Strip markdown code fences if the model wrapped the JSON in ```json ... ```
        if _text.startswith('```'):
            _text = _text.split('```')[1]
            if _text.startswith('json'):
                _text = _text[4:]
            _text = _text.strip()
        parsed_metadata = json.loads(_text)
    else:
        # No response_text means it's already parsed (shouldn't happen)
        parsed_metadata = result

    # Start with clean metadata dict (not the wrapper)
    metadata = {
        'activity_id': activity_id,
        'title': parsed_metadata.get('title', 'N/A'),
    }

    # Post-process: convert country_locations to pipe-separated format
    if "country_locations" in parsed_metadata and parsed_metadata["country_locations"]:
        location_parts = []
        for loc in parsed_metadata["country_locations"]:
            location_parts.append(loc["iso2_code"])
            location_parts.append(str(int(loc["percentage"])))
        metadata["country_location"] = "|".join(location_parts)
    else:
        metadata["country_location"] = None

    # Format participating_orgs as comma-separated string
    if "participating_orgs" in parsed_metadata and parsed_metadata["participating_orgs"]:
        metadata["participating_orgs"] = ", ".join(parsed_metadata["participating_orgs"])
    else:
        metadata["participating_orgs"] = None

    # Add optional fields
    metadata["planned_start_date"] = parsed_metadata.get("planned_start_date")
    metadata["planned_end_date"] = parsed_metadata.get("planned_end_date")
    metadata["implementing_org_type"] = None

    # Save to metadata.json (clean format, not JSONL wrapper)
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Metadata extracted and saved to {metadata_file}")

    return metadata
