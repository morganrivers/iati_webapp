#!/usr/bin/env python3
"""
Extract sector-cluster expenditure allocation from baseline pages.
Copied from finance_extractor.py; the LLM allocates expenditure directly across
the model's sector clusters (names sourced from get_sector_clusters()), so the
result maps one-to-one onto the Sector Allocation UI. No embedding / KMeans.
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Add webapp root and utils to path
WEBAPP_DIR = Path(__file__).resolve().parent.parent
UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "utils"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from model_loader import get_sector_clusters

from extracting_and_grading_helper_functions import (
    loop_over_rows_to_call_model,
    consolidate_rows_by_activity,
    read_last_success_row,
)


def get_sector_allocation_schema(clusters: List[str]) -> dict:
    """JSON schema: an allocation entry per non-zero sector cluster."""
    return {
        "type": "object",
        "required": ["sector_allocations"],
        "properties": {
            "sector_allocations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["cluster", "percentage"],
                    "properties": {
                        "cluster": {"type": "string", "enum": clusters},
                        "percentage": {"type": "number", "minimum": 0, "maximum": 100},
                    },
                },
            },
        },
    }


def get_sector_allocation_prompt(chunked_by_activity_id: List[dict], clusters: List[str]) -> Dict[str, str]:
    """activity_id -> prompt string."""
    cluster_lines = "\n".join(f"- {c}" for c in clusters)
    prompts: Dict[str, str] = {}
    for bundle in chunked_by_activity_id:
        aid = bundle.get("activity_id")
        if not aid:
            continue
        title = (bundle.get("activity_title") or bundle.get("title") or "").strip()
        desc = (bundle.get("chatgpt_description") or "").strip()
        prompts[aid] = (
            "Return only valid JSON matching the provided schema.\n\n"
            "CONTEXT:\n"
            f"- ACTIVITY TITLE: {title}\n"
            f"- ACTIVITY DESCRIPTION: {desc}\n\n"
            "TASK:\n"
            "Estimate how this international aid activity's total expenditure is allocated "
            "across the following sector clusters. Return an entry for each cluster that "
            "receives a non-zero share; omit clusters that receive nothing.\n\n"
            "SECTOR CLUSTERS (use the 'cluster' value exactly as spelled here):\n"
            f"{cluster_lines}\n\n"
            "RULES:\n"
            "- percentage is that cluster's share of total expenditure (0-100).\n"
            "- The percentages across all returned clusters must sum to approximately 100.\n"
            "- Base the allocation on evidence in the pages; do not invent detail the document does not imply.\n"
        )
    return prompts


def extract_sector_allocation(
    pdf_path: str,
    activity_id: str,
    title: str,
    chatgpt_description: str,
    page_categories: List[Dict[str, Any]],
    output_dir: Path,
    model: str = "gemini-2.5-flash",
) -> Dict[str, float]:
    """
    Extract sector-cluster expenditure allocation from baseline pages.

    Returns:
        Dict mapping sector-cluster name -> percentage (0-100).
    """
    clusters = get_sector_clusters()

    # Filter to finance pages (same pages that carry the expenditure breakdown)
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
        print("  ⚠ No finance pages found, using first 10 pages")
        relevant_pages = page_categories[:10]

    print(f"  Using {len(relevant_pages)} pages for sector allocation extraction")

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
    for bundle in chunked:
        bundle["chatgpt_description"] = chatgpt_description

    # Prompts and schema built from the model's own cluster list
    prompts = get_sector_allocation_prompt(chunked, clusters)
    schema = get_sector_allocation_schema(clusters)

    # Output file
    output_jsonl = output_dir / "sector_allocation.jsonl"

    # Create executor
    execpool = ThreadPoolExecutor(max_workers=1)

    try:
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
    if 'response_text' in result:
        parsed = json.loads(result['response_text'])
    else:
        parsed = result

    return parse_sector_allocation(parsed, clusters)


def parse_sector_allocation(parsed: Dict[str, Any], clusters: List[str]) -> Dict[str, float]:
    """Convert the raw {'sector_allocations': [...]} response into {cluster: percentage}."""
    known = set(clusters)
    allocation: Dict[str, float] = {}
    for entry in parsed.get("sector_allocations", []) or []:
        cluster = entry.get("cluster")
        pct = entry.get("percentage")
        if cluster in known and pct is not None:
            allocation[cluster] = allocation.get(cluster, 0.0) + float(pct)
    print(f"  ✓ Sector allocation extracted: {len(allocation)} non-zero clusters")
    return allocation
