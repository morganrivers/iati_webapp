#!/usr/bin/env python3
"""
Main orchestrator for processing uploaded PDFs in the webapp.
Runs the full extraction pipeline on a single PDF.

Pipeline:
0. Extract activity metadata (title, orgs, locations, dates)
1. Categorize pages (baseline/outcome, subcategories)
2. Generate summary (activity description)
3. Extract finance breakdown (sectors, allocations)
4. Extract baseline features (implementer, targets, risks)

All outputs saved to: extracted_pdf_data/{activity_id}/
"""

import logging
import hashlib
import shutil
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from pypdf import PdfReader
from model_loader import get_sector_clusters
from metadata_extractor import extract_metadata_from_pdf
from page_categorizer import categorize_single_pdf
from summary_generator import generate_activity_summary
from finance_extractor import extract_finance_breakdown
from sector_extractor import extract_sector_allocation, parse_sector_allocation
from feature_extractor import extract_baseline_features

logger = logging.getLogger(__name__)


def _parse_finance_raw(raw: dict, activity_id: str) -> dict:
    if 'response_text' in raw:
        result = json.loads(raw['response_text'])
        result['activity_id'] = activity_id
        return result
    return raw


def _parse_sector_raw(raw: dict) -> dict:
    if 'response_text' in raw:
        return json.loads(raw['response_text'])
    return raw


def _parse_misc_data(data: dict) -> tuple:
    if 'response_text' in data:
        try:
            parsed = json.loads(data['response_text'])
            return parsed.get('complexity_details', ''), parsed.get('how_integrated_description', '')
        except json.JSONDecodeError:
            text = data.get('response_text', '')
            return text, text
    text = data.get('response', '')
    return text, text


def _parse_feature_text(data: dict) -> str:
    return data.get('response_text', data.get('response', ''))


def generate_activity_id_from_content(pdf_content: bytes) -> str:
    """
    Generate activity_id from PDF content hash.
    Same PDF = same ID = can reuse cached results.
    Format: webapp_{content_hash}
    """
    content_hash = hashlib.md5(pdf_content).hexdigest()[:12]
    return f"webapp_{content_hash}"


def process_uploaded_pdf(
    pdf_file,
    output_base_dir: Path,
    model: str = "gemini-2.5-flash",
    skip_if_exists: bool = True,
    progress_callback: Optional[callable] = None,
    log_callback: Optional[callable] = None,
    partial_result_callback: Optional[callable] = None,
    stop_after_phase_3: bool = False,
) -> Dict[str, Any]:
    """
    Process a single uploaded PDF through the full extraction pipeline.

    Args:
        pdf_file: Streamlit UploadedFile object or file path
        output_base_dir: Base directory for all extracted data (e.g., webapp/extracted_pdf_data)
        model: Gemini model to use for all extractions
        skip_if_exists: If True, skip steps that already have output files and reuse cached results

    Returns:
        Dict containing all extracted information:
        {
            'activity_id': str,
            'metadata': dict,
            'page_categories': list,
            'summary': str,
            'finance': dict,
            'features': dict,
            'output_dir': Path,
            'cached': bool,  # True if loaded from cache
            'num_pages': int,
        }
    """

    # Define log helper (used throughout)
    def log(msg):
        """Helper to log to both stdout and callback"""
        print(msg)
        if log_callback:
            log_callback(msg)

    # Read PDF content for hashing
    if hasattr(pdf_file, 'read'):
        # Streamlit UploadedFile - read content for hashing
        pdf_file.seek(0)  # Reset to beginning
        pdf_content = pdf_file.read()
        pdf_file.seek(0)  # Reset again for later use
        filename = pdf_file.name
    else:
        # File path - read content
        with open(pdf_file, 'rb') as f:
            pdf_content = f.read()
        filename = Path(pdf_file).name

    # Generate activity ID from content hash (same PDF = same ID)
    activity_id = generate_activity_id_from_content(pdf_content)
    output_dir = output_base_dir / activity_id

    # Check if this PDF was already processed (validate cache completeness)
    if skip_if_exists and output_dir.exists():
        # Check if all required files exist
        required_files = [
            output_dir / "metadata.json",
            output_dir / "page_categories.jsonl",
            output_dir / "summary.jsonl",
            output_dir / "finance_breakdown.jsonl",
            output_dir / "implementer_performance.jsonl",
            output_dir / "targets.jsonl",
            output_dir / "risks.jsonl",
            output_dir / "context.jsonl",
            output_dir / "finance_qualitative.jsonl",
            output_dir / "misc.jsonl",
        ]

        all_exist = all(f.exists() and f.stat().st_size > 0 for f in required_files)

        if all_exist:
            # Validate page_categories has actual content (not just whitespace)
            page_cat_file = output_dir / "page_categories.jsonl"
            try:
                with jsonlines.open(page_cat_file, 'r') as reader:
                    page_cats = list(reader)
                if not page_cats:
                    log(f"⚠️ Cached page_categories is empty, invalidating cache...")
                    all_exist = False
            except Exception as e:
                log(f"⚠️ Failed to validate cached page_categories: {e}")
                all_exist = False

        if all_exist:
            if log_callback:
                log_callback(f"{'='*60}")
                log_callback(f"✓ Found complete cached results for: {filename}")
                log_callback(f"Activity ID: {activity_id}")
                log_callback(f"Loading from: {output_dir}")
                log_callback(f"{'='*60}")

            # Load cached results
            result = load_cached_results(output_dir, activity_id, log_callback)
            result['cached'] = True
            return result
        else:
            # Incomplete cache, will re-process
            log(f"⚠️ Found incomplete cache for {activity_id}, re-processing...")
            missing = [f.name for f in required_files if not (f.exists() and f.stat().st_size > 0)]
            log(f"   Missing or empty files: {', '.join(missing)}")

            # Delete corrupted page_categories.jsonl if it exists but is empty
            if (output_dir / "page_categories.jsonl").exists():
                try:
                    with jsonlines.open(output_dir / "page_categories.jsonl", 'r') as reader:
                        if not list(reader):
                            log(f"   Deleting empty page_categories.jsonl")
                            (output_dir / "page_categories.jsonl").unlink()
                except Exception as e:
                    log(f"⚠️ Could not check/clean page_categories.jsonl: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"{'='*60}")
    log(f"Processing PDF: {filename}")
    log(f"Activity ID: {activity_id}")
    log(f"Output directory: {output_dir}")
    log(f"{'='*60}")

    # Save uploaded PDF
    pdf_path = output_dir / "uploaded.pdf"
    if not pdf_path.exists():
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)
        log(f"✓ PDF saved to {pdf_path}")

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    log(f"🗎 PDF has {num_pages} pages")

    # Estimate processing time (~7 seconds per page for categorization)
    estimated_minutes = (num_pages * 7) / 60
    log(f"⏱️  Estimated processing time: ~{estimated_minutes:.1f} minutes")

    result = {
        'activity_id': activity_id,
        'output_dir': output_dir,
        'pdf_path': str(pdf_path),
        'num_pages': num_pages,
        'cached': False,
        'metadata': {},
        'page_categories': [],
        'summary': '',
        'finance': {},
        'sector_allocation': {},
        'features': {},
    }

    # -------------------------------------------------------------------------
    # PHASE 0: Extract Metadata
    # -------------------------------------------------------------------------
    if progress_callback:
        progress_callback(1, "📋 Phase 1/5: Extracting metadata...")
    log("[PHASE 0] Extracting activity metadata...")
    metadata_file = output_dir / "metadata.json"

    try:
        if skip_if_exists and metadata_file.exists():
            log(f"  → Skipping (already exists): {metadata_file}")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = extract_metadata_from_pdf(
                pdf_path=str(pdf_path),
                activity_id=activity_id,
                output_dir=output_dir,
                model=model,
            )

        result['metadata'] = metadata
        log(f"  ✓ Extracted: title='{metadata.get('title', 'N/A')[:50]}...'")
        log(f"              locations={metadata.get('country_location', 'N/A')}")

        # Report partial progress
        if partial_result_callback:
            partial_result_callback(dict(result))
    except Exception as e:
        log(f"  ❌ ERROR in Phase 0 (Metadata): {str(e)}")
        raise RuntimeError(f"Phase 0 (Metadata extraction) failed: {str(e)}") from e

    # -------------------------------------------------------------------------
    # PHASE 1: Categorize Pages
    # -------------------------------------------------------------------------
    if progress_callback:
        progress_callback(2, "🗃️ Phase 2/5: Categorizing pages...")
    log("[PHASE 1] Categorizing PDF pages...")
    page_categories_file = output_dir / "page_categories.jsonl"

    try:
        if skip_if_exists and page_categories_file.exists() and page_categories_file.stat().st_size > 0:
            log(f"  → Loading from cache: {page_categories_file}")
            with jsonlines.open(page_categories_file, 'r') as reader:
                page_categories = list(reader)

            # Validate that we have pages
            if not page_categories:
                log(f"  ⚠️ Cached file is empty, re-running categorization...")
                page_categories = categorize_single_pdf(
                    pdf_path=str(pdf_path),
                    activity_id=activity_id,
                    title=metadata.get('title', ''),
                    output_dir=output_dir,
                    model=model,
                )
        else:
            page_categories = categorize_single_pdf(
                pdf_path=str(pdf_path),
                activity_id=activity_id,
                title=metadata.get('title', ''),
                output_dir=output_dir,
                model=model,
            )

        result['page_categories'] = page_categories
        log(f"  ✓ Categorized {len(page_categories)} pages")

        # Validate we have at least 1 page
        if not page_categories:
            raise RuntimeError(f"Page categorization returned 0 pages for PDF with {num_pages} pages. Check that the PDF is readable.")

        # Report partial progress
        if partial_result_callback:
            partial_result_callback(dict(result))
    except Exception as e:
        log(f"  ❌ ERROR in Phase 1 (Page Categorization): {str(e)}")
        raise RuntimeError(f"Phase 1 (Page Categorization) failed: {str(e)}") from e

    # -------------------------------------------------------------------------
    # PHASE 2: Generate Summary
    # -------------------------------------------------------------------------
    if progress_callback:
        progress_callback(3, "🗒 Phase 3/5: Generating summary...")
    log("[PHASE 2] Generating activity summary...")
    summary_file = output_dir / "summary.jsonl"

    try:
        if skip_if_exists and summary_file.exists() and summary_file.stat().st_size > 0:
            log(f"  → Skipping (already exists): {summary_file}")
            with jsonlines.open(summary_file, 'r') as reader:
                summary_data = list(reader)[0]
                summary = summary_data.get('chatgpt_description', '')
        else:
            summary = generate_activity_summary(
                pdf_path=str(pdf_path),
                activity_id=activity_id,
                title=metadata.get('title', ''),
                page_categories=page_categories,
                output_dir=output_dir,
                model=model,
            )

        result['summary'] = summary
        log(f"  ✓ Summary generated ({len(summary)} chars)")

        # Report partial progress
        if partial_result_callback:
            partial_result_callback(dict(result))
    except Exception as e:
        log(f"  ❌ ERROR in Phase 2 (Summary Generation): {str(e)}")
        import traceback
        log(f"  Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Phase 2 (Summary Generation) failed: {str(e)}") from e

    # -------------------------------------------------------------------------
    # PHASE 3: Extract Finance Breakdown
    # -------------------------------------------------------------------------
    if progress_callback:
        progress_callback(4, "💰 Phase 4/5: Extracting finance...")
    log("[PHASE 3] Extracting finance breakdown...")
    finance_file = output_dir / "finance_breakdown.jsonl"

    try:
        if skip_if_exists and finance_file.exists():
            log(f"  → Skipping (already exists): {finance_file}")
            with jsonlines.open(finance_file, 'r') as reader:
                finance = _parse_finance_raw(list(reader)[0], activity_id)
        else:
            finance = extract_finance_breakdown(
                pdf_path=str(pdf_path),
                activity_id=activity_id,
                title=metadata.get('title', ''),
                chatgpt_description=summary,
                page_categories=page_categories,
                output_dir=output_dir,
                model=model,
            )

        result['finance'] = finance
        total_allocation = finance.get('total_allocation', {})
        log(f"  ✓ Finance extracted: {total_allocation.get('amount', 0)} {total_allocation.get('currency', 'N/A')}")

        # Report partial progress
        if partial_result_callback:
            partial_result_callback(dict(result))
    except Exception as e:
        log(f"  ❌ ERROR in Phase 3 (Finance Extraction): {str(e)}")
        raise RuntimeError(f"Phase 3 (Finance Extraction) failed: {str(e)}") from e

    # -------------------------------------------------------------------------
    # PHASE 3b: Extract Sector Allocation (expenditure % across sector clusters)
    # -------------------------------------------------------------------------
    log("[PHASE 3b] Extracting sector allocation...")
    sector_file = output_dir / "sector_allocation.jsonl"

    try:
        if skip_if_exists and sector_file.exists():
            log(f"  → Skipping (already exists): {sector_file}")
            with jsonlines.open(sector_file, 'r') as reader:
                sector_allocation = parse_sector_allocation(
                    _parse_sector_raw(list(reader)[0]), get_sector_clusters()
                )
        else:
            sector_allocation = extract_sector_allocation(
                pdf_path=str(pdf_path),
                activity_id=activity_id,
                title=metadata.get('title', ''),
                chatgpt_description=summary,
                page_categories=page_categories,
                output_dir=output_dir,
                model=model,
            )

        result['sector_allocation'] = sector_allocation
        log(f"  ✓ Sector allocation extracted: {len(sector_allocation)} non-zero clusters")

        if partial_result_callback:
            partial_result_callback(dict(result))
    except Exception as e:
        log(f"  ❌ ERROR in Phase 3b (Sector Allocation): {str(e)}")
        raise RuntimeError(f"Phase 3b (Sector Allocation) failed: {str(e)}") from e

    # -------------------------------------------------------------------------
    # Stop here if requested (for webapp confirmation flow)
    # -------------------------------------------------------------------------
    if stop_after_phase_3:
        log(f"{'='*60}")
        log(f"✓ PHASES 0-3 COMPLETE for {activity_id}")
        log(f"  Awaiting user confirmation to continue with feature extraction...")
        log(f"{'='*60}")
        return result

    # -------------------------------------------------------------------------
    # PHASE 4: Extract Baseline Features
    # -------------------------------------------------------------------------
    if progress_callback:
        progress_callback(5, "🔍 Phase 5/5: Extracting features...")
    log("[PHASE 4] Extracting baseline features...")
    features_prefix = output_dir / "features"

    # Check if all feature files exist
    feature_files = [
        output_dir / "implementer_performance.jsonl",
        output_dir / "targets.jsonl",
        output_dir / "risks.jsonl",
        output_dir / "context.jsonl",
        output_dir / "finance_qualitative.jsonl",
        output_dir / "misc.jsonl",
    ]
    all_exist = all(f.exists() for f in feature_files)

    try:
        if skip_if_exists and all_exist:
            log(f"  → Skipping (all feature files already exist)")
            features = {}
            for ffile in feature_files:
                fname = ffile.stem
                with jsonlines.open(ffile, 'r') as reader:
                    data = list(reader)[0]
                if fname == "misc":
                    features['complexity'], features['integratedness'] = _parse_misc_data(data)
                elif fname == "finance_qualitative":
                    features['finance'] = _parse_feature_text(data)
                else:
                    features[fname] = _parse_feature_text(data)
        else:
            features = extract_baseline_features(
                pdf_path=str(pdf_path),
                activity_id=activity_id,
                metadata_dict=metadata,
                chatgpt_description=summary,
                page_categories=page_categories,
                output_dir=output_dir,
                model=model,
            )

        result['features'] = features
        log(f"  ✓ Features extracted: {', '.join(features.keys())}")

        # Report partial progress
        if partial_result_callback:
            partial_result_callback(dict(result))
    except Exception as e:
        log(f"  ❌ ERROR in Phase 4 (Feature Extraction): {str(e)}")
        raise RuntimeError(f"Phase 4 (Feature Extraction) failed: {str(e)}") from e

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log(f"{'='*60}")
    log(f"✓ PIPELINE COMPLETE for {activity_id}")
    log(f"  All outputs saved to: {output_dir}")
    log(f"{'='*60}")

    return result


def load_cached_results(output_dir: Path, activity_id: str, log_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Load previously extracted results from disk.

    Args:
        output_dir: Directory containing cached results
        activity_id: Activity identifier

    Returns:
        Complete result dict
    """
    result = {
        'activity_id': activity_id,
        'output_dir': output_dir,
        'pdf_path': str(output_dir / "uploaded.pdf"),
    }

    # Load metadata
    metadata_file = output_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            result['metadata'] = json.load(f)

    # Load page categories
    page_cat_file = output_dir / "page_categories.jsonl"
    if page_cat_file.exists():
        with jsonlines.open(page_cat_file, 'r') as reader:
            result['page_categories'] = list(reader)

    # Get num_pages from page categories
    if 'page_categories' in result:
        result['num_pages'] = len(result['page_categories'])
    else:
        reader = PdfReader(result['pdf_path'])
        result['num_pages'] = len(reader.pages)

    # Load summary
    summary_file = output_dir / "summary.jsonl"
    if summary_file.exists() and summary_file.stat().st_size > 0:
        try:
            with jsonlines.open(summary_file, 'r') as reader:
                lines = list(reader)
                if lines:
                    summary_data = lines[0]
                    result['summary'] = summary_data.get('chatgpt_description', '')
                else:
                    result['summary'] = ''
        except Exception as e:
            logger.warning(f"Warning: Failed to load summary from cache: {e}")
            result['summary'] = ''

    # Load finance
    finance_file = output_dir / "finance_breakdown.jsonl"
    if finance_file.exists() and finance_file.stat().st_size > 0:
        try:
            with jsonlines.open(finance_file, 'r') as reader:
                lines = list(reader)
            result['finance'] = _parse_finance_raw(lines[0], activity_id) if lines else {}
        except Exception as e:
            logger.warning("Failed to load finance from cache: %s", e)
            result['finance'] = {}

    # Load sector allocation
    sector_file = output_dir / "sector_allocation.jsonl"
    if sector_file.exists() and sector_file.stat().st_size > 0:
        try:
            with jsonlines.open(sector_file, 'r') as reader:
                lines = list(reader)
            result['sector_allocation'] = (
                parse_sector_allocation(_parse_sector_raw(lines[0]), get_sector_clusters())
                if lines else {}
            )
        except Exception as e:
            logger.warning("Failed to load sector allocation from cache: %s", e)
            result['sector_allocation'] = {}

    # Load features
    features = {}
    for feature_name in ['implementer_performance', 'targets', 'risks', 'context', 'finance']:
        feature_file = output_dir / f"{feature_name}.jsonl"
        if feature_file.exists() and feature_file.stat().st_size > 0:
            try:
                with jsonlines.open(feature_file, 'r') as reader:
                    lines = list(reader)
                if lines:
                    features[feature_name] = _parse_feature_text(lines[0])
            except Exception as e:
                logger.warning("Failed to load %s from cache: %s", feature_name, e)

    misc_file = output_dir / "misc.jsonl"
    if misc_file.exists() and misc_file.stat().st_size > 0:
        try:
            with jsonlines.open(misc_file, 'r') as reader:
                lines = list(reader)
            if lines:
                features['complexity'], features['integratedness'] = _parse_misc_data(lines[0])
        except Exception as e:
            logger.warning("Failed to load misc features from cache: %s", e)

    finance_qual_file = output_dir / "finance_qualitative.jsonl"
    if finance_qual_file.exists() and finance_qual_file.stat().st_size > 0:
        try:
            with jsonlines.open(finance_qual_file, 'r') as reader:
                lines = list(reader)
            if lines:
                features['finance'] = _parse_feature_text(lines[0])
        except Exception as e:
            logger.warning("Failed to load finance_qualitative from cache: %s", e)

    result['features'] = features

    # Print summary
    num_pages = len(result.get('page_categories', []))
    num_features = len(features)
    if log_callback:
        log_callback(f"✓ Loaded cached results: {num_pages} pages, {num_features} features")
    logger.info(f"Loaded cached results: {num_pages} pages, {num_features} features")

    return result
