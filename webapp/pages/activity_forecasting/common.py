import logging
import json
import asyncio
import traceback
from pathlib import Path
from datetime import datetime

import streamlit as st

from project_manager import create_new_project, save_project_state_temp
from webapp_pipeline import process_uploaded_pdf
from location_features import KEEP_REPORTING_ORGS, ORG_NAME_TO_DUMMY
from modules.feature_grader import grade_features_with_llm
from modules.feature_extractor import extract_baseline_features

logger = logging.getLogger(__name__)

USE_CACHED_PDFS = False

from webapp_paths import EXTRACTED_PDF_DIR

# IATI standard activity scope codes
SCOPE_LABELS = {
    1: "1 - Global",
    2: "2 - Regional",
    3: "3 - Multi-national",
    4: "4 - National",
    5: "5 - Sub-national: Multi first-level admin",
    6: "6 - Sub-national: Single first-level admin",
    7: "7 - Sub-national: Single second-level admin",
}


@st.cache_data
def _compute_org_counts(_training_data):
    """Pre-compute org counts from training data. Called once; result is cached for the session."""
    org_cols = ['rep_org_0', 'rep_org_1', 'rep_org_2']
    if not all(col in _training_data.columns for col in org_cols):
        return {}
    org_counts = {}
    for org_name in KEEP_REPORTING_ORGS:
        dummies = ORG_NAME_TO_DUMMY.get(org_name, {})
        one_col = next((c for c, v in dummies.items() if v == 1), None)
        if one_col and one_col in _training_data.columns:
            org_counts[org_name] = int(_training_data[one_col].sum())
        else:
            mask = (_training_data[org_cols] == 0).all(axis=1)
            org_counts[org_name] = int(mask.sum())
    return org_counts


@st.cache_data
def _compute_region_totals(_training_data):
    """Pre-compute region sums from training data. Called once; result is cached."""
    region_cols = ['region_AFE', 'region_AFW', 'region_EAP', 'region_ECA',
                   'region_LAC', 'region_MENA', 'region_SAS']
    region_names = ['Africa East', 'Africa West', 'East Asia & Pacific',
                    'Europe & Central Asia', 'Latin America & Caribbean',
                    'Middle East & North Africa', 'South Asia']
    return {
        name: float(_training_data[col].sum())
        for col, name in zip(region_cols, region_names)
        if col in _training_data.columns
    }


class _NamedBytesIO:
    """BytesIO wrapper with a .name attribute for process_uploaded_pdf compatibility."""
    def __init__(self, data: bytes, name: str):
        from io import BytesIO
        self._buf = BytesIO(data)
        self.name = name

    def read(self, *args):
        return self._buf.read(*args)

    def seek(self, *args):
        return self._buf.seek(*args)


def _run_phases_0_3_background(snapshot: dict, extraction_state: dict) -> None:
    """
    Runs in a background thread so it survives Streamlit page switches.
    Calls process_uploaded_pdf (phases 0-3, stop_after_phase_3=True).
    Writes only to `extraction_state` (a plain dict) — never touches st.session_state.
    """
    logs = extraction_state['logs']

    def log_cb(msg: str) -> None:
        logs.append(msg)
        if len(logs) > 100:
            logs.pop(0)

    def progress_cb(phase, msg):
        extraction_state['progress'] = (phase, msg)

    try:
        pdf_file = _NamedBytesIO(snapshot['pdf_bytes'], snapshot['filename'])
        result = process_uploaded_pdf(
            pdf_file=pdf_file,
            output_base_dir=snapshot['output_base_dir'],
            model="gemini-2.5-flash",
            skip_if_exists=USE_CACHED_PDFS,
            progress_callback=progress_cb,
            log_callback=log_cb,
            partial_result_callback=lambda partial: extraction_state.update({'partial_result': dict(partial)}),
            stop_after_phase_3=False,
        )
        result['status'] = 'phases_0_3_complete'
        extraction_state['result'] = result
        extraction_state['done'] = True
    except Exception as exc:
        extraction_state['error'] = f"{exc}\n{traceback.format_exc()}"
        extraction_state['done'] = True


def _run_phase4_background(snapshot: dict, grading_state: dict) -> None:
    """
    Runs in a background thread so it survives Streamlit script interruptions.
    Writes only to `grading_state` (a plain dict) — never touches st.session_state.
    """
    logs = grading_state['logs']

    def log_cb(msg: str) -> None:
        logs.append(msg)
        if len(logs) > 100:
            logs.pop(0)

    try:
        result = snapshot['extraction_result']
        logger.info("[DEBUG PHASE4-THREAD START]")
        logger.info(f"result activity_id = {(result or {}).get('activity_id', '<None>')!r}")
        logger.info(f"result output_dir  = {(result or {}).get('output_dir', '<None>')!r}")

        log_cb("Starting feature extraction...")
        features = extract_baseline_features(
            pdf_path=result['pdf_path'],
            activity_id=result['activity_id'],
            metadata_dict=snapshot['confirmed_metadata'],
            chatgpt_description=result['summary'],
            page_categories=result['page_categories'],
            output_dir=Path(result['output_dir']),
            model="gemini-2.5-flash",
            log_callback=log_cb,
        )
        grading_state['features'] = features
        log_cb("Starting LLM grading of features...")

        feature_grades = asyncio.run(
            grade_features_with_llm(
                activity_id=result['activity_id'],
                title=snapshot['title'],
                chatgpt_description=result['summary'],
                features=features,
                metadata=snapshot['confirmed_metadata'],
                model="gemini-2.5-flash",
                log_callback=log_cb,
            )
        )

        log_cb(f"✅ Grading complete! Received {len(feature_grades)} grades")

        grades_file = Path(result['output_dir']) / "feature_grades.jsonl"
        try:
            with grades_file.open('w', encoding='utf-8') as f:
                json.dump({'activity_id': result['activity_id'], 'grades': feature_grades}, f)
            log_cb(f"💾 Saved grades to {grades_file.name}")
        except Exception as save_err:
            log_cb(f"⚠️ Could not save grades to disk: {save_err}")

        grading_state['grades'] = feature_grades
        grading_state['done'] = True

    except Exception as exc:
        grading_state['error'] = f"{exc}\n{traceback.format_exc()}"
        grading_state['done'] = True


def _ensure_project_folder():
    """Create a project folder if we're in 'new project' mode and don't have one yet."""
    if st.session_state.get('creating_new_project') and not st.session_state.get('selected_project_folder'):
        project_name = st.session_state.get('pending_project_name') or f"Untitled - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        folder_name = create_new_project(project_name)
        st.session_state.selected_project_folder = folder_name
        st.session_state.project_name = project_name
        st.session_state.creating_new_project = False
        st.session_state.pending_project_name = None

def _save_and_rerun():
    _ensure_project_folder()
    if st.session_state.get('selected_project_folder'):
        save_project_state_temp(st.session_state.selected_project_folder)
    st.rerun()


def _parse_pdf_date(date_str: str):
    """Parse a PDF metadata date string (D:YYYYMMDDHHmmSS...) to a datetime.date, or None."""
    if not date_str:
        return None
    s = date_str.strip()
    if s.startswith("D:"):
        s = s[2:]
    # Take just the first 8 chars: YYYYMMDD
    s = s[:8]
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except Exception:
        return None


def _extract_pdf_creation_date(pdf_bytes: bytes):
    """Return (creation_date, source) from PDF metadata bytes, or (None, None)."""
    try:
        from pypdf import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_bytes))
        meta = reader.metadata or {}
        creation_raw = meta.get('/CreationDate') or meta.get('CreationDate')
        if creation_raw:
            d = _parse_pdf_date(str(creation_raw))
            if d:
                return d, 'creation'
        mod_raw = meta.get('/ModDate') or meta.get('ModDate')
        if mod_raw:
            d = _parse_pdf_date(str(mod_raw))
            if d:
                return d, 'modification'
    except Exception as e:
        logger.warning(f"Could not read PDF metadata dates: {e}")
    return None, None
