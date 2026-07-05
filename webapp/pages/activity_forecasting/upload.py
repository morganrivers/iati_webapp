import logging
import os
import time
import threading
from datetime import datetime

import streamlit as st

from utils import parse_location_string, notify_telegram, LLM_SESSION_CAP
from project_manager import (load_project_name, save_project_name,
                             load_project_state, save_project_state)
from model_loader import get_sector_clusters

from .common import EXTRACTED_PDF_DIR, _extract_pdf_creation_date, _run_phases_0_3_background

logger = logging.getLogger(__name__)


def render_llm_upload_section(_llm_running: bool, model_metadata: dict = None) -> None:
    # Params:
    #   _llm_running  — disables file_uploader + extract button while running
    #   model_metadata — for train_medians (used to set default end date from PDF creation date)
    # Returns: None

    st.subheader("Upload & Auto-Extract Features with LLM")

    # Password protection for LLM feature
    if not st.session_state.llm_authenticated:
        with st.form("auth_form"):
            password = st.text_input(
                "Enter Password for LLM Access",
                type="password",
                help="Contact the administrator for the password"
            )
            submitted = st.form_submit_button("Authenticate")

            if submitted:
                correct_password = os.getenv("APP_PASSWORD")
                if not correct_password:
                    st.error("❌ APP_PASSWORD not configured in environment")
                    # st.write("Debug — ALL env keys:", sorted(os.environ.keys()))
                elif password == correct_password:
                    # Load API keys from environment
                    google_api_key = os.getenv("GOOGLE_API_KEY")
                    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

                    if google_api_key:
                        os.environ["GOOGLE_API_KEY"] = google_api_key
                        logger.info("Loaded GOOGLE_API_KEY")
                    else:
                        st.warning("⚠️ GOOGLE_API_KEY not found in .env file")

                    if deepseek_api_key:
                        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
                        logger.info("Loaded DEEPSEEK_API_KEY")
                    else:
                        st.warning("⚠️ DEEPSEEK_API_KEY not found in .env file")

                    st.session_state.llm_authenticated = True
                    st.success("✅ Authenticated! API keys loaded. You can now use LLM extraction.")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")

    if st.session_state.llm_authenticated:
        # Show which API keys are available
        keys_available = []
        if os.getenv("GOOGLE_API_KEY"):
            keys_available.append("Gemini")
        if os.getenv("DEEPSEEK_API_KEY"):
            keys_available.append("DeepSeek")

        if keys_available:
            st.success(f"🔓 LLM features unlocked ({', '.join(keys_available)} available)")
        else:
            st.warning("🔓 LLM features unlocked (but no API keys found in environment)")

        uploaded_file = st.file_uploader(
            "Upload a PDF document for automatic feature extraction",
            type=["pdf"],
            help="The LLM will analyze the document and extract feature grades. Locked fields won't be overwritten.",
            disabled=_llm_running,
        )

        if uploaded_file:
            # On first upload of a new file, set start/end dates from PDF metadata
            _pdf_file_id = getattr(uploaded_file, 'file_id', None) or uploaded_file.name
            if st.session_state.get('_pdf_date_file_id') != _pdf_file_id:
                uploaded_file.seek(0)
                _pdf_bytes_for_meta = uploaded_file.read()
                uploaded_file.seek(0)
                _pdf_date, _pdf_date_src = _extract_pdf_creation_date(_pdf_bytes_for_meta)
                if _pdf_date and not st.session_state.field_locks.get('start_date', False):
                    st.session_state['input_start_date'] = _pdf_date
                    st.session_state.field_edited['start_date'] = True
                    # Compute end date = start + train median duration
                    _median_dur = (model_metadata or {}).get('train_medians', {}).get('planned_duration', 8.46)
                    from datetime import timedelta
                    try:
                        _end_date = _pdf_date.replace(year=_pdf_date.year + int(_median_dur))
                        _extra_days = round((_median_dur % 1) * 365.25)
                        _end_date = _end_date + timedelta(days=_extra_days)
                    except ValueError:
                        _end_date = _pdf_date + timedelta(days=round(_median_dur * 365.25))
                    if not st.session_state.field_locks.get('planned_end_date', False):
                        st.session_state['input_planned_end_date'] = _end_date
                        st.session_state.field_edited['planned_end_date'] = True
                    logger.info(f"📅 PDF metadata date ({_pdf_date_src}): {_pdf_date} end: {_end_date}")
                st.session_state['_pdf_date_file_id'] = _pdf_file_id

            if not _llm_running:
                st.caption("⚠️ While processing runs, project name editing and save buttons will be disabled.")
            if st.button("Extract Features with LLM", disabled=_llm_running):
                if st.session_state.llm_call_count >= LLM_SESSION_CAP:
                    st.error(f"❌ Session limit reached ({LLM_SESSION_CAP} LLM calls). Refresh the page to reset.")
                    st.stop()
                # Load API keys from environment (ensure they're set)
                google_api_key = os.getenv("GOOGLE_API_KEY")
                deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

                if not google_api_key:
                    st.error("❌ GOOGLE_API_KEY not found in .env file")
                    st.stop()

                if deepseek_api_key:
                    logger.info("DeepSeek API key loaded for this extraction")
                else:
                    logger.warning("DeepSeek API key not found - only Gemini will be available")

                # Read bytes now — UploadedFile is not thread-safe
                uploaded_file.seek(0)
                pdf_bytes = uploaded_file.read()

                extraction_state = {
                    'logs': [],
                    'done': False,
                    'result': None,
                    'error': None,
                    'progress': None,
                }
                st.session_state.extraction_state = extraction_state
                st.session_state.extraction_in_progress = True
                st.session_state.llm_call_count += 1
                notify_telegram(
                    f"🔔 LLM extraction triggered\n"
                    f"Session calls so far: {st.session_state.llm_call_count}\n"
                    f"File: {uploaded_file.name}"
                )

                snapshot = {
                    'pdf_bytes': pdf_bytes,
                    'filename': uploaded_file.name,
                    'output_base_dir': EXTRACTED_PDF_DIR,
                }
                t = threading.Thread(
                    target=_run_phases_0_3_background,
                    args=(snapshot, extraction_state),
                    daemon=True,
                )
                t.start()
                st.rerun()

            # ---- end of button handler ----


def poll_extraction_phases_0_3() -> None:
    # Params: (none) — reads st.session_state.extraction_state / extraction_in_progress
    # Returns: None  (writes result back to st.session_state; calls st.rerun on completion)
    
    if st.session_state.extraction_in_progress:
        es = st.session_state.extraction_state
        log_box = st.empty()
        progress_box = st.empty()

        while not es['done']:
            current_logs = list(es['logs'])
            _prog = es.get('progress')
            if _prog:
                progress_box.progress(min(_prog[0] * 20, 95), text=_prog[1])
            log_box.code(
                "\n".join(current_logs) if current_logs else "(waiting...)",
                language="text",
            )
            time.sleep(1)

        # Done — handle result
        current_logs = list(es['logs'])
        if es['error']:
            st.session_state.processing_logs = current_logs
            full_traceback = es['error']
            st.error("❌ Error processing PDF")
            if "Phase" in full_traceback:
                st.warning("⚠️ The extraction pipeline failed during one of the processing phases. See details below.")
            with st.expander("🔍 Show full error details", expanded=True):
                st.code(full_traceback, language="python")
            if st.session_state.extraction_result and st.session_state.extraction_result.get('metadata'):
                st.info("💡 Some data was extracted before the error. Check the 'View Extracted Data' page to see partial results.")
            st.markdown("### Possible solutions:")
            st.markdown("- Check that your GOOGLE_API_KEY is valid and has not exceeded quota")
            st.markdown("- Check the 'View Processing Logs' above for more details about what went wrong")
            st.markdown("- Try uploading a different PDF to see if the issue is PDF-specific")
            st.markdown("- If the error is in 'Phase 2 (Summary Generation)', the PDF might have unusual formatting")
        else:
            result = es['result']
            result['status'] = 'phases_0_3_complete'
            st.session_state.extraction_result = result
            st.session_state.phases_0_3_complete = True
            st.session_state.ready_for_phase_4 = True

            metadata = result.get('metadata', {})
            finance_data = result.get('finance', {})
            st.session_state.confirmed_metadata = metadata

            _extracted_loc = metadata.get('country_location', '')
            if _extracted_loc and not st.session_state.field_locks.get('location', False):
                st.session_state.location_countries = parse_location_string(_extracted_loc)
                st.session_state.field_edited['location'] = True
                for _wk in ['input_gdp_percap', 'input_cpia_score', 'input_governance_composite']:
                    st.session_state.pop(_wk, None)
            st.session_state.extracted_values = {
                'location': _extracted_loc,
                'start_date': metadata.get('planned_start_date', datetime.now().strftime('%Y-%m-%d')),
                'planned_end_date': metadata.get('planned_end_date'),
                'participating_orgs': metadata.get('participating_orgs', ''),
                'implementing_org_type': metadata.get('implementing_org_type', ''),
            }

            if metadata.get('planned_start_date') and metadata.get('planned_end_date'):
                try:
                    start = datetime.fromisoformat(metadata['planned_start_date'])
                    end = datetime.fromisoformat(metadata['planned_end_date'])
                    duration_years = (end - start).days / 365.25
                    st.session_state.extracted_values['planned_duration'] = duration_years
                    # LLM extracted real dates from the document → mark as "Set"
                    st.session_state.field_edited['planned_duration'] = True
                except Exception as _date_err:
                    logger.warning(f"Could not compute planned_duration from metadata dates: {_date_err}")

            if finance_data:
                total_alloc = finance_data.get('total_allocation', {})
                amount = total_alloc.get('amount')
                currency = total_alloc.get('currency', '').lower()
                if amount is not None:
                    if 'million' in currency:
                        amount_usd = float(amount) * 1_000_000
                    elif 'billion' in currency:
                        amount_usd = float(amount) * 1_000_000_000
                    else:
                        amount_usd = float(amount)
                    st.session_state.extracted_values['planned_expenditure'] = amount_usd
                    st.session_state.field_edited['planned_expenditure'] = True

            _ALL_SECTOR_CLUSTERS = get_sector_clusters()
            _sector_allocation = result.get('sector_allocation', {})
            if _sector_allocation:
                if 'sector_percentages' not in st.session_state:
                    st.session_state.sector_percentages = {}
                for _cl in _ALL_SECTOR_CLUSTERS:
                    if not st.session_state.field_locks.get(f'sector_{_cl}', False):
                        _pct = float(_sector_allocation.get(_cl, 0.0))
                        st.session_state.sector_percentages[_cl] = _pct
                        st.session_state[f'input_sector_{_cl}'] = _pct
                        st.session_state.field_edited[f'sector_{_cl}'] = True

            fields_to_mark = {'location': True, 'start_date': True}
            if 'planned_expenditure' in st.session_state.extracted_values:
                fields_to_mark['planned_expenditure'] = True
            if 'planned_duration' in st.session_state.extracted_values:
                fields_to_mark['planned_duration'] = True
            st.session_state.field_edited.update(fields_to_mark)

            # Clear LLM grade feature flags — only "Confirm and Extract Feature Grades"
            # should mark these as Set.  Prevents carry-over from a previously graded
            # project when a new (or cached) PDF is extracted.
            for _gf in ('targets', 'context', 'risks', 'finance', 'integratedness',
                        'implementer_performance', 'complexity'):
                st.session_state.field_edited.pop(_gf, None)

            if 'planned_expenditure' in st.session_state.extracted_values:
                st.session_state['input_planned_expenditure'] = (
                    st.session_state.extracted_values['planned_expenditure'] / 1_000_000
                )
            if 'planned_duration' in st.session_state.extracted_values:
                st.session_state['input_planned_duration'] = (
                    st.session_state.extracted_values['planned_duration']
                )
            if 'start_date' in st.session_state.extracted_values:
                try:
                    st.session_state['input_start_date'] = datetime.fromisoformat(
                        st.session_state.extracted_values['start_date']
                    ).date()
                except Exception:
                    pass
            if 'planned_end_date' in st.session_state.extracted_values:
                try:
                    st.session_state['input_planned_end_date'] = datetime.fromisoformat(
                        st.session_state.extracted_values['planned_end_date']
                    ).date()
                except Exception:
                    pass

            activity_id = result.get('activity_id')
            if activity_id:
                _prev_folder = st.session_state.selected_project_folder
                logger.info("[DEBUG PDF-EXTRACTION COMPLETE]")
                logger.info(f"extraction result activity_id  = {activity_id!r}")
                logger.info(f"selected_project_folder BEFORE = {_prev_folder!r}")

                if _prev_folder and _prev_folder != activity_id:
                    # An existing project is open — merge the new PDF extraction data
                    # into it instead of switching to the new hash-based folder.
                    import shutil as _shutil
                    _new_dir = os.path.join('extracted_pdf_data', activity_id)
                    _old_dir = os.path.join('extracted_pdf_data', _prev_folder)
                    logger.info(f"Merging {_new_dir!r} {_old_dir!r} (keeping existing project)")
                    if os.path.isdir(_new_dir) and os.path.isdir(_old_dir):
                        for _fname in os.listdir(_new_dir):
                            _src = os.path.join(_new_dir, _fname)
                            _dst = os.path.join(_old_dir, _fname)
                            _shutil.copy2(_src, _dst)
                        _shutil.rmtree(_new_dir)
                        logger.info(f"Merged & removed temp folder {_new_dir!r}")
                    # Redirect the extraction result to the existing project folder
                    result['activity_id'] = _prev_folder
                    result['output_dir'] = _old_dir
                    st.session_state.extraction_result = result
                    activity_id = _prev_folder
                    logger.info(f"selected_project_folder stays = {_prev_folder!r}")
                else:
                    # No existing project — use the new hash-based folder as normal
                    st.session_state.selected_project_folder = activity_id
                    logger.info(f"selected_project_folder AFTER  = {activity_id!r}")

                if st.session_state.pending_project_name:
                    st.session_state.project_name = st.session_state.pending_project_name
                    save_project_name(activity_id, st.session_state.pending_project_name)
                    st.session_state.pending_project_name = None
                else:
                    st.session_state.project_name = load_project_name(activity_id)
                _extracted_title = metadata.get('title', '')
                if _extracted_title and not st.session_state.field_locks.get('project_name', False):
                    st.session_state.project_name = _extracted_title
                    st.session_state.project_name_source = 'llm'
                    save_project_name(activity_id, _extracted_title)
                st.session_state.creating_new_project = False
                save_project_state(activity_id)
                load_project_state(activity_id)

            if result.get('cached', False):
                st.success(f"✅ Loaded cached results! Activity ID: {result['activity_id']}")
                st.info(f"♻️ This PDF ({result['num_pages']} pages) was already processed. Using cached results.")
            else:
                st.success(f"✅ Document processed! Activity ID: {result['activity_id']}")
                st.info(f"📄 Processed {result['num_pages']} pages. Scroll down to review auto-filled fields.")

            st.session_state.processing_logs = current_logs
        st.session_state.extraction_in_progress = False
        st.rerun()
