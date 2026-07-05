import logging
import time
import threading

import streamlit as st

from project_manager import load_project_state, save_project_state
from ui_components import render_llm_feature_slider

from .common import EXTRACTED_PDF_DIR, _run_phase4_background

logger = logging.getLogger(__name__)


def render_confirm_and_poll_phase4(_llm_running: bool, model_metadata: dict, training_data, _shap):
    # Params: _llm_running (bool)
    # Returns: None  (sets grading_in_progress, grading_complete, feature_grades, etc.)
    # ====================================================================
    # CONFIRMATION BUTTON FOR PHASE 4
    # ====================================================================
    if st.session_state.ready_for_phase_4:
        st.markdown("---")
        st.markdown("### Ready to Extract Feature Grades?")

        # Check if grades already exist
        if st.session_state.extraction_result:
            activity_id = st.session_state.extraction_result.get('activity_id')
            grades_file = EXTRACTED_PDF_DIR / activity_id / "feature_grades.jsonl"
            if grades_file.exists():
                st.success("✅ Feature grades already exist for this activity (loaded from disk)")
                st.info("💡 Grades were previously generated and loaded. You can proceed to prediction below, or re-run grading with the button.")
            else:
                st.info("📋 Review the auto-filled fields above. You can edit any values or lock fields to prevent changes.")
        else:
            st.info("📋 Review the auto-filled fields above. You can edit any values or lock fields to prevent changes.")

    _has_features = bool(st.session_state.get('extraction_result') and
                         st.session_state.extraction_result.get('features'))
    if not _has_features:
        st.warning("⚠️ No extracted features available — upload and extract a PDF first.")
    if not st.session_state.get('llm_authenticated', False):
        st.warning("🔒 LLM access required — authenticate above to extract feature grades.")
    if st.button("✅ Confirm and Extract Feature Grades", type="primary", width='stretch',
                 disabled=not _has_features or not st.session_state.get('llm_authenticated', False)):
        st.session_state.ready_for_phase_4 = False
        st.session_state.extract_phase_4_now = True
        st.rerun()


    # ============================================================================
    # GRADING POLL (must be before input section so editing is blocked while running)
    # ============================================================================
    if st.session_state.grading_in_progress:
        # print("!!! grading in progress!!")
        gs = st.session_state.grading_state
        log_box = st.empty()

        while not gs['done']:
            current_logs = list(gs['logs']) if gs else []
            log_box.code(
                "\n".join(current_logs) if current_logs else "(waiting for progress updates...)",
                language="text",
            )
            time.sleep(1)

        # Done — handle result
        current_logs = list(gs['logs']) if gs else []
        if gs['error']:
            st.error(f"❌ Error extracting features: {gs['error']}")
        else:
            feature_grades = gs['grades']
            # print("!!! checking if features are locked... ")
            st.session_state.feature_grades = feature_grades.copy()
            for feature, grade in feature_grades.items():
                if not st.session_state.field_locks.get(feature, False):
                    prev = st.session_state.get(f"input_{feature}", '<unset>')
                    st.session_state.features[feature] = grade
                    st.session_state.field_edited[feature] = True
                    st.session_state[f"input_{feature}"] = grade
                else:
                    pass
            result = st.session_state.extraction_result
            result['features'] = gs.get('features')
            result['status'] = 'complete'
            st.session_state.extraction_result = result

            logger.info("[DEBUG GRADING-COMPLETE A] just before save")
            logger.info(f"selected_project_folder      = {st.session_state.selected_project_folder!r}")
            logger.info(f"extraction_result.activity_id= {(result or {}).get('activity_id','<None>')!r}")
            logger.info(f"extraction_result.output_dir = {(result or {}).get('output_dir','<None>')!r}")
            import glob as _glob
            _existing = [p.split('/')[-2] for p in _glob.glob('extracted_pdf_data/*/app_state.json')]
            logger.info(f"projects-on-disk BEFORE save = {_existing}")

            if st.session_state.selected_project_folder:
                save_project_state(st.session_state.selected_project_folder)

                logger.info("[DEBUG GRADING-COMPLETE B] just before load_project_state")
                logger.info(f"selected_project_folder      = {st.session_state.selected_project_folder!r}")
                logger.info(f"extraction_result.activity_id= {st.session_state.extraction_result.get('activity_id','<None>')!r}")

                load_project_state(st.session_state.selected_project_folder)

                logger.info("[DEBUG GRADING-COMPLETE C] after load_project_state")
                logger.info(f"selected_project_folder      = {st.session_state.selected_project_folder!r}")
                _er_after = st.session_state.extraction_result or {}
                logger.info(f"extraction_result.activity_id= {_er_after.get('activity_id','<None>')!r}")
                _existing2 = [p.split('/')[-2] for p in _glob.glob('extracted_pdf_data/*/app_state.json')]
                logger.info(f"projects-on-disk AFTER  save = {_existing2}")
                _new_projects = set(_existing2) - set(_existing)
                if _new_projects:
                    logger.warning(f"NEW project folder(s) created: {_new_projects}")

            st.session_state.grading_complete = True

        st.session_state.grading_in_progress = False
        st.session_state.grading_logs = current_logs
        st.rerun()


    if not st.session_state.grading_in_progress:

        # ====================================================================
        # PHASE 4 EXTRACTION (runs below the button)
        # ====================================================================
        # ---- Launch background thread when button was pressed ----
        if st.session_state.extract_phase_4_now:
            st.session_state.extract_phase_4_now = False
            st.session_state.grading_complete = False
            grading_state = {
                'logs': [],
                'done': False,
                'grades': None,
                'features': None,
                'error': None,
            }
            st.session_state.grading_state = grading_state
            st.session_state.grading_in_progress = True
            snapshot = {
                'extraction_result': st.session_state.extraction_result,
                'confirmed_metadata': dict(st.session_state.confirmed_metadata),
                'title': st.session_state.confirmed_metadata.get('title', 'Unknown Activity'),
            }
            t = threading.Thread(
                target=_run_phase4_background,
                args=(snapshot, grading_state),
                daemon=True,
            )
            t.start()
            st.rerun()  # Force rerun to avoid double display


        # ---- Persist completed logs across reruns ----
        if st.session_state.grading_complete and st.session_state.grading_logs:
            st.success("✅ LLM feature grades extracted! Scroll down to see the grades.")
            with st.expander("📋 View Processing Logs", expanded=False):
                st.code("\n".join(st.session_state.grading_logs), language="text")

        # Only show LLM feature grades if Phase 4 is complete or in progress
        # ---- REFACTOR → render_llm_feature_grades_subsection(model_metadata,
        #         training_data, _shap, _shap_sum, _llm_running)
        #         -> (finance, integratedness, implementer_performance,
        #             targets, context, risks, complexity) ----
        if not st.session_state.ready_for_phase_4:
            # Manual feature entry
            st.subheader("LLM Feature Grades (0-100 scale)")
            st.markdown("Enter grades for each dimension. **0 = Very Poor, 100 = Excellent**. Click lock icon to prevent LLM from overwriting.")

        # Get default values from training medians
        default_finance = model_metadata["train_medians"]["finance"]
        default_integratedness = model_metadata["train_medians"]["integratedness"]
        default_implementer_performance = model_metadata["train_medians"]["implementer_performance"]
        default_targets = model_metadata["train_medians"]["targets"]
        default_context = model_metadata["train_medians"]["context"]
        default_risks = model_metadata["train_medians"]["risks"]
        default_complexity = model_metadata["train_medians"]["complexity"]

        # Finance
        finance = render_llm_feature_slider(
            "finance",
            "Finance Quality",
            default_finance,
            "How well-financed is the activity? (0=Very Poor, 100=Excellent)",
            "finance",
            training_data=training_data,
            shap_contribution=_shap('finance')
        )

        # Integratedness
        integratedness = render_llm_feature_slider(
            "integratedness",
            "Integratedness",
            default_integratedness,
            "Integration with other programs/systems (0=Very Poor, 100=Excellent)",
            "integratedness",
            training_data=training_data,
            shap_contribution=_shap('integratedness')
        )

        # Implementer Performance
        implementer_performance = render_llm_feature_slider(
            "implementer_performance",
            "Implementer Performance",
            default_implementer_performance,
            "Quality of implementing organization (0=Very Poor, 100=Excellent)",
            "implementer_performance",
            training_data=training_data,
            shap_contribution=_shap('implementer_performance')
        )

        # Targets
        targets = render_llm_feature_slider(
            "targets",
            "Target Quality",
            default_targets,
            "Clarity and achievability of targets (0=Very Poor, 100=Excellent)",
            "targets",
            training_data=training_data,
            shap_contribution=_shap('targets')
        )

        # Context
        context = render_llm_feature_slider(
            "context",
            "Context",
            default_context,
            "External context favorability (0=Very Poor, 100=Excellent)",
            "context",
            training_data=training_data,
            shap_contribution=_shap('context')
        )

        # Risks
        risks = render_llm_feature_slider(
            "risks",
            "Risks (inverted)",
            default_risks,
            "Higher = lower risk (0=Very High Risk, 100=Very Low Risk)",
            "risks",
            training_data=training_data,
            shap_contribution=_shap('risks')
        )

        # Complexity
        complexity = render_llm_feature_slider(
            "complexity",
            "Complexity",
            default_complexity,
            "Project complexity (0=Very Simple, 100=Very Complex)",
            "complexity",
            training_data=training_data,
            shap_contribution=_shap('complexity')
        )
    return finance, integratedness, implementer_performance, targets, context, risks, complexity 
