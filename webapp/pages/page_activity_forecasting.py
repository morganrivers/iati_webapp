import logging

import streamlit as st
import pandas as pd


from ui_components import render_histogram
from rf_predictor import impute_and_run_statistical_model
from shap_explainer import compute_shap_values
from explain_single_prediction import generate_explanation_for_webapp
from tag_model_loader import load_tag_models
from tag_predictor import predict_outcome_tags

from pages.activity_forecasting.common import SCOPE_LABELS
from pages.activity_forecasting.project_selector import render_project_selector
from pages.activity_forecasting.upload import render_llm_upload_section, poll_extraction_phases_0_3
from pages.activity_forecasting.inputs import (render_location_input,
                                               render_basic_info_subsection,
                                               render_activity_features_subsection)
from pages.activity_forecasting.sectors import render_sector_allocation_subsection
from pages.activity_forecasting.grading import render_confirm_and_poll_phase4
from pages.activity_forecasting.embeddings import render_targets_embeddings_subsection
from pages.activity_forecasting.analysis import render_analysis_section
from pages.activity_forecasting.tags import render_tag_predictions

logger = logging.getLogger(__name__)


def render_activity_forecasting_page(rf_model, extra_model, per_org_baseline, start_year_correction, model_metadata, training_data):
    # Title and description
    st.title("🌍 IATI Activity Success Forecasting")
    st.markdown("""
    This tool predicts the likely success of international aid and cooperation activities using a Random Forest and ExtraTrees ensemble
    trained on IATI (International Aid Transparency Initiative) evaluation data. Upload a project document or enter
    activity details to produce an evaluation from the model. Further details on model performance available on the "Model Performance" page. 

    💡 **Tip:** After extracting features from a PDF, view detailed extraction results in the "View Extracted Data" page.
    """)

    # Convenience flag — true whenever any background LLM call is running
    _llm_running = (
        st.session_state.extraction_in_progress or
        st.session_state.grading_in_progress or
        st.session_state.get('rag_forecast_in_progress', False)
    )

    # ---- REFACTOR → render_project_selector(_llm_running) ----
    # ============================================================================
    # PROJECT SELECTOR
    # ============================================================================

    render_project_selector(_llm_running)

    # ---- REFACTOR → render_llm_upload_section(_llm_running) ----
    # ============================================================================
    # LLM UPLOAD & EXTRACTION (Moved to top)
    # ============================================================================
    
    render_llm_upload_section(_llm_running, model_metadata=model_metadata)

    # ---- REFACTOR → poll_extraction_phases_0_3() ----
    # =========================================================================
    # PHASES 0-3 POLLING (runs every rerun; survives page navigation)
    # =========================================================================
    poll_extraction_phases_0_3()
        # Show persistent processing logs after rerun
    if st.session_state.get('processing_logs'):
        with st.expander("📋 Processing Logs", expanded=False):
            st.code("\n".join(st.session_state.processing_logs), language="text")


        # # current_logs = list(gs['logs']) if gs else []
        # # if gs and gs['done']:
        #     # print("!!! grading state is set to done!")
        #     if gs['error']:
        #         st.error(f"❌ Error extracting features: {gs['error']}")
        #     else:
        #         feature_grades = gs['grades']
        #         # print("!!! checking if features are locked... ")
        #         st.session_state.feature_grades = feature_grades.copy()
        #         for feature, grade in feature_grades.items():
        #             if not st.session_state.field_locks.get(feature, False):
        #                 prev = st.session_state.get(f"input_{feature}", '<unset>')
        #                 st.session_state.features[feature] = grade
        #                 st.session_state.field_edited[feature] = True
        #                 st.session_state[f"input_{feature}"] = grade
        #             else:
        #                 pass
        #         result = st.session_state.extraction_result
        #         result['features'] = gs.get('features')
        #         result['status'] = 'complete'
        #         st.session_state.extraction_result = result
        #         if st.session_state.selected_project_folder:
        #             save_project_state(st.session_state.selected_project_folder)
        #             # FIX: Load the state back immediately so widget keys are restored before rerun
        #             load_project_state(st.session_state.selected_project_folder)
        #         st.session_state.grading_complete = True
        #     st.session_state.grading_in_progress = False
        #     st.session_state.grading_logs = current_logs
        #     st.rerun()
        # # else:
        # #     st.warning("⚠️ LLM feature grading in progress — field editing is disabled until grading completes.")
        # #     with st.expander("📋 View Processing Logs", expanded=True):
        # #         st.code("\n".join(current_logs) if current_logs else "(waiting...)", language="text")
        # #     time.sleep(1)
        #     st.rerun()


    # ============================================================================
    # INPUT SECTION
    # ============================================================================


    project_selected = st.session_state.selected_project_folder is not None
    if project_selected or st.session_state.creating_new_project:

        # Build SHAP lookup once for the whole input section (empty before first forecast)
        _shap_by_feature = {}
        if st.session_state.get('shap_result'):
            _sr = st.session_state.shap_result
            _shap_by_feature = dict(zip(_sr['feature_names'], _sr['shap_values']))

        def _shap(key):
            """Return SHAP value for key, or None if stale (field changed since last forecast)."""
            if key in st.session_state.shap_stale_fields:
                return None
            return _shap_by_feature.get(key)

        def _shap_sum(*keys):
            """Sum SHAP values for multiple keys, or None if any key is stale."""
            if any(k in st.session_state.shap_stale_fields for k in keys):
                return None
            if not _shap_by_feature:
                return None
            return sum(_shap_by_feature.get(k, 0) for k in keys)

        location = render_location_input(_llm_running, _shap_sum)
        
        st.markdown("---")

        with st.expander("Edit or View Activity Information", expanded=False):


            # Lock/Unlock All buttons
            def _lock_all_cb():
                for key in st.session_state.field_locks:
                    st.session_state.field_locks[key] = True
                    widget_key = "lock_location_outer" if key == "location" else f"lock_{key}"
                    if widget_key in st.session_state:
                        st.session_state[widget_key] = True

            def _unlock_all_cb():
                for key in st.session_state.field_locks:
                    st.session_state.field_locks[key] = False
                    widget_key = "lock_location_outer" if key == "location" else f"lock_{key}"
                    if widget_key in st.session_state:
                        st.session_state[widget_key] = False

            col_lock1, col_lock2, col_lock3 = st.columns([1, 1, 3])
            with col_lock1:
                st.button("🔒 Lock All", on_click=_lock_all_cb)
            with col_lock2:
                st.button("🔓 Unlock All", on_click=_unlock_all_cb)
            with col_lock3:
                st.info("💡 Locked fields won't be overwritten by LLM extraction")

            reporting_org, start_date, location_features = render_basic_info_subsection(model_metadata, training_data,  location, _llm_running, _shap, _shap_sum)

            planned_expenditure, planned_duration, activity_scope, finance_is_loan, gdp_percap_input, cpia_score_input, governance_composite_input = render_activity_features_subsection(model_metadata, training_data, _llm_running, _shap, _shap_sum, location_features)
            
            sector_percentages = render_sector_allocation_subsection(model_metadata, training_data, _shap, _shap_sum)
            # ---- REFACTOR → render_sector_allocation_subsection(model_metadata, training_data)
            #         -> sector_percentages ----

            finance, integratedness, implementer_performance, targets, context, risks, complexity = render_confirm_and_poll_phase4(_llm_running, model_metadata, training_data, _shap)
            # ---- REFACTOR → render_confirm_and_poll_phase4(_llm_running) ----
            #   (wraps the confirm button, the grading poll, and phase-4 thread launch)

            if not st.session_state.grading_in_progress:

                umap3_x, umap3_y, umap3_z, sector_distance, country_distance = render_targets_embeddings_subsection(model_metadata, training_data, _shap, start_date)

                # ---- REFACTOR → render_missingness_indicators(location_features,
                #         sector_percentages, training_data) -> None ----
                # ============================================================================
                # MISSINGNESS INDICATORS
                # ============================================================================

                with st.expander("🔍 Missing Indicators Distribution", expanded=False):
                    st.markdown("See how your data completeness compares to the database.")

                    # Compute missingness indicators from current inputs
                    current_cpia_missing = 1.0 if location_features.get('cpia_score') is None else 0.0
                    current_gdp_missing = 1.0 if location_features.get('gdp_percap') is None else 0.0
                    current_planned_exp_missing = 0.0  # Always provided
                    current_planned_dur_missing = 0.0  # Always provided
                    current_sector_missing = 1.0 if sum(sector_percentages.values()) == 0 else 0.0
                    current_umap_missing = 0.0  # Always use median
                    current_wgi_missing = 1.0 if location_features.get('wgi_any_missing') else 0.0
                    current_gov_missing_count = (5 if location_features.get('wgi_any_missing') else 0) if location_features.get('wgi_any_missing') is not None else 5

                    # Show histograms for missing indicators
                    missing_features = [
                        ("cpia_missing", current_cpia_missing),
                        ("gdp_percap_missing", current_gdp_missing),
                        ("planned_expenditure_missing", current_planned_exp_missing),
                        ("planned_duration_missing", current_planned_dur_missing),
                        ("sector_clusters_missing", current_sector_missing),
                        ("umap_missing", current_umap_missing),
                        ("wgi_any_missing", current_wgi_missing),
                        ("governance_missing_count", current_gov_missing_count),
                    ]

                    cols = st.columns(3)
                    for idx, (feat_name, current_val) in enumerate(missing_features):
                        with cols[idx % 3]:
                            if feat_name in training_data.columns:
                                train_data_col = training_data[feat_name].dropna()
                                fig = render_histogram(
                                    feat_name,
                                    train_data_col,
                                    current_val,
                                    show_marker=st.session_state.show_distribution_markers,
                                    height=200
                                )
                                if fig:
                                    st.plotly_chart(fig, width='stretch', key=f"hist_{feat_name}")

                # ============================================================================
                # PREDICTION SECTION
                # ============================================================================

        st.header("Forecast")

        # Show warning for fields using median values
        unedited_fields = []
        median_values_used = {}

        # Check which fields are unedited
        field_checks = {
            'reporting_org': (reporting_org, 'Reporting Organization'),
            'location': (location, 'Activity Location'),
            'start_date': (start_date, 'Planned Start Date'),
            'planned_expenditure': (planned_expenditure, f'Planned Expenditure: ${planned_expenditure:,.0f}'),
            'planned_duration': (planned_duration, f'Planned Duration: {planned_duration:.1f} years'),
            'activity_scope': (activity_scope, f'Activity Scope: {SCOPE_LABELS.get(activity_scope, activity_scope)}'),
            'finance_is_loan': (finance_is_loan, 'Finance Type: ' + ('Loan' if finance_is_loan else 'Grant')),
            'finance': (finance, f'Finance Quality: {finance:.1f}'),
            'integratedness': (integratedness, f'Integratedness: {integratedness:.1f}'),
            'implementer_performance': (implementer_performance, f'Implementer Performance: {implementer_performance:.1f}'),
            'targets': (targets, f'Target Quality: {targets:.1f}'),
            'context': (context, f'Context: {context:.1f}'),
            'risks': (risks, f'Risks: {risks:.1f}'),
            'complexity': (complexity, f'Complexity: {complexity:.1f}'),
        }

        for field_name, (current_value, display_text) in field_checks.items():
            if not st.session_state.field_edited.get(field_name, False):
                unedited_fields.append(display_text)
                median_values_used[field_name] = current_value

        if unedited_fields:
            st.warning("⚠️ **Using median values for unedited fields:**")
            st.markdown("The following fields were not edited and will use database median values:")
            for field_display in unedited_fields:
                st.markdown(f"- {field_display}")
            st.markdown("💡 Edit these fields above for more accurate predictions, or proceed with median values.")
        else:
            st.success("✅ All fields have been set! Ready for prediction.")

        st.caption("After running the forecast, colored arrow badges (↑ green / ↓ red) will appear next to each input field showing how much that input shifts the predicted rating. Hover over a badge to see the exact contribution.")

        if st.button("Predict Success Rating (Statistical Model)", type="primary"):
                
            with st.spinner("Generating forecast..."):
                feature_vector, feature_vector_imputed, base, ens_delta, prediction = impute_and_run_statistical_model(rf_model,extra_model,per_org_baseline,start_year_correction,model_metadata,reporting_org,gdp_percap_input,st.session_state.field_edited,st.session_state,location_features)
                # Compute SHAP values for this prediction
                shap_result = compute_shap_values(
                    rf_model=rf_model,
                    extra_model=extra_model,
                    feature_vector_imputed=feature_vector_imputed,
                    feature_names=model_metadata['feature_names']
                )

                # Compute feature sources at prediction time and cache as serialisable table
                _embedding_computed = bool(st.session_state.get('embedding_results'))
                _sources = []
                for _feat in feature_vector_imputed.columns:
                    _orig = feature_vector.iloc[0][_feat]
                    if pd.isna(_orig):
                        _sources.append('⛁ Train median')
                    elif _feat in ['finance', 'integratedness', 'implementer_performance', 'targets',
                                   'context', 'risks', 'complexity']:
                        _sources.append('🟢 User input (LLM grade)')
                    elif _feat in ['activity_scope', 'finance_is_loan', 'planned_duration',
                                   'planned_expenditure', 'log_planned_expenditure']:
                        _sources.append('🟢 User input')
                    elif _feat in ['gdp_percap', 'cpia_score', 'governance_composite', 'wgi_any_missing'] + \
                            [f'region_{r}' for r in ['AFE', 'AFW', 'EAP', 'ECA', 'LAC', 'MENA', 'SAS']]:
                        _sources.append('🟡 Extracted (location)')
                    elif _feat.startswith('rep_org_'):
                        _sources.append('🟢 User input (org)')
                    elif _feat.startswith('sector_cluster_'):
                        _sources.append('🟢 User input (sector %)')
                    elif _feat in ['country_distance', 'sector_distance', 'umap3_x', 'umap3_y', 'umap3_z']:
                        _sources.append('🟢 Targets embedding' if _embedding_computed else '⛁ Train median')
                    else:
                        _sources.append('🟡 Computed')
                st.session_state.feature_table = [
                    {'Feature': feat, 'Value': float(val), 'Source': src}
                    for feat, val, src in zip(
                        feature_vector_imputed.columns,
                        feature_vector_imputed.iloc[0].values,
                        _sources
                    )
                ]

                # Generate explanation while spinner is still active
                _explanation = None
                try:
                    _explanation = generate_explanation_for_webapp(
                        rf_model=rf_model,
                        feature_vector_imputed=feature_vector_imputed,
                        feature_names=model_metadata['feature_names'],
                        final_pred=prediction
                    )
                except Exception as e:
                    logger.exception("ERROR generating explanation:")

                # Store in session state
                st.session_state.prediction = prediction
                st.session_state.base = base
                st.session_state.ens_delta = ens_delta
                st.session_state.feature_vector = feature_vector
                st.session_state.feature_vector_imputed = feature_vector_imputed
                st.session_state.shap_result = shap_result
                st.session_state.shap_stale_fields = set()
                st.session_state.prediction_explanation = _explanation

                # Compute outcome tag probabilities
                try:
                    _tag_models = load_tag_models()
                    st.session_state.tag_predictions = predict_outcome_tags(
                        _tag_models, feature_vector_imputed
                    )
                except Exception as _e:
                    logger.error(f"[tag_predictor] ERROR: {_e}")
                    st.session_state.tag_predictions = {}

                st.rerun()


        render_analysis_section()
        render_tag_predictions()
