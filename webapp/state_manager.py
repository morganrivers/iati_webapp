import json
import math
import streamlit as st

from webapp_paths import TRAIN_MEDIANS_PATH


import logging

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize all session state variables."""
    # Purge stale button widget keys that cannot be set in session_state.
    # These can end up in session_state when loaded from old app_state.json files
    # that were saved before the lock_all_* exclusion was added to save_project_state.
    for _btn_key in [
        'lock_all_activity_features', 'unlock_all_activity_features',
        'lock_all_sectors', 'unlock_all_sectors',
    ]:
        if _btn_key in st.session_state:
            del st.session_state[_btn_key]

    # Initialize session ID for temporary auto-save (unique per browser session)
    if "session_id" not in st.session_state:
        import secrets
        st.session_state.session_id = secrets.token_hex(8)

    if "features" not in st.session_state:
        st.session_state.features = {}
    if "llm_authenticated" not in st.session_state:
        st.session_state.llm_authenticated = False
    if "field_edited" not in st.session_state:
        # Track which fields have been edited (by human or LLM)
        st.session_state.field_edited = {}
    if "shap_stale_fields" not in st.session_state:
        # SHAP feature keys whose badges should be hidden (value changed since last forecast)
        st.session_state.shap_stale_fields = set()
    if "phases_0_3_complete" not in st.session_state:
        st.session_state.phases_0_3_complete = False
    if "ready_for_phase_4" not in st.session_state:
        st.session_state.ready_for_phase_4 = False
    if "extract_phase_4_now" not in st.session_state:
        st.session_state.extract_phase_4_now = False
    if "grading_logs" not in st.session_state:
        st.session_state.grading_logs = []
    if "grading_complete" not in st.session_state:
        st.session_state.grading_complete = False
    if "grading_in_progress" not in st.session_state:
        st.session_state.grading_in_progress = False
    if "grading_state" not in st.session_state:
        st.session_state.grading_state = None
    if "rag_forecast_in_progress" not in st.session_state:
        st.session_state.rag_forecast_in_progress = False
    if "rag_forecast_state" not in st.session_state:
        st.session_state.rag_forecast_state = None
    if "rag_forecast_confirm_pending" not in st.session_state:
        st.session_state.rag_forecast_confirm_pending = False
    if "llm_call_count" not in st.session_state:
        st.session_state.llm_call_count = 0
    if "extraction_in_progress" not in st.session_state:
        st.session_state.extraction_in_progress = False
    if "extraction_state" not in st.session_state:
        st.session_state.extraction_state = None
    if "extracted_values" not in st.session_state:
        st.session_state.extracted_values = {}
    if "show_distribution_markers" not in st.session_state:
        st.session_state.show_distribution_markers = True  # Show markers on page load
    if "current_input_values" not in st.session_state:
        # Store current input values for distribution markers
        st.session_state.current_input_values = {}
    if "extraction_result" not in st.session_state:
        st.session_state.extraction_result = None
    if "confirmed_metadata" not in st.session_state:
        st.session_state.confirmed_metadata = {}
    if "field_locks" not in st.session_state:
        # Initialize all fields as unlocked
        st.session_state.field_locks = {
            "reporting_org": False,
            "planned_expenditure": False,
            "activity_scope": False,
            "finance_is_loan": False,
            "location": False,
            "start_date": False,
            "planned_end_date": False,
            "planned_duration": False,
            "finance": False,
            "integratedness": False,
            "implementer_performance": False,
            "targets": False,
            "context": False,
            "risks": False,
            "complexity": False,
            "gdp_percap": False,
            "cpia_score": False,
            "governance_composite": False,
            "wgi_any_missing": False,
            "project_name": False,
        }
    # Backfill any keys added after initial deployment (session state may predate them)
    for _new_lock_key in ['gdp_percap', 'cpia_score', 'governance_composite', 'wgi_any_missing', 'planned_end_date', 'project_name']:
        if _new_lock_key not in st.session_state.field_locks:
            st.session_state.field_locks[_new_lock_key] = False
    if "project_name_source" not in st.session_state:
        st.session_state.project_name_source = None  # None, "llm", or "human"
    if "feature_grades" not in st.session_state:
        # Store LLM-generated grades separately
        st.session_state.feature_grades = {}
    if "selected_project_folder" not in st.session_state:
        st.session_state.selected_project_folder = None
    if "project_name" not in st.session_state:
        st.session_state.project_name = None
    if "creating_new_project" not in st.session_state:
        st.session_state.creating_new_project = False
    if "location_countries" not in st.session_state:
        st.session_state.location_countries = []
    if "pending_project_name" not in st.session_state:
        st.session_state.pending_project_name = None

    # Initialize sector locks (backfill if missing - must run every time, not just first time!).
    # Derive cluster names from the model artifact so they never drift from feature_names.json.
    from model_loader import get_sector_clusters
    for cluster in get_sector_clusters():
        if f"sector_{cluster}" not in st.session_state.field_locks:
            st.session_state.field_locks[f"sector_{cluster}"] = False


def validate_and_sync_field_edited(train_medians: dict):
    """
    Recalculate field_edited flags from current widget key values vs training medians.
    Called after loading state to fix stale field_edited values from old saves.
    Uses widget keys as the source of truth (not st.session_state.features, which
    only holds LLM-extracted values and is unreliable for manually-edited fields).
    Only updates flags for fields where an input_ widget key is present in session state.
    """
    logger.info("RESYNCING field_edited flags from widget keys vs medians...")

    # Grade features are exclusively controlled by the grading completion handler.
    # Never auto-detect them as "Set" by comparing widget values to medians.
    _GRADE_FEATURES = {'targets', 'context', 'risks', 'finance', 'integratedness',
                       'implementer_performance', 'complexity'}

    fixed_count = 0

    for feature_name, median_value in train_medians.items():
        if feature_name in _GRADE_FEATURES:
            continue  # only the grading handler sets/clears these
        widget_key = f"input_{feature_name}"
        if widget_key not in st.session_state:
            continue  # widget not yet rendered — leave field_edited unchanged

        try:
            current_value = float(st.session_state[widget_key])
            if feature_name == 'gdp_percap':
                # Widget holds raw USD; the median is stored on the log scale
                # (matching on_gdp_percap_change and _train_median_gdp=np.exp(median)).
                # Compare in log space so this agrees with the widget callback.
                if current_value <= 0:
                    continue
                is_different = abs(math.log(current_value) - float(median_value)) > 0.01
            else:
                is_different = abs(current_value - float(median_value)) > 0.01
        except (TypeError, ValueError):
            continue

        currently_marked = st.session_state.field_edited.get(feature_name, False)
        if currently_marked != is_different:
            st.session_state.field_edited[feature_name] = is_different
            logger.info(f"Fixed {feature_name}: widget={current_value:.3f}, median={float(median_value):.3f} edited={is_different}")
            fixed_count += 1

    logger.info(f"Done: fixed {fixed_count} field_edited flags")

    return fixed_count


def clear_project_state():
    """
    Clear all project-specific state when switching projects.
    This prevents state from one project leaking into another.
    """
    logger.info("CLEARING PROJECT STATE...")

    # Clear features and tracking
    st.session_state.features = {}
    st.session_state.field_edited = {}
    st.session_state.extracted_values = {}
    logger.info("!!! feature grades cleared!")
    st.session_state.feature_grades = {}
    st.session_state.project_name_source = None

    # Clear location and sectors
    st.session_state.location_countries = []
    if "sector_percentages" in st.session_state:
        st.session_state.sector_percentages = {}

    # Bump the location widget nonce so the location row selectboxes are rebuilt
    # as fresh widgets after a project switch (prevents the Streamlit frontend
    # from displaying the previous project's country even when the server value
    # is correct). The old (prev-nonce) keys are removed by the prefix clear below.
    st.session_state.loc_widget_nonce = st.session_state.get('loc_widget_nonce', 0) + 1

    # Clear extraction results
    st.session_state.extraction_result = None
    st.session_state.phases_0_3_complete = False
    st.session_state.ready_for_phase_4 = False
    st.session_state.grading_logs = []
    st.session_state.grading_complete = False
    st.session_state.grading_in_progress = False
    st.session_state.grading_state = None
    st.session_state.extraction_in_progress = False
    st.session_state.extraction_state = None
    st.session_state.rag_forecast_in_progress = False
    st.session_state.rag_forecast_state = None

    # Clear forecast results (user must re-run prediction for each project)
    for _key in ['prediction', 'base', 'ens_delta',
                 'feature_vector', 'feature_vector_imputed',
                 'shap_result', 'prediction_explanation']:
        st.session_state.pop(_key, None)

    # Clear widget keys (input_*, select_*) but NOT button keys or other non-modifiable widgets
    # Buttons, file_uploaders, and some other widgets don't allow state modification
    keys_to_clear = []
    for key in st.session_state.keys():
        # Skip internal keys
        if key.startswith('_'):
            continue
        # Only clear input and select widget keys, not buttons or other widget types
        # Buttons typically have keys like 'lock_all_*', 'btn_*', etc. but their state can't be modified
        # Also clear the per-row and add-row location widget keys. These are the sole
        # source Streamlit reads for the location selectboxes; if left stale on a project
        # switch they override the freshly loaded location_countries (index=0 is ignored
        # once a widget key already holds a value), corrupting the loaded location.
        if key.startswith(('input_', 'select_', 'loc_country_', 'loc_pct_')) or key in ('loc_new_country', 'loc_new_pct'):
            keys_to_clear.append(key)

    _GRADE_KEYS = {'input_finance', 'input_integratedness', 'input_implementer_performance',
                   'input_targets', 'input_context', 'input_risks', 'input_complexity'}

    with open(TRAIN_MEDIANS_PATH, 'r') as f:
        _train_medians = json.load(f)

    cleared_count = 0
    for key in keys_to_clear:
        try:
            if key in _GRADE_KEYS:
                feature_name = key[len("input_"):]
                st.session_state[key] = float(_train_medians[feature_name])
            else:
                del st.session_state[key]
            cleared_count += 1
        except Exception as e:
            logger.warning(f"Could not clear {key}: {e}")

    logger.info(f"Cleared {cleared_count} widget keys and all project state")
