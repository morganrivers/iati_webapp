import logging
from datetime import datetime, date

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils import COUNTRY_NAMES, _COUNTRY_OPTIONS, parse_location_string, build_location_string
from ui_components import get_field_indicator, render_histogram, render_shap_annotation
from location_features import extract_features_from_location, KEEP_REPORTING_ORGS, get_org_dummies
from project_manager import save_project_state_temp

from .common import (SCOPE_LABELS, _ensure_project_folder, _save_and_rerun,
                     _compute_org_counts, _compute_region_totals)

logger = logging.getLogger(__name__)


def render_location_input(_llm_running: bool, _shap_sum) -> str:
    st.markdown("**Activity Location** *(required — drives GDP, CPIA, governance lookups)*")

    _ev_loc = st.session_state.extracted_values.get('location', '')
    if _ev_loc and not st.session_state.location_countries:
        st.session_state.location_countries = parse_location_string(_ev_loc)

    _loc_nonce = st.session_state.get('loc_widget_nonce', 0)
    _countries_changed = False
    for _idx, _entry in enumerate(list(st.session_state.location_countries)):
        _code = _entry["code"]
        _name = COUNTRY_NAMES.get(_code, _code)
        _display = f"{_name} ({_code})"
        _col_country, _col_pct, _col_remove = st.columns([3, 1, 0.5])
        with _col_country:
            _opts = [_display] + [o for o in _COUNTRY_OPTIONS if o != _display]
            _new_display = st.selectbox(
                "Country", _opts, index=0,
                key=f"loc_country_{_loc_nonce}_{_idx}",
                label_visibility="collapsed",
                disabled=st.session_state.field_locks.get('location', False)
            )
            if _new_display != _display:
                _new_code = _new_display.rsplit("(", 1)[-1].rstrip(")")
                st.session_state.location_countries[_idx]["code"] = _new_code
                _countries_changed = True
        with _col_pct:
            _new_pct = st.number_input(
                "%", min_value=1.0, max_value=100.0,
                value=float(_entry["pct"]),
                key=f"loc_pct_{_loc_nonce}_{_idx}",
                label_visibility="collapsed",
                disabled=st.session_state.field_locks.get('location', False)
            )
            if _new_pct != _entry["pct"]:
                st.session_state.location_countries[_idx]["pct"] = _new_pct
                _countries_changed = True
        with _col_remove:
            if st.button("✕", key=f"loc_remove_{_loc_nonce}_{_idx}",
                         disabled=st.session_state.field_locks.get('location', False)):
                logger.info("DEBUG REMOVE COUNTRY BUTTON CLICKED")
                logger.info(f"Removing index {_idx}")
                logger.info(f"Before pop: {st.session_state.location_countries}")
                st.session_state.location_countries.pop(_idx)
                logger.info(f"After pop: {st.session_state.location_countries}")
                _countries_changed = True

                _ensure_project_folder()

                if st.session_state.get('selected_project_folder'):
                    logger.info(f">>> Saving temp state for: {st.session_state.selected_project_folder}")
                    save_project_state_temp(st.session_state.selected_project_folder)
                else:
                    logger.warning(">>> WARNING: No project folder - can't save temp state!")
                logger.info(">>> Calling st.rerun()")
                st.rerun()
    if _countries_changed:
        _save_and_rerun()

    if not st.session_state.field_locks.get('location', False):
        with st.container():
            st.markdown("""
                <style>
                [data-testid="stVerticalBlock"] > div:has(> div > div > [data-baseweb="select"] [aria-label*="Add country"]) {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px dashed #dee2e6;
                }
                </style>
            """, unsafe_allow_html=True)

            st.caption("➕ Add a country:")
            _col_new, _col_newpct, _col_add = st.columns([3, 1, 0.5])
            with _col_new:
                _new_country_sel = st.selectbox(
                    "Add country", ["— select —"] + _COUNTRY_OPTIONS,
                    key="loc_new_country", label_visibility="collapsed",
                    help="Select a country to add to the activity location"
                )
            with _col_newpct:
                _remaining = max(1.0, 100.0 - sum(c["pct"] for c in st.session_state.location_countries))
                _new_pct_val = st.number_input(
                    "%", min_value=1.0, max_value=100.0, value=float(_remaining),
                    key="loc_new_pct", label_visibility="collapsed",
                    help="Percentage of activity in this country"
                )
            with _col_add:
                if st.button("＋ Add", key="loc_add_btn", type="secondary"):
                    logger.info("DEBUG ADD COUNTRY BUTTON CLICKED")
                    logger.info(f"_new_country_sel: {_new_country_sel}")
                    logger.info(f"_new_pct_val: {_new_pct_val}")
                    logger.info(f"Current location_countries: {st.session_state.location_countries}")

                    if _new_country_sel != "— select —":
                        logger.info(">>> Country is selected (not '— select —')")
                        _add_code = _new_country_sel.rsplit("(", 1)[-1].rstrip(")")
                        logger.info(f">>> Extracted code: {_add_code}")
                        _existing_codes = [c["code"] for c in st.session_state.location_countries]
                        logger.info(f">>> Existing codes: {_existing_codes}")

                        if _add_code not in _existing_codes:
                            logger.info(">>> Code not in existing list - adding!")
                            st.session_state.location_countries.append({"code": _add_code, "pct": _new_pct_val})
                            logger.info(f">>> Updated location_countries: {st.session_state.location_countries}")
                            _countries_changed = True
                            for _wk in ['input_gdp_percap', 'input_cpia_score', 'input_governance_composite']:
                                st.session_state.pop(_wk, None)
                            _save_and_rerun()
                        else:
                            logger.info(">>> Code already in list - not adding (duplicate)")
                    else:
                        logger.info(">>> No country selected - button clicked but '— select —' is still selected")

    _total_pct = sum(c["pct"] for c in st.session_state.location_countries)
    if st.session_state.location_countries:
        if _total_pct != 100:
            st.warning(f"⚠️ Percentages sum to {_total_pct}% — must equal 100%")
        else:
            _summary = ", ".join(COUNTRY_NAMES.get(c["code"], c["code"]) + f" {c['pct']}%" for c in st.session_state.location_countries)
            st.caption(f"✓ {_summary}")
    else:
        st.caption("No country set — add at least one country above")
    _region_shap = _shap_sum('region_AFE', 'region_AFW', 'region_EAP', 'region_ECA',
                            'region_LAC', 'region_MENA', 'region_SAS')
    render_shap_annotation(_region_shap, label="Location region (note: contribution is for the region, not the individual country)")

    if _countries_changed:
        logger.info("[LOC-CHANGED] location_countries changed popping widget keys")
        logger.info(f"[LOC-CHANGED] new location_countries = {st.session_state.location_countries!r}")
        st.session_state.field_edited['location'] = True
        for _wk in ['input_gdp_percap', 'input_cpia_score', 'input_governance_composite']:
            was = _wk in st.session_state
            st.session_state.pop(_wk, None)
            logger.info(f"[LOC-CHANGED]   popped {_wk} (was present: {was})")
        st.session_state.shap_stale_fields.update(
            ['region_AFE', 'region_AFW', 'region_EAP', 'region_ECA', 'region_LAC', 'region_MENA', 'region_SAS']
        )

    location = build_location_string(st.session_state.location_countries)

    def on_location_lock_change():
        st.session_state.field_locks['location'] = st.session_state['lock_location_outer']

    _loc_locked = st.session_state.field_locks.get('location', False)
    st.checkbox(
        "🔒 Lock location" if _loc_locked else "🔓 Lock location",
        value=_loc_locked,
        key="lock_location_outer",
        help="Lock to prevent LLM extraction from changing this value",
        on_change=on_location_lock_change
    )

    return location


def render_basic_info_subsection(model_metadata: dict, training_data,
                                 location: str, _llm_running: bool, _shap, _shap_sum):
    st.subheader("Basic Information")
    col1, col3 = st.columns(2)

    with col1:
        def on_reporting_org_change():
            st.session_state.field_edited['reporting_org'] = True
            st.session_state.shap_stale_fields.update(['rep_org_0', 'rep_org_1', 'rep_org_2'])
            selected_org = st.session_state.get('select_reporting_org')
            if selected_org:
                org_dummies = get_org_dummies(selected_org)
                st.session_state.features.update(org_dummies)
                for key in org_dummies.keys():
                    st.session_state.field_edited[key] = True

        badge = get_field_indicator('reporting_org') if st.session_state.field_edited.get('reporting_org', False) else ''
        st.markdown(f"<b>Reporting Organization</b> {badge}", unsafe_allow_html=True)

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_reporting_org_lock_change():
                st.session_state.field_locks['reporting_org'] = st.session_state['lock_reporting_org']
            st.checkbox(
                "🔒" if st.session_state.field_locks.get('reporting_org', False) else "🔓",
                value=st.session_state.field_locks.get('reporting_org', False),
                key="lock_reporting_org",
                help="Lock to prevent LLM from changing this value",
                on_change=on_reporting_org_lock_change
            )
        with col_input:
            if 'select_reporting_org' not in st.session_state and st.session_state.get('features'):
                _f = st.session_state.features
                if _f.get('rep_org_0') == 1:
                    st.session_state['select_reporting_org'] = "UK - Foreign, Commonwealth Development Office (FCDO)"
                elif _f.get('rep_org_1') == 1:
                    st.session_state['select_reporting_org'] = "Asian Development Bank"
                elif _f.get('rep_org_2') == 1:
                    st.session_state['select_reporting_org'] = "World Bank"

            reporting_org = st.selectbox(
                "Reporting Organization",
                KEEP_REPORTING_ORGS,
                help="Select the organization reporting this activity",
                label_visibility="collapsed",
                key="select_reporting_org",
                disabled=st.session_state.field_locks.get('reporting_org', False),
                on_change=on_reporting_org_change
            )
        render_shap_annotation(_shap_sum('rep_org_0', 'rep_org_1', 'rep_org_2'), label="Reporting organisation")

    with col3:
        _ss_start_raw = st.session_state.get('input_start_date')
        if _ss_start_raw:
            default_start = _ss_start_raw
        else:
            extracted_start = st.session_state.extracted_values.get('start_date')
            if extracted_start:
                try:
                    default_start = datetime.fromisoformat(extracted_start).date()
                except Exception:
                    default_start = datetime.now().date()
            else:
                default_start = datetime.now().date()

        def _recompute_duration_from_dates():
            """Recompute planned_duration from the two date widgets."""
            _start = st.session_state.get('input_start_date')
            _end = st.session_state.get('input_planned_end_date')
            if _start and _end:
                _dur = (_end - _start).days / 365.25
                st.session_state.extracted_values['planned_duration'] = _dur
                st.session_state.field_edited['planned_duration'] = True
                st.session_state['input_planned_duration'] = _dur

        def on_start_date_change():
            st.session_state.field_edited['start_date'] = True
            st.session_state.shap_stale_fields.add('planned_duration')
            logger.info(f"[START-DATE-CB] new input_start_date = {st.session_state.get('input_start_date')!r}")
            for _wkey, _fkey in [('input_gdp_percap', 'gdp_percap'),
                                 ('input_cpia_score', 'cpia_score'),
                                 ('input_governance_composite', 'governance_composite')]:
                locked = st.session_state.field_locks.get(_fkey, False)
                was_present = _wkey in st.session_state
                if not locked:
                    st.session_state.pop(_wkey, None)
                logger.info(f"[START-DATE-CB]   {_wkey}: locked={locked}, was_present={was_present}, popped={not locked and was_present}")
            _recompute_duration_from_dates()
            _save_and_rerun()

        badge = get_field_indicator('start_date') if st.session_state.field_edited.get('start_date', False) else ''
        st.markdown(f"<b>Planned Start Date</b> {badge}", unsafe_allow_html=True)

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_start_date_lock_change():
                st.session_state.field_locks['start_date'] = st.session_state['lock_start_date']
            st.checkbox(
                "🔒" if st.session_state.field_locks.get('start_date', False) else "🔓",
                value=st.session_state.field_locks.get('start_date', False),
                key="lock_start_date",
                help="Lock to prevent LLM from changing this value",
                on_change=on_start_date_lock_change
            )
        with col_input:
            _start_kwargs = {} if 'input_start_date' in st.session_state else {"value": default_start}
            start_date = st.date_input(
                "Planned Start Date",
                help="Activity start date (used for GDP, CPIA, and WGI year lookup)",
                label_visibility="collapsed",
                key="input_start_date",
                min_value=date(1900, 1, 1),
                max_value=date(2100, 1, 1),
                disabled=st.session_state.field_locks.get('start_date', False),
                on_change=on_start_date_change,
                **_start_kwargs
            )
        if start_date:
            st.session_state.extracted_values['start_date'] = start_date.isoformat()

        _ss_end_raw = st.session_state.get('input_planned_end_date')
        if _ss_end_raw:
            default_end = _ss_end_raw
        else:
            extracted_end = st.session_state.extracted_values.get('planned_end_date')
            if extracted_end:
                try:
                    default_end = datetime.fromisoformat(extracted_end).date()
                except Exception:
                    _median_dur = model_metadata.get('train_medians', {}).get('planned_duration', 8.46)
                    default_end = default_start.replace(year=default_start.year + int(_median_dur))
            else:
                _median_dur = model_metadata.get('train_medians', {}).get('planned_duration', 8.46)
                default_end = default_start.replace(year=default_start.year + int(_median_dur))

        def on_end_date_change():
            st.session_state.field_edited['planned_end_date'] = True
            st.session_state.shap_stale_fields.add('planned_duration')
            _recompute_duration_from_dates()
            _save_and_rerun()

        badge = get_field_indicator('planned_end_date') if st.session_state.field_edited.get('planned_end_date', False) else ''
        st.markdown(f"<b>Planned End Date</b> {badge}", unsafe_allow_html=True)

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_planned_end_date_lock_change():
                st.session_state.field_locks['planned_end_date'] = st.session_state['lock_planned_end_date']
            st.checkbox(
                "🔒" if st.session_state.field_locks.get('planned_end_date', False) else "🔓",
                value=st.session_state.field_locks.get('planned_end_date', False),
                key="lock_planned_end_date",
                help="Lock to prevent LLM from changing this value",
                on_change=on_planned_end_date_lock_change,
            )
        with col_input:
            _end_kwargs = {} if 'input_planned_end_date' in st.session_state else {"value": default_end}
            rendered_end_date = st.date_input(
                "Planned End Date",
                help="Activity end date (used to compute planned duration)",
                label_visibility="collapsed",
                key="input_planned_end_date",
                min_value=date(1900, 1, 1),
                max_value=date(2100, 1, 1),
                disabled=st.session_state.field_locks.get('planned_end_date', False),
                on_change=on_end_date_change,
                **_end_kwargs
            )
        if rendered_end_date:
            st.session_state.extracted_values['planned_end_date'] = rendered_end_date.isoformat()

        _ss_start = st.session_state.get('input_start_date') or start_date
        _ss_end = st.session_state.get('input_planned_end_date') or rendered_end_date
        if _ss_start and _ss_end:
            _live_dur = (_ss_end - _ss_start).days / 365.25
            st.session_state.extracted_values['planned_duration'] = _live_dur
            st.session_state['input_planned_duration'] = max(0.5, min(10.0, _live_dur)) if _live_dur > 0 else 0.5
            if _live_dur <= 0:
                st.warning(
                    f"End date ({_ss_end}) is before or equal to start date "
                    f"({_ss_start}). Please correct the Planned End Date.",
                    icon="⚠️",
                )

    st.markdown("---")
    st.markdown("### Reporting Organization Distribution")

    org_counts = _compute_org_counts(training_data)
    if org_counts:
        fig_org = go.Figure()
        pull_values = [0.1 if org == reporting_org else 0 for org in KEEP_REPORTING_ORGS]

        fig_org.add_trace(go.Pie(
            labels=list(org_counts.keys()),
            values=list(org_counts.values()),
            pull=pull_values,
            marker=dict(
                colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
            ),
            textinfo='label+percent',
            hovertemplate='%{label}<br>%{value:.0f} activities<br>%{percent}<extra></extra>'
        ))

        fig_org.update_layout(
            title="Activity Database Organization Distribution (your selection highlighted)",
            height=400,
            showlegend=True,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig_org, width='stretch', key="org_pie_chart")
        st.info(f"**Your selection:** {reporting_org}")

    st.markdown("---")
    st.markdown("### Regional Distribution")

    region_cols = ['region_AFE', 'region_AFW', 'region_EAP', 'region_ECA',
                  'region_LAC', 'region_MENA', 'region_SAS']
    region_names = ['Africa East', 'Africa West', 'East Asia & Pacific',
                   'Europe & Central Asia', 'Latin America & Caribbean',
                   'Middle East & North Africa', 'South Asia']

    region_totals = _compute_region_totals(training_data)

    if location and start_date:
        logger.info(f"[LOC-FEAT] location string passed to extract_features_from_location: {location!r}")
        logger.info(f"[LOC-FEAT] start_date: {start_date!r}")
        logger.info(f"[LOC-FEAT] input_gdp_percap in session_state BEFORE extraction: {'input_gdp_percap' in st.session_state}")
        logger.info(f"[LOC-FEAT] input_cpia_score in session_state BEFORE extraction: {'input_cpia_score' in st.session_state}")
        logger.info(f"[LOC-FEAT] input_governance_composite in session_state BEFORE extraction: {'input_governance_composite' in st.session_state}")

        location_features = extract_features_from_location(
            location,
            start_date=start_date,
            activity_scope=str(activity_scope) if 'activity_scope' in locals() else "1"
        )
        logger.info(f"[LOC-FEAT] extracted gdp_percap={location_features.get('gdp_percap')!r}  cpia={location_features.get('cpia_score')!r}  gov={location_features.get('governance_composite')!r}")
        _prev_gdp = st.session_state.extracted_values.get('gdp_percap')
        st.session_state.extracted_values['gdp_percap'] = location_features.get('gdp_percap')
        st.session_state.extracted_values['cpia_score'] = location_features.get('cpia_score')
        st.session_state.extracted_values['governance_composite'] = location_features.get('governance_composite')
        st.session_state.extracted_values['wgi_any_missing'] = location_features.get('wgi_any_missing')
        if location_features.get('gdp_percap') is not None:
            st.session_state.field_edited['gdp_percap'] = True
        if location_features.get('cpia_score') is not None:
            st.session_state.field_edited['cpia_score'] = True
        if location_features.get('governance_composite') is not None:
            st.session_state.field_edited['governance_composite'] = True
        user_regions = {
            region_names[i]: location_features.get(region_cols[i], 0.0)
            for i in range(len(region_cols))
        }
    else:
        location_features = {
            'gdp_percap': None, 'cpia_score': None,
            'governance_composite': None, 'wgi_any_missing': None,
            'region_AFE': 0.0, 'region_AFW': 0.0, 'region_EAP': 0.0,
            'region_ECA': 0.0, 'region_LAC': 0.0, 'region_MENA': 0.0, 'region_SAS': 0.0,
        }
        user_regions = {name: 0.0 for name in region_names}

    fig_region = go.Figure()
    pull_values = [0.1 if user_regions[name] > 0 else 0 for name in region_names]

    fig_region.add_trace(go.Pie(
        labels=region_names,
        values=list(region_totals.values()),
        pull=pull_values,
        marker=dict(
            colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
                   '#FFA15A', '#19D3F3', '#FF6692'],
        ),
        textinfo='label+percent',
        hovertemplate='%{label}<br>%{value:.0f} activities<br>%{percent}<extra></extra>'
    ))

    fig_region.update_layout(
        title="Activity Database: Regional Distribution",
        height=400,
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_region, width='stretch', key="region_chart_main")

    user_region_str = ", ".join([f"{name} ({val*100:.0f}%)"
                                for name, val in user_regions.items() if val > 0])
    if user_region_str:
        st.info(f"**Your activity:** {user_region_str}")
    else:
        st.info("**Your activity:** Enter a location to see your regional allocation")

    return reporting_org, start_date, location_features


def render_activity_features_subsection(model_metadata: dict, training_data,
                                        _llm_running: bool, _shap, _shap_sum,
                                        location_features: dict):
    _raw_pe_median = model_metadata["train_medians"]["planned_expenditure"]
    logger.info(f"[DEBUG] train_medians['planned_expenditure'] = {_raw_pe_median:.4g}  (expecting raw USD ~52M, not log ~17)")
    if _raw_pe_median > 1000:
        default_planned_expenditure = _raw_pe_median
    else:
        logger.warning(f"[DEBUG] WARNING: planned_expenditure median looks like log-USD ({_raw_pe_median:.2f}), doing np.exp()")
        default_planned_expenditure = np.exp(_raw_pe_median)
    logger.info(f"[DEBUG] default_planned_expenditure (USD) = {default_planned_expenditure:.4g}")
    default_planned_duration = model_metadata["train_medians"]["planned_duration"]
    default_activity_scope = int(model_metadata["train_medians"]["activity_scope"])
    default_finance_is_loan = int(model_metadata["train_medians"]["finance_is_loan"])

    st.subheader("Activity Features")

    def _lock_all_features_cb():
        for field in ["planned_expenditure", "planned_duration", "activity_scope", "finance_is_loan"]:
            st.session_state.field_locks[field] = True
            if f"lock_{field}" in st.session_state:
                st.session_state[f"lock_{field}"] = True

    def _unlock_all_features_cb():
        for field in ["planned_expenditure", "planned_duration", "activity_scope", "finance_is_loan"]:
            st.session_state.field_locks[field] = False
            if f"lock_{field}" in st.session_state:
                st.session_state[f"lock_{field}"] = False

    col_lock1, col_lock2, col_lock3 = st.columns([1, 1, 3])
    with col_lock1:
        st.button("🔒 Lock All Features", key="lock_all_activity_features", on_click=_lock_all_features_cb)
    with col_lock2:
        st.button("🔓 Unlock All Features", key="unlock_all_activity_features", on_click=_unlock_all_features_cb)
    with col_lock3:
        st.info("💡 Locked features won't be overwritten during extraction")

    col_input, col_hist = st.columns([1, 1])
    with col_input:
        def on_planned_expenditure_change():
            logger.info("DEBUG CALLBACK - on_planned_expenditure_change FIRED")
            logger.info(f"Time: {datetime.now().isoformat()}")
            logger.info(f"New value from widget: {st.session_state.get('input_planned_expenditure', 'NOT_IN_SESSION_STATE')}")
            st.session_state.field_edited['planned_expenditure'] = True
            st.session_state.shap_stale_fields.add('log_planned_expenditure')

        st.markdown(f"<b>Planned Expenditure (millions USD)</b> {get_field_indicator('planned_expenditure')}", unsafe_allow_html=True)

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_planned_expenditure_lock_change():
                st.session_state.field_locks['planned_expenditure'] = st.session_state['lock_planned_expenditure']

            st.checkbox(
                "🔒" if st.session_state.field_locks.get('planned_expenditure', False) else "🔓",
                value=st.session_state.field_locks.get('planned_expenditure', False),
                key="lock_planned_expenditure",
                help="Lock to prevent LLM from changing this value",
                on_change=on_planned_expenditure_lock_change
            )
        with col_input:
            logger.info("DEBUG WIDGET RENDERING - PLANNED_EXPENDITURE")
            logger.info(f"Time: {datetime.now().isoformat()}")
            logger.info(f"'input_planned_expenditure' in session_state: {'input_planned_expenditure' in st.session_state}")
            if 'input_planned_expenditure' in st.session_state:
                logger.info(f"Current session_state['input_planned_expenditure']: {st.session_state['input_planned_expenditure']}")
            logger.info(f"extracted_values.get('planned_expenditure'): {st.session_state.extracted_values.get('planned_expenditure', 'NOT_SET')}")
            logger.info(f"default median (from activity database): {default_planned_expenditure}")

            if 'input_planned_expenditure' not in st.session_state:
                extracted_expenditure = st.session_state.extracted_values.get('planned_expenditure')
                if extracted_expenditure:
                    default_exp_value = extracted_expenditure / 1_000_000
                    logger.info(f">>> WIDGET NOT IN SESSION_STATE - Using extracted value: {default_exp_value}")
                else:
                    default_exp_value = default_planned_expenditure / 1_000_000
                    logger.info(f">>> WIDGET NOT IN SESSION_STATE - Using default median: {default_exp_value}")
                st.session_state['input_planned_expenditure'] = default_exp_value
                logger.info(f">>> INITIALIZED widget to: {default_exp_value}")
            else:
                logger.info(f">>> WIDGET ALREADY IN SESSION_STATE - KEEPING existing value: {st.session_state['input_planned_expenditure']}")

            planned_expenditure_millions = st.number_input(
                "Planned Expenditure",
                min_value=0.01,
                max_value=100000.0, # THIS IS IMPORTANT, ACTIVITIES CAN BE IN THE BILLIONS, DO NOT LOWER TO LESS THAN THIS!
                step=0.1,
                help="Total planned budget in millions of USD",
                key="input_planned_expenditure",
                label_visibility="collapsed",
                disabled=st.session_state.field_locks.get('planned_expenditure', False),
                on_change=on_planned_expenditure_change,
            )
            render_shap_annotation(_shap('log_planned_expenditure'), label="Planned expenditure")
        planned_expenditure = planned_expenditure_millions * 1_000_000

    with col_hist:
        if "planned_expenditure" in training_data.columns:
            train_exp_log = np.log(training_data["planned_expenditure"].dropna())
            fig_exp = go.Figure()

            fig_exp.add_trace(go.Histogram(
                x=train_exp_log,
                name="Activity Database",
                marker_color="lightblue",
                opacity=0.7,
                nbinsx=30
            ))

            median_log_exp = np.median(train_exp_log)
            fig_exp.add_vline(
                x=median_log_exp,
                line_dash="dot",
                line_color="grey",
                line_width=1.5,
                annotation_text="Median",
                annotation_position="top left",
                annotation_font_color="grey"
            )

            if st.session_state.show_distribution_markers:
                user_log_exp = np.log(planned_expenditure)
                fig_exp.add_vline(
                    x=user_log_exp,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text="You",
                    annotation_position="top"
                )

            fig_exp.update_layout(
                title="Planned Expenditure (log scale)",
                xaxis_title="ln(Expenditure in USD)",
                yaxis_title="Count",
                height=200,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig_exp, width='stretch', key="hist_planned_expenditure")

    col_input, col_hist = st.columns([1, 1])
    with col_input:
        extracted_duration = st.session_state.extracted_values.get('planned_duration')
        if 'input_planned_duration' not in st.session_state:
            if extracted_duration and extracted_duration > 0:
                default_dur_value = max(0.1, min(100.0, extracted_duration))
            else:
                default_dur_value = default_planned_duration
            st.session_state['input_planned_duration'] = default_dur_value

        _dur_indicator = get_field_indicator('planned_duration')
        st.markdown(f"**Planned Duration (years)** {_dur_indicator}", unsafe_allow_html=True)

        planned_duration = st.number_input(
            "Planned Duration",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            help="Computed automatically from Planned Start Date and Planned End Date above.",
            key="input_planned_duration",
            label_visibility="collapsed",
            disabled=True,
        )
        render_shap_annotation(_shap('planned_duration'), label="Planned duration")

    with col_hist:
        if "planned_duration" in training_data.columns:
            train_dur = training_data["planned_duration"].dropna()
            fig_dur = render_histogram(
                "Planned Duration (years)",
                train_dur,
                planned_duration,
                show_marker=st.session_state.show_distribution_markers,
                height=200
            )
            if fig_dur:
                st.plotly_chart(fig_dur, width='stretch', key="hist_planned_duration")

    col_input, col_hist = st.columns([1, 1])
    with col_input:
        def on_activity_scope_change():
            st.session_state.field_edited['activity_scope'] = True
            st.session_state.shap_stale_fields.add('activity_scope')

        st.markdown(f"<b>Activity Scope</b> {get_field_indicator('activity_scope')}", unsafe_allow_html=True)

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_activity_scope_lock_change():
                st.session_state.field_locks['activity_scope'] = st.session_state['lock_activity_scope']

            st.checkbox(
                "🔒" if st.session_state.field_locks.get('activity_scope', False) else "🔓",
                value=st.session_state.field_locks.get('activity_scope', False),
                key="lock_activity_scope",
                help="Lock to prevent LLM from changing this value",
                on_change=on_activity_scope_lock_change
            )
        with col_input:
            _scope_options = list(SCOPE_LABELS.keys())
            if 'input_activity_scope' not in st.session_state:
                _scope_val = int(st.session_state.extracted_values.get('activity_scope', default_activity_scope))
                st.session_state['input_activity_scope'] = _scope_val

            activity_scope = st.selectbox(
                "Activity Scope",
                _scope_options,
                format_func=lambda x: SCOPE_LABELS[x],
                help="Geographic scope of the activity (IATI standard codes)",
                key="input_activity_scope",
                label_visibility="collapsed",
                disabled=st.session_state.field_locks.get('activity_scope', False),
                on_change=on_activity_scope_change
            )
            render_shap_annotation(_shap('activity_scope'), label="Activity scope")

    with col_hist:
        if "activity_scope" in training_data.columns:
            SCOPE_NAMES = {
                1: "Global",
                2: "Regional",
                3: "Multi-national",
                4: "National",
                5: "Sub-national: Multi-first-level admin areas",
                6: "Sub-national: Single first-level admin area",
                7: "Sub-national: Single second-level admin area",
                8: "Single location",
            }

            train_scope = training_data["activity_scope"].dropna()
            fig_scope = go.Figure()

            fig_scope.add_trace(go.Histogram(
                x=train_scope,
                name="median",
                marker_color="lightblue",
                opacity=0.7,
                xbins=dict(start=0.5, end=8.5, size=1)
            ))

            median_scope = np.median(train_scope)
            fig_scope.add_vline(
                x=median_scope,
                line_dash="dot",
                line_color="grey",
                line_width=1.5,
                annotation_text="Median",
                annotation_position="top left",
                annotation_font_color="grey"
            )

            if st.session_state.show_distribution_markers:
                fig_scope.add_vline(
                    x=activity_scope,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text="You",
                    annotation_position="top"
                )

            fig_scope.update_layout(
                title="Activity Scope",
                xaxis_title="",
                yaxis_title="Count",
                height=200,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=120),
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(1, 9)),
                    ticktext=[SCOPE_NAMES[i] for i in range(1, 9)],
                    tickangle=-90
                )
            )

            st.plotly_chart(fig_scope, width='stretch', key="hist_activity_scope")

    col_input, col_hist = st.columns([1, 1])
    with col_input:
        def on_finance_is_loan_change():
            st.session_state.field_edited['finance_is_loan'] = True
            st.session_state.shap_stale_fields.add('finance_is_loan')

        st.markdown(f"<b>Finance Type</b> {get_field_indicator('finance_is_loan')}", unsafe_allow_html=True)

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_finance_is_loan_lock_change():
                st.session_state.field_locks['finance_is_loan'] = st.session_state['lock_finance_is_loan']

            st.checkbox(
                "🔒" if st.session_state.field_locks.get('finance_is_loan', False) else "🔓",
                value=st.session_state.field_locks.get('finance_is_loan', False),
                key="lock_finance_is_loan",
                help="Lock to prevent LLM from changing this value",
                on_change=on_finance_is_loan_lock_change
            )
        with col_input:
            finance_is_loan = st.selectbox(
                "Finance Type",
                [0, 1],
                index=[0, 1].index(default_finance_is_loan) if default_finance_is_loan in [0, 1] else 0,
                format_func=lambda x: "Grant" if x == 0 else "Loan",
                key="input_finance_is_loan",
                label_visibility="collapsed",
                disabled=st.session_state.field_locks.get('finance_is_loan', False),
                on_change=on_finance_is_loan_change
            )
            render_shap_annotation(_shap('finance_is_loan'), label="Finance type (loan vs grant)")

    with col_hist:
        if "finance_is_loan" in training_data.columns:
            train_loan = training_data["finance_is_loan"].dropna()
            fig_loan = render_histogram(
                "Finance Type (0=Grant, 1=Loan)",
                train_loan,
                finance_is_loan,
                show_marker=st.session_state.show_distribution_markers,
                height=200
            )
            if fig_loan:
                st.plotly_chart(fig_loan, width='stretch', key="hist_finance_is_loan")

    st.markdown("**Country Features** *(auto-extracted from location and start date above — edit to override)*")
    _train_median_gdp = np.exp(model_metadata["train_medians"]["gdp_percap"])  # back from log scale
    _train_median_cpia = model_metadata["train_medians"]["cpia_score"]
    _train_median_gov = model_metadata["train_medians"]["governance_composite"]

    col_gdp_input, col_gdp_hist = st.columns([1, 1])
    with col_gdp_input:
        st.markdown(f"<b>GDP per Capita (USD)</b> {get_field_indicator('gdp_percap')}", unsafe_allow_html=True)

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_gdp_percap_lock_change():
                st.session_state.field_locks['gdp_percap'] = st.session_state['lock_gdp_percap']

            st.checkbox(
                "🔒" if st.session_state.field_locks.get('gdp_percap', False) else "🔓",
                value=st.session_state.field_locks.get('gdp_percap', False),
                key="lock_gdp_percap",
                help="Lock to prevent auto-extraction from changing this value",
                on_change=on_gdp_percap_lock_change
            )
        with col_input:
            if 'input_gdp_percap' not in st.session_state:
                _extracted_gdp = st.session_state.extracted_values.get('gdp_percap')
                _default_gdp = float(_extracted_gdp) if _extracted_gdp is not None else float(_train_median_gdp)
                st.session_state['input_gdp_percap'] = _default_gdp

            def on_gdp_percap_change():
                val = st.session_state.get('input_gdp_percap', _train_median_gdp)
                log_val = np.log(float(val)) if float(val) > 0 else None
                log_median = model_metadata["train_medians"]["gdp_percap"]
                if log_val is not None and abs(log_val - float(log_median)) > 0.01:
                    st.session_state.field_edited['gdp_percap'] = True
                else:
                    st.session_state.field_edited['gdp_percap'] = False
                st.session_state.shap_stale_fields.add('gdp_percap')

            gdp_percap_input = st.number_input(
                "GDP per Capita",
                min_value=0.0,
                step=500.0,
                help="GDP per capita in USD (sourced from World Bank, matched to activity start year)",
                label_visibility="collapsed",
                key="input_gdp_percap",
                disabled=st.session_state.field_locks.get('gdp_percap', False),
                on_change=on_gdp_percap_change,
            )
            render_shap_annotation(_shap('gdp_percap'), label="GDP per capita")
    with col_gdp_hist:
        if "gdp_percap" in training_data.columns:
            _gdp_log = np.log(gdp_percap_input) if gdp_percap_input > 0 else None
            fig_gdp = render_histogram(
                "GDP per Capita (log scale)",
                training_data["gdp_percap"].dropna(),
                _gdp_log,
                show_marker=st.session_state.show_distribution_markers and _gdp_log is not None,
                height=200
            )
            if fig_gdp:
                st.plotly_chart(fig_gdp, width='stretch', key="hist_gdp_percap")

    col_cpia_input, col_cpia_hist = st.columns([1, 1])
    with col_cpia_input:
        st.markdown(f"<b>CPIA Score (1–6)</b> {get_field_indicator('cpia_score')}", unsafe_allow_html=True)
        st.caption("Country Policy & Institutional Assessment. N/A for non-IDA countries.")

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_cpia_score_lock_change():
                st.session_state.field_locks['cpia_score'] = st.session_state['lock_cpia_score']

            st.checkbox(
                "🔒" if st.session_state.field_locks.get('cpia_score', False) else "🔓",
                value=st.session_state.field_locks.get('cpia_score', False),
                key="lock_cpia_score",
                help="Lock to prevent auto-extraction from changing this value",
                on_change=on_cpia_score_lock_change
            )
        with col_input:
            _extracted_cpia = st.session_state.extracted_values.get('cpia_score')
            if 'input_cpia_score' not in st.session_state:
                _default_cpia = float(_extracted_cpia) if _extracted_cpia is not None else float(_train_median_cpia)
                st.session_state['input_cpia_score'] = _default_cpia

            def on_cpia_score_change():
                val = st.session_state.get('input_cpia_score', _train_median_cpia)
                if abs(float(val) - float(_train_median_cpia)) > 0.01:
                    st.session_state.field_edited['cpia_score'] = True
                else:
                    st.session_state.field_edited['cpia_score'] = False
                st.session_state.shap_stale_fields.add('cpia_score')

            cpia_score_input = st.number_input(
                "CPIA Score",
                min_value=1.0,
                max_value=6.0,
                step=0.1,
                help="CPIA score 1–6 (World Bank IDA countries only; use database median for others)",
                label_visibility="collapsed",
                key="input_cpia_score",
                disabled=st.session_state.field_locks.get('cpia_score', False),
                on_change=on_cpia_score_change,
            )
            render_shap_annotation(_shap('cpia_score'), label="CPIA score")
            cpia_missing_input = _extracted_cpia is None and not st.session_state.field_edited.get('cpia_score', False)
    with col_cpia_hist:
        if "cpia_score" in training_data.columns:
            fig_cpia = render_histogram(
                "CPIA Score",
                training_data["cpia_score"].dropna(),
                cpia_score_input,
                show_marker=st.session_state.show_distribution_markers,
                height=200
            )
            if fig_cpia:
                st.plotly_chart(fig_cpia, width='stretch', key="hist_cpia_score")

    col_gov_input, col_gov_hist = st.columns([1, 1])
    with col_gov_input:
        st.markdown(f"<b>Governance Composite</b> {get_field_indicator('governance_composite')}", unsafe_allow_html=True)
        st.caption("Mean of 5 World Governance Indicators (WGI). Range ~−2.5 to +2.5.")

        col_lock, col_input = st.columns([0.15, 0.85])
        with col_lock:
            def on_governance_composite_lock_change():
                st.session_state.field_locks['governance_composite'] = st.session_state['lock_governance_composite']

            st.checkbox(
                "🔒" if st.session_state.field_locks.get('governance_composite', False) else "🔓",
                value=st.session_state.field_locks.get('governance_composite', False),
                key="lock_governance_composite",
                help="Lock to prevent auto-extraction from changing this value",
                on_change=on_governance_composite_lock_change
            )
        with col_input:
            if 'input_governance_composite' not in st.session_state:
                _extracted_gov = st.session_state.extracted_values.get('governance_composite')
                _default_gov = float(_extracted_gov) if _extracted_gov is not None else float(_train_median_gov)
                st.session_state['input_governance_composite'] = _default_gov

            def on_governance_composite_change():
                val = st.session_state.get('input_governance_composite', _train_median_gov)
                if abs(float(val) - float(_train_median_gov)) > 0.001:
                    st.session_state.field_edited['governance_composite'] = True
                else:
                    st.session_state.field_edited['governance_composite'] = False
                st.session_state.shap_stale_fields.add('governance_composite')

            governance_composite_input = st.number_input(
                "Governance Composite",
                min_value=-3.0,
                max_value=3.0,
                step=0.01,
                help="Mean of Control of Corruption, Political Stability, Govt Effectiveness, Regulatory Quality, Rule of Law",
                label_visibility="collapsed",
                key="input_governance_composite",
                disabled=st.session_state.field_locks.get('governance_composite', False),
                on_change=on_governance_composite_change,
            )
            render_shap_annotation(_shap('governance_composite'), label="Governance composite")
            _extracted_wgi = st.session_state.extracted_values.get('wgi_any_missing')
            wgi_any_missing_input = float(_extracted_wgi) if _extracted_wgi is not None else 0.0
    with col_gov_hist:
        if "governance_composite" in training_data.columns:
            fig_gov = render_histogram(
                "Governance Composite",
                training_data["governance_composite"].dropna(),
                governance_composite_input,
                show_marker=st.session_state.show_distribution_markers,
                height=200
            )
            if fig_gov:
                st.plotly_chart(fig_gov, width='stretch', key="hist_governance_composite")

    return planned_expenditure, planned_duration, activity_scope, finance_is_loan, gdp_percap_input, cpia_score_input, governance_composite_input
