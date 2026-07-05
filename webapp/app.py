"""IATI Activity Success Forecasting Web App"""
import logging
import warnings

from logging_config import setup_logging

setup_logging()

import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings('ignore', message=".*force_all_finite.*", category=FutureWarning)
load_dotenv()

from webapp_paths import ensure_src_paths
ensure_src_paths()


st.set_page_config(
    page_title="IATI Activity Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
    /* Hide Streamlit's auto-generated pages/ navigation */
    [data-testid="stSidebarNav"] { display: none; }

    /* Override Streamlit's Source Sans variable font with system fonts.
       Source Sans has wider metrics that cause a layout reflow on load.
       Exclude Streamlit's icon spans so their Material Symbols ligatures still
       render as glyphs instead of the literal ligature text (e.g. "arrow_right").
       Streamlit renders icons two ways, neither using a material-symbols class:
         - StyledMaterialIcon: <span data-testid="stIconMaterial"> (emotion font-family)
         - remark material icons / emoji: <span role="img" style="font-family:..."> (inline) */
    *:not([data-testid="stIconMaterial"]):not([role="img"]),
    *::before, *::after {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     Helvetica, Arial, sans-serif !important;
        letter-spacing: normal !important;
        word-spacing: normal !important;
    }
    /* Re-assert the icon font on the emotion-styled icon span (no inline style,
       so it needs an explicit rule that outranks the wildcard override). */
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
        letter-spacing: normal !important;
        word-spacing: normal !important;
    }
    code, pre, kbd {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
    }

    /* Nav buttons: full-width, left-aligned text, subtle styling */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start;
        border-radius: 6px;
        border: none;
        padding: 0.45rem 0.75rem;
        font-size: 0.95rem;
        background: transparent;
        color: inherit;
        transition: background 0.15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(128,128,128,0.15);
    }
    /* Active page button */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(255,75,75,0.12);
        color: #ff4b4b;
        font-weight: 600;
    }
</style>""", unsafe_allow_html=True)

_loading_slot = st.empty()
if '_app_initialized' not in st.session_state:
    with _loading_slot.container():
        _c1, _c2, _c3 = st.columns([1, 2, 1])
        with _c2:
            st.markdown("# 🌍 IATI Activity Forecasting")
            st.markdown("### Predicting the success of international aid activities")
            st.markdown("""
This tool forecasts the likely overall success rating of an aid activity at time of evaluation,
using a machine learning model trained on ~1,300 environmental and sustainability activities
from the [International Aid Transparency Initiative (IATI)](https://iatistandard.org/) database.

**How it works:**
1. Upload a project document (PDF) or manually enter activity details
2. The system extracts key features using an LLM and computes embeddings
3. A Random Forest and ExtraTrees ensemble predicts a success rating on a 0–5 scale
4. A narrative forecast with LLM commentary uses the prior prediction as a reference point to predict the likely outcomes of the activity

**Organisations covered:** UK FCDO · Asian Development Bank · World Bank · BMZ
            """)
            st.info("⏳ Loading model and data, please wait…")

from model_loader import load_model_and_data
from state_manager import initialize_session_state
from pages.page_activity_forecasting import render_activity_forecasting_page
from pages.page_extracted_data import render_extracted_data_page
from pages.page_model_performance import render_model_performance_page
from pages.page_rag_forecast import render_rag_forecast_page
from pages.page_glossary import render_glossary_page
from pages.page_about import render_about_page
from pages.page_feedback import render_feedback_page
from project_manager import save_project_state_temp

rf_model, extra_model, per_org_baseline, start_year_correction, model_metadata, training_data = load_model_and_data()
initialize_session_state()
st.session_state._app_initialized = True
_loading_slot.empty()

_PAGES = [
    ("ℹ️", "About"),
    ("🏠", "Activity Forecasting"),
    ("📋", "View Extracted Data"),
    ("🕮", "Narrative Forecast [beta]"),
    ("📊", "Model Performance"),
    ("🔍︎", "Glossary"),
    ("🖂", "Feedback"),
]
_PAGE_KEYS = [f"{icon} {name}" for icon, name in _PAGES]

if "current_page" not in st.session_state:
    st.session_state.current_page = _PAGE_KEYS[0]

st.markdown("""
<style>
    [data-testid="stSidebarContent"] { padding-top: 1rem; }
    [data-testid="stSidebarContent"] h3 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("### Navigation")

# Track previous page for auto-save detection
if "previous_page" not in st.session_state:
    st.session_state.previous_page = st.session_state.current_page

for _key in _PAGE_KEYS:
    _is_active = st.session_state.current_page == _key
    if st.sidebar.button(_key, use_container_width=True, type="primary" if _is_active else "secondary"):
        # Auto-save temp state if leaving Activity Forecasting page
        if (st.session_state.current_page == "🏠 Activity Forecasting" and
            _key != "🏠 Activity Forecasting" and
            st.session_state.get('selected_project_folder')):
            logger.info(">>> Auto-saving temp state before navigating away from Activity Forecasting")
            save_project_state_temp(st.session_state.selected_project_folder)

        st.session_state.previous_page = st.session_state.current_page
        st.session_state.current_page = _key
        st.rerun()

page = st.session_state.current_page

_page_container = st.empty()
with _page_container.container():
    if page == "🏠 Activity Forecasting":
        render_activity_forecasting_page(rf_model, extra_model, per_org_baseline, start_year_correction, model_metadata, training_data)
    elif page == "📋 View Extracted Data":
        render_extracted_data_page()
    elif page == "🕮 Narrative Forecast [beta]":
        render_rag_forecast_page()
    elif page == "📊 Model Performance":
        render_model_performance_page(training_data, model_metadata)
    elif page == "🔍︎ Glossary":
        render_glossary_page()
    elif page == "ℹ️ About":
        render_about_page()
    elif page == "🖂 Feedback":
        render_feedback_page()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with a Random Forest and ExtraTrees ensemble trained on IATI activity data.<p>
    <p>© Morgan Rivers, 2026.</p>
</div>
""", unsafe_allow_html=True)
