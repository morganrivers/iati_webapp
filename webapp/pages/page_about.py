"""About page – static description of the forecasting tool."""
import streamlit as st


def render_about_page():
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
