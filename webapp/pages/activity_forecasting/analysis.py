import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shap_explainer import get_sorted_contributions
from ui_components import add_glossary_hover_bar, render_world_bank_rating_headline


def render_analysis_section() -> None:
    # Params: (none) — reads st.session_state.prediction, shap_result, feature_table, etc.
    # Returns: None
    TRAIN_MEAN_RATING = 3.33

    if "prediction" not in st.session_state:
        return

    render_world_bank_rating_headline(st.session_state.prediction)

    # Show baseline, adjustment, and final prediction
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Per-Org Baseline",
            value=f"{st.session_state.base:.2f}",
            help="Baseline rating for the reporting organization"
        )
    with col2:
        _total_adjustment = st.session_state.prediction - st.session_state.base
        st.metric(
            label="Model Adjustment",
            value=f"{_total_adjustment:+.2f}",
            help="Total shift from the per-org baseline (RF/ExtraTrees ensemble delta + start-year correction)"
        )
    with col3:
        st.metric(
            label="⭐ Final Prediction",
            value=f"{st.session_state.prediction:.2f}",
            help="Baseline + ensemble adjustment + start-year correction"
        )
    with col4:
        st.markdown(
            f'<div style="color: grey">Database Average</div>'
            f'<div style="color: grey">{TRAIN_MEAN_RATING:.2f}</div>',
            unsafe_allow_html=True
        )

    _HALF_WIDTH_90 = 1.25
    _ci_lo = float(np.clip(st.session_state.prediction - _HALF_WIDTH_90, 0, 5))
    _ci_hi = float(np.clip(st.session_state.prediction + _HALF_WIDTH_90, 0, 5))
    st.caption(
        f"Model-level 90% CI: **{_ci_lo:.2f} – {_ci_hi:.2f}** "
        f"(±{_HALF_WIDTH_90} half-width, clipped to [0, 5]; "
        "this reflects typical model error, not uncertainty specific to this activity)"
    )

    st.header("Prediction Interpretation")
    st.subheader("Understanding Your Forecast")
    st.markdown("See what factors drove the prediction for this specific activity.")

    if st.session_state.get("prediction_explanation"):
        st.markdown(f"### Explanation\n\n{st.session_state.prediction_explanation}")

    # SHAP Feature Contributions Chart
    st.markdown("---")
    st.subheader("Feature Contributions to This Prediction")
    st.markdown("Shows how each feature pushed the prediction up or down from the baseline.")

    # Feature values table — persists across reruns (reads from session state)
    with st.expander("🔍 Feature Values (click to expand)", expanded=False):
        st.markdown("**Feature values used for prediction:**")
        st.markdown("- 🟢 **User-provided** values (LLM grades, activity details, org, sectors)")
        st.markdown("- 🟡 **Extracted** from location/activity data (GDP, CPIA, governance, regions)")
        st.markdown("- ⛁ **Database median** (UMAP, distance features)")
        st.markdown("- 🟡 **Computed** (missing flags, completeness metrics)")

        if 'feature_table' in st.session_state:
            feature_df = pd.DataFrame(st.session_state.feature_table)
            st.dataframe(feature_df, width='stretch', hide_index=True)

    if 'shap_result' in st.session_state:
        sorted_contrib = get_sorted_contributions(st.session_state.shap_result)

        bar_colors = ['crimson' if x < 0 else 'mediumseagreen'
                      for x in sorted_contrib['shap_values']]

        fig_shap = go.Figure()
        hover = add_glossary_hover_bar(
            fig_shap,
            sorted_contrib['feature_names'],
            sorted_contrib['shap_values'],
            x_label='Contribution',
        )

        fig_shap.add_trace(go.Bar(
            x=sorted_contrib['shap_values'],
            y=sorted_contrib['feature_names'],
            orientation='h',
            marker_color=bar_colors,
            hovertemplate=hover['hover_template'],
            customdata=hover['customdata'],
        ))

        num_features = len(sorted_contrib['feature_names'])
        fig_shap.update_layout(
            barmode='overlay',
            xaxis=dict(
                title="SHAP Value (Contribution to Prediction)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                range=[hover['x_lo'], hover['x_hi']],
            ),
            yaxis=dict(
                title="Feature",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.08)',
                griddash='dot',
                autorange='reversed'
            ),
            height=max(600, num_features * 25),
            margin=dict(l=20, r=20, t=40, b=40),
            showlegend=False,
            hovermode='closest',
        )

        st.plotly_chart(fig_shap, width='stretch')

        base_val = st.session_state.shap_result['base_value']
        st.info(f"💡 **Baseline prediction** (model average): {base_val:.2f}. Positive contributions (green) push the prediction higher, negative contributions (red) push it lower. All contributions sum to the final prediction.")
    else:
        st.warning("SHAP values not available. Please regenerate the prediction.")

    # Bar chart comparing per-org baseline and final prediction
    st.markdown("---")
    st.subheader("Model Components")
    st.markdown("Per-organization baseline plus the Random Forest / ExtraTrees ensemble adjustment.")

    fig = go.Figure(data=[
        go.Bar(name='Per-Org Baseline', x=['Prediction'], y=[st.session_state.base], marker_color='lightblue'),
        go.Bar(name='Final Prediction', x=['Prediction'], y=[st.session_state.prediction], marker_color='gold')
    ])
    fig.add_hline(
        y=TRAIN_MEAN_RATING,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"Database avg ({TRAIN_MEAN_RATING:.2f})",
        annotation_position="top right",
        annotation_font_color="grey"
    )
    fig.update_layout(
        title='Model Components',
        yaxis_title='Predicted Rating',
        yaxis=dict(range=[0, 5]),
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, width='stretch')

    _ens = st.session_state.ens_delta
    _year_corr = st.session_state.get("year_correction", 0.0)
    st.info(
        f"RF/ExtraTrees delta: {_ens:+.3f} | "
        f"Start-year correction: {_year_corr:+.3f} | "
        f"Total adjustment: {st.session_state.prediction - st.session_state.base:+.3f}"
    )
