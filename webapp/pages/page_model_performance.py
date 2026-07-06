import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model_loader import BASE_PATH
from ui_components import add_glossary_hover_bar


def render_model_performance_page(training_data, model_metadata):
    st.title("📊 Model Performance & Accuracy")
    st.markdown("""
    This page shows which features have the most impact on the Random Forest model's predictions,
    and reports model performance on the validation set.
    This analysis is an average of all activities in the database.
    """)

    # # Show selected project info
    # if st.session_state.selected_project_folder:
    #     st.info(f"**Current Activity:** {st.session_state.project_name} (ID: `{st.session_state.selected_project_folder}`)")

    # Load feature importances from CSV
    importance_df = pd.read_csv(BASE_PATH / "feature_importances.csv")

    # Sort by absolute delta_pred_1sd (all features)
    importance_df_sorted = importance_df.sort_values("delta_pred_1sd", ascending=True, key=abs)

    # Calculate height based on number of features (min 800)
    num_features = len(importance_df_sorted)
    chart_height = max(800, num_features * 25)

    # -------------------------------------------------------------------------
    # 1. Tree Importance (ALL FEATURES)
    # -------------------------------------------------------------------------
    st.markdown("## Tree Importance (All Features)")
    st.markdown("""
    Shows how often each feature is used for splitting decisions in the Random Forest.
    Higher values indicate the feature is more frequently used by the model.
    """)

    fig_tree = go.Figure()
    _tree_hover = add_glossary_hover_bar(
        fig_tree,
        importance_df_sorted["feature"].tolist(),
        importance_df_sorted["importance"].tolist(),
        x_label='Tree Importance',
        x_format='.4f',
    )
    fig_tree.add_trace(go.Bar(
        x=importance_df_sorted["importance"],
        y=importance_df_sorted["feature"],
        orientation='h',
        marker_color='coral',
        hovertemplate=_tree_hover['hover_template'],
        customdata=_tree_hover['customdata'],
    ))

    fig_tree.update_layout(
        barmode='overlay',
        xaxis=dict(title="Tree Importance", range=[_tree_hover['x_lo'], _tree_hover['x_hi']]),
        yaxis=dict(
            title="Feature",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            griddash='dot'
        ),
        height=chart_height,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=40),
        hovermode='closest',
    )

    st.plotly_chart(fig_tree, width='stretch')

    # -------------------------------------------------------------------------
    # 2. Delta Prediction (ALL FEATURES, sorted by absolute value)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## Δ Prediction Impact (All Features, sorted by absolute impact)")
    st.markdown("""
    Shows how much the prediction changes when a feature shifts by 1 standard deviation.
    Sorted by absolute magnitude of impact.
    """)

    fig_delta_abs = go.Figure()

    # Color bars by sign (negative = red, positive = green)
    bar_colors_abs = ['crimson' if x < 0 else 'mediumseagreen'
                     for x in importance_df_sorted["delta_pred_1sd"]]

    _delta_abs_hover = add_glossary_hover_bar(
        fig_delta_abs,
        importance_df_sorted["feature"].tolist(),
        importance_df_sorted["delta_pred_1sd"].tolist(),
        x_label='Δ Prediction',
    )
    fig_delta_abs.add_trace(go.Bar(
        x=importance_df_sorted["delta_pred_1sd"],
        y=importance_df_sorted["feature"],
        orientation='h',
        marker_color=bar_colors_abs,
        hovertemplate=_delta_abs_hover['hover_template'],
        customdata=_delta_abs_hover['customdata'],
    ))

    fig_delta_abs.update_layout(
        barmode='overlay',
        xaxis=dict(
            title="Δ Prediction (1 SD shift)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            range=[_delta_abs_hover['x_lo'], _delta_abs_hover['x_hi']],
        ),
        yaxis=dict(
            title="Feature",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            griddash='dot'
        ),
        height=chart_height,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=40),
        hovermode='closest',
    )

    st.plotly_chart(fig_delta_abs, width='stretch')

    st.info("💡 **Green bars** = feature increases prediction when increased. **Red bars** = feature decreases prediction when increased.")

    # -------------------------------------------------------------------------
    # 3. Delta Prediction Impact (sorted by signed value, negative to positive)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## Δ Prediction Impact (sorted from most negative to most positive)")
    st.markdown("""
    Same data as above, but sorted from most negative impact (bottom) to most positive impact (top).
    This view helps identify which features increase vs decrease predictions.
    """)

    # Sort by delta_pred_1sd (preserving sign, most negative to most positive)
    importance_df_delta_sorted = importance_df.sort_values("delta_pred_1sd", ascending=True)

    # Create single bar chart for delta prediction
    fig_delta = go.Figure()

    # Color bars by sign (negative = red, positive = green)
    bar_colors = ['crimson' if x < 0 else 'mediumseagreen'
                 for x in importance_df_delta_sorted["delta_pred_1sd"]]

    _delta_hover = add_glossary_hover_bar(
        fig_delta,
        importance_df_delta_sorted["feature"].tolist(),
        importance_df_delta_sorted["delta_pred_1sd"].tolist(),
        x_label='Δ Prediction',
    )
    fig_delta.add_trace(go.Bar(
        x=importance_df_delta_sorted["delta_pred_1sd"],
        y=importance_df_delta_sorted["feature"],
        orientation='h',
        marker_color=bar_colors,
        hovertemplate=_delta_hover['hover_template'],
        customdata=_delta_hover['customdata'],
    ))

    fig_delta.update_layout(
        barmode='overlay',
        xaxis=dict(
            title="Δ Prediction (1 SD shift)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            range=[_delta_hover['x_lo'], _delta_hover['x_hi']],
        ),
        yaxis=dict(
            title="Feature",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            griddash='dot'
        ),
        height=chart_height,
        margin=dict(l=20, r=20, t=40, b=40),
        showlegend=False,
        hovermode='closest',
    )

    st.plotly_chart(fig_delta, width='stretch')

    st.info("💡 **Green bars** = feature increases prediction when increased. **Red bars** = feature decreases prediction when increased.")

    # -------------------------------------------------------------------------
    # 4. Model Performance Metrics
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Model Performance on Validation Set")
    st.markdown("""
    Performance measured on held-out activities from later-starting projects.
    The per-organization baseline is combined with a Random Forest / ExtraTrees ensemble
    and a start-year correction that adjusts for temporal drift.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Ensemble - Pairwise accuracy", "77%")
        st.metric("Ensemble - RMSE", "0.821")

    with col2:
        st.metric("Ensemble - R²", "0.248")

    st.info("**Pairwise accuracy** measures how likely it is that two random activity's evaluation scores are correctly ranked by the model, such that the higher rating was predicted to be higher. **RMSE** measures average prediction error in rating points (lower is better). **R²** measures variance explained (higher is better).")