import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


import logging

logger = logging.getLogger(__name__)

def render_shap_annotation(shap_contribution: float | None, label: str = None):
    """Render a small inline SHAP contribution badge below a widget."""
    if shap_contribution is None:
        return
    _sign = "+" if shap_contribution > 0 else ""
    _tooltip = f"{label + ': ' if label else ''}shifts predicted rating by {_sign}{shap_contribution:.2f} (on a 0–5 scale)"
    if round(shap_contribution, 2) == 0.0:
        st.markdown(
            f'<div style="background:#e9ecef; color:#6c757d; '
            f'padding:2px 8px; border-radius:4px; font-size:0.82em; '
            f'margin-top:-8px; display:inline-block; cursor:default;" '
            f'title="{_tooltip}">0.00</div>',
            unsafe_allow_html=True
        )
        return
    _abs = abs(shap_contribution)
    _arrows = "↑↑↑" if _abs >= 0.25 else ("↑↑" if _abs >= 0.10 else "↑")
    if shap_contribution < 0:
        _arrows = _arrows.replace("↑", "↓")
    _color = "#d4edda" if shap_contribution >= 0 else "#f8d7da"
    _text_color = "#155724" if shap_contribution >= 0 else "#721c24"
    st.markdown(
        f'<div style="background:{_color}; color:{_text_color}; '
        f'padding:2px 8px; border-radius:4px; font-size:0.82em; '
        f'margin-top:-8px; display:inline-block; cursor:default;" '
        f'title="{_tooltip}">'
        f'{_arrows} {_sign}{shap_contribution:.2f}</div>',
        unsafe_allow_html=True
    )


def get_field_indicator(field_name: str) -> str:
    """
    Get color indicator for a field based on whether it's been edited.
    Returns HTML for colored badge.
    """
    is_edited = st.session_state.field_edited.get(field_name, False)
    if is_edited:
        return '<span style="background-color: #d4edda; color: #155724; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; margin-left: 5px;">✓ Set</span>'
    else:
        return '<span style="background-color: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; margin-left: 5px;">⚠ Using median</span>'


def render_llm_feature_slider(feature_name: str, display_name: str, default_value: float, help_text: str, training_data_col: str = None, training_data=None, shap_contribution: float = None):
    """Render a single LLM feature slider with lock and indicator badge."""

    # Define callback to track edits (runs BEFORE page renders)
    def on_slider_change():
        try:
            import traceback
            value = st.session_state.get(f"input_{feature_name}", default_value)
            # print(f"[SLIDER_DEBUG] on_slider_change: input_{feature_name} = {value}  (default={default_value})")
            # print(f"[SLIDER_DEBUG] on_slider_change caller stack:")
            traceback.print_stack(limit=8)
            if abs(value - default_value) > 0.01:
                st.session_state.field_edited[feature_name] = True
            else:
                st.session_state.field_edited[feature_name] = False
            st.session_state.shap_stale_fields.add(feature_name)
        except Exception as e:
            # Print errors during state transitions for debugging
            import traceback
            logger.exception(f"ERROR during state transition (on_value_change for {feature_name}):")

    def on_lock_change():
        try:
            # Store the lock state
            st.session_state.field_locks[feature_name] = st.session_state[f"lock_{feature_name}"]
        except Exception as e:
            # Print errors during state transitions for debugging
            import traceback
            logger.exception(f"ERROR during state transition (on_lock_change for {feature_name}):")

    col_input, col_hist = st.columns([1, 1])
    with col_input:
        # Lock checkbox and label with badge on same row
        col_lock, col_label = st.columns([0.15, 0.85])
        with col_lock:
            is_locked = st.session_state.field_locks.get(feature_name, False)
            st.checkbox(
                "🔒" if is_locked else "🔓",
                value=is_locked,
                key=f"lock_{feature_name}",
                help="Lock to prevent LLM from changing this value",
                on_change=on_lock_change
            )

        with col_label:
            # print("llm feature slider.")
            # print("get_field_indicator(feature_name)")    
            # print(get_field_indicator(feature_name))    
            st.markdown(f"**{display_name}** {get_field_indicator(feature_name)}", unsafe_allow_html=True)
            st.caption(help_text)

        # Slider with on_change callback
        is_locked = st.session_state.field_locks.get(feature_name, False)
        # Prefer the live widget-key value (manual edits), fall back to LLM features, then default.
        # Using the widget-key value as value= avoids the "default vs session state" conflict warning.
        current_value = float(st.session_state.get(f"input_{feature_name}",
                              st.session_state.features.get(feature_name, default_value)))
        # print(f"[SLIDER_DEBUG] render: input_{feature_name} = {current_value}  (session_state key={st.session_state.get(f'input_{feature_name}', '<unset>')}, features={st.session_state.features.get(feature_name, '<unset>')}, default={default_value})")

        value = st.slider(
            f"{display_name} slider",
            min_value=0.0,
            max_value=100.0,
            value=current_value,
            step=5.0,
            disabled=is_locked,
            key=f"input_{feature_name}",
            label_visibility="collapsed",
            on_change=on_slider_change
        )

        render_shap_annotation(shap_contribution, label=display_name)

    # Histogram
    with col_hist:
        if training_data_col and training_data is not None and training_data_col in training_data.columns:
            fig = render_histogram(display_name, training_data[training_data_col].dropna(), value,
                                  show_marker=st.session_state.show_distribution_markers, height=200)
            if fig:
                st.plotly_chart(fig, width='stretch', key=f"hist_{feature_name}")

    return value


def render_histogram(feature_name, train_data, user_value, show_marker=True, height=200, subtitle=None):
    """
    Render a histogram for a feature with optional 'You are here' marker.

    Args:
        feature_name: Display name for the feature
        train_data: Series or array of training data values
        user_value: Current user's value for this feature
        show_marker: Whether to show the 'You' marker
        height: Height of the chart in pixels
        subtitle: Optional grey label shown under the title
    """
    if len(train_data) == 0 or pd.isna(user_value):
        return None

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=train_data,
        name="median",
        marker_color="lightblue",
        opacity=0.7,
        nbinsx=20
    ))

    median_val = np.median(train_data)
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="grey",
        line_width=1.5,
        annotation_text="Median",
        annotation_position="top left",
        annotation_font_color="grey"
    )

    if show_marker:
        fig.add_vline(
            x=user_value,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text="You",
            annotation_position="top"
        )

    # Clean feature name for display
    display_name = feature_name.replace("_", " ").replace("sector cluster ", "").title()
    if len(display_name) > 30:
        display_name = display_name[:27] + "..."

    title_text = display_name
    if subtitle:
        title_text += f"<br><span style='font-size:10px;color:grey'>{subtitle}</span>"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13)),
        xaxis_title=display_name,
        xaxis_type="linear",
        yaxis_title="Count",
        height=height,
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=20)
    )

    return fig
