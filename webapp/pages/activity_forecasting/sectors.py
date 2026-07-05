import streamlit as st
import plotly.graph_objects as go

from ui_components import get_field_indicator, render_shap_annotation


def render_sector_allocation_subsection(model_metadata: dict, training_data, _shap, _shap_sum):
    # Params:
    #   model_metadata  — for train-median sector-cluster defaults
    #   training_data   — for parallel-coordinates chart data
    # Returns: sector_percentages (dict[str, float])
    # ============================================================================
    # SECTOR ALLOCATION
    # ============================================================================
    st.subheader("Sector Allocation")
    st.markdown("Allocate expenditure percentages across sector clusters (must sum to 100%)")

    # Define all 16 sector clusters
    SECTOR_CLUSTERS = model_metadata['sector_clusters']

    # Initialize session state for sector percentages using training medians
    if 'sector_percentages' not in st.session_state:
        st.session_state.sector_percentages = {}
    # Backfill any clusters missing (e.g. loaded from older saved state)
    for cluster in SECTOR_CLUSTERS:
        if cluster not in st.session_state.sector_percentages:
            feature_name = f"sector_cluster_{cluster}"
            if feature_name in model_metadata['train_medians']:
                st.session_state.sector_percentages[cluster] = model_metadata['train_medians'][feature_name] * 100.0
            else:
                st.session_state.sector_percentages[cluster] = 0.0
    # Initialise widget keys from sector_percentages if not yet set.
    # This lets normalize/reset write directly to the widget key without
    # conflicting with a value= parameter.
    for cluster in SECTOR_CLUSTERS:
        wk = f"input_sector_{cluster}"
        if wk not in st.session_state:
            st.session_state[wk] = st.session_state.sector_percentages[cluster]

    # Lock/Unlock buttons for sectors
    def _lock_all_sectors_cb():
        for cluster in SECTOR_CLUSTERS:
            st.session_state.field_locks[f"sector_{cluster}"] = True
            if f"lock_sector_{cluster}" in st.session_state:
                st.session_state[f"lock_sector_{cluster}"] = True

    def _unlock_all_sectors_cb():
        for cluster in SECTOR_CLUSTERS:
            st.session_state.field_locks[f"sector_{cluster}"] = False
            if f"lock_sector_{cluster}" in st.session_state:
                st.session_state[f"lock_sector_{cluster}"] = False

    col_lock1, col_lock2, col_lock3 = st.columns([1, 1, 3])
    with col_lock1:
        st.button("🔒 Lock All Sectors", key="lock_all_sectors", on_click=_lock_all_sectors_cb)
    with col_lock2:
        st.button("🔓 Unlock All Sectors", key="unlock_all_sectors", on_click=_unlock_all_sectors_cb)
    with col_lock3:
        st.info("💡 Locked sectors won't be overwritten during extraction")

    # Create columns for sector inputs
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    sector_percentages = {}
    for idx, cluster in enumerate(SECTOR_CLUSTERS):
        with columns[idx % 3]:
            field_name = f"sector_{cluster}"

            # Clean name for display
            display_name = cluster.replace("_", " ").title()
            if len(display_name) > 35:
                display_name = display_name[:32] + "..."

            st.markdown(f"<b>{display_name}</b> {get_field_indicator(field_name)}", unsafe_allow_html=True)

            # Input row - lock, checkbox, help, input
            col_lock, col_input = st.columns([0.25, 0.75])

            # Define on_change callback to track editing
            def on_sector_change(cluster_name=cluster, field_n=field_name):
                def _cb():
                    st.session_state.field_edited.update({field_n: True})
                    st.session_state.shap_stale_fields.add(f'sector_cluster_{cluster_name}')
                return _cb

            with col_lock:
                def on_sector_lock_change(fn=field_name):
                    st.session_state.field_locks[fn] = st.session_state[f"lock_{fn}"]

                st.checkbox(
                    "🔒" if st.session_state.field_locks.get(field_name, False) else "🔓",
                    value=st.session_state.field_locks.get(field_name, False),
                    key=f"lock_{field_name}",
                    help="Lock to prevent LLM from changing this value",
                    on_change=on_sector_lock_change
                )

            with col_input:
                sector_percentages[cluster] = st.number_input(
                    display_name,
                    min_value=0.0,
                    max_value=100.0,
                    step=5.0,
                    key=f"input_{field_name}",
                    help=f"Percentage of expenditure for {cluster.replace('_', ' ')}",
                    label_visibility="collapsed",
                    disabled=st.session_state.field_locks.get(field_name, False),
                    on_change=on_sector_change(cluster, field_name)
                )
                render_shap_annotation(_shap(f"sector_cluster_{cluster}"), label=f"Sector: {cluster.replace('_', ' ')}")

    # Update session state
    st.session_state.sector_percentages = sector_percentages

    # Sector SHAP total
    _sector_shap_keys = [f"sector_cluster_{c}" for c in SECTOR_CLUSTERS] + ['sector_clusters_missing']
    _sector_shap_total = _shap_sum(*_sector_shap_keys)
    if _sector_shap_total is not None:
        st.markdown("**Total sector allocation contribution:**")
        render_shap_annotation(_sector_shap_total, label="All sectors (combined contribution)")

    # Calculate total and show warning if not 100%
    total_percentage = sum(sector_percentages.values())
    if abs(total_percentage - 100.0) < 0.01:
        st.success(f"✅ Total: {total_percentage:.1f}% (Perfect!)")
    elif total_percentage > 100.0:
        st.error(f"⚠️ Total: {total_percentage:.1f}% (Exceeds 100%! Please reduce.)")
    else:
        st.warning(f"⚠️ Total: {total_percentage:.1f}% (Should sum to 100%)")

    # Normalize and reset buttons
    def normalize_sectors():
        """Normalize sector percentages to sum to 100%"""
        # Get current values from widgets (they're in session_state with keys input_sector_{cluster})
        current_values = {}
        for cluster in SECTOR_CLUSTERS:
            widget_key = f"input_sector_{cluster}"
            if widget_key in st.session_state:
                current_values[cluster] = st.session_state[widget_key]
            else:
                current_values[cluster] = st.session_state.sector_percentages.get(cluster, 0.0)

        total = sum(current_values.values())
        if total > 0:
            normalized = {k: (v / total) * 100.0 for k, v in current_values.items()}
        else:
            num_sectors = len(SECTOR_CLUSTERS)
            normalized = {k: 100.0 / num_sectors for k in SECTOR_CLUSTERS}

        st.session_state.sector_percentages = normalized
        for cluster, value in normalized.items():
            st.session_state[f"input_sector_{cluster}"] = value
            st.session_state.field_edited[f"sector_{cluster}"] = True

    def reset_sectors_to_zero():
        """Reset all sector percentages to 0"""
        zero_dict = {k: 0.0 for k in SECTOR_CLUSTERS}
        st.session_state.sector_percentages = zero_dict
        for cluster in SECTOR_CLUSTERS:
            st.session_state[f"input_sector_{cluster}"] = 0.0
            st.session_state.field_edited[f"sector_{cluster}"] = True

    col_norm, col_reset = st.columns(2)
    with col_norm:
        st.button("Normalize to 100%", on_click=normalize_sectors, key="normalize_sectors_btn", use_container_width=True)
    with col_reset:
        st.button("Set all to 0", on_click=reset_sectors_to_zero, key="reset_sectors_btn", use_container_width=True)

    # Sector allocation visualization
    st.markdown("---")
    st.markdown("### Sector Allocation Mix")
    st.markdown("Compare your activity's spending allocation across sectors to the training portfolio.")

    # Get sector cluster columns
    sector_cols = [f for f in model_metadata['feature_names'] if f.startswith('sector_cluster_')]

    if sector_cols and all(col in training_data.columns for col in sector_cols):
        # Clean sector names for display
        sector_names = [col.replace('sector_cluster_', '').replace('_', ' ').title()
                       for col in sector_cols]

        # Get user's sector values (convert from percentage to proportion 0-1)
        user_sectors = [sector_percentages.get(col.replace('sector_cluster_', ''), 0.0) / 100.0
                       for col in sector_cols]

        # Sample training data (limit to 100 projects for performance)
        sample_size = min(100, len(training_data))
        train_sample = training_data[sector_cols].sample(n=sample_size, random_state=42)

        # Convert to percentages for display
        train_sample_pct = train_sample * 100
        user_sectors_pct_list = [v * 100 for v in user_sectors]

        # Build dimensions for parallel coordinates with fixed 0-100 range
        dimensions = []
        for i, (col, name) in enumerate(zip(sector_cols, sector_names)):
            # Wrap long names by adding line breaks
            words = name.split()
            wrapped_label = ""
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 15:  # Max 15 chars per line
                    current_line += (" " if current_line else "") + word
                else:
                    wrapped_label += current_line + "<br>"
                    current_line = word
            wrapped_label += current_line

            dim = dict(
                label=wrapped_label,
                values=list(train_sample_pct[col]) + [user_sectors_pct_list[i]],
                range=[0, 100],
            )
            # # Only show tick labels on the leftmost axis
            # if i == 0:
            #     dim['tickvals'] = [0, 20, 40, 60, 80, 100]
            # else:
            #     dim['tickvals'] = []
            if i == 0:
                dim['tickvals'] = [0, 20, 40, 60, 80, 100]
                dim['ticktext'] = ['0', '20', '40', '60', '80', '100']
            else:
                dim['tickvals'] = [0, 20, 40, 60, 80, 100]
                dim['ticktext'] = [''] * 6

            dimensions.append(dim)

        # Create color array: 0 for training (gray), 1 for user (red)
        color_array = [0] * sample_size + [1]

        # Create parallel coordinates using graph_objects
        fig_parallel = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=color_array,
                    colorscale=[[0, 'rgba(211, 211, 211, 0.8)'], [1, 'orangered']],
                    showscale=False
                ),
                dimensions=dimensions
            )
        )

        fig_parallel.update_layout(
            height=500,
            margin=dict(l=60, r=40, t=140, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#333333'),
        )

        st.plotly_chart(fig_parallel, width='stretch')
        st.info("💡 **Parallel coordinates** shows the full distribution of sector mixes across training activities (your activity in red).")

    return sector_percentages
