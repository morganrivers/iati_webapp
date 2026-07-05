import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_extracted_data_page():
    st.title("View Extracted Data")
    st.markdown("""
    This page shows the detailed extraction results from the PDF processing pipeline.
    Upload and extract a PDF first to see results here.
    """)

    already_showed_activity_warning = False

    # Show selected project info
    if st.session_state.selected_project_folder:
        st.info(f"**Current Activity:** {st.session_state.project_name}") #(ID: `{st.session_state.selected_project_folder}`)")
    else:
        already_showed_activity_warning = True

        st.warning("⚠️ No activity selected. Go to the Activity Forecasting page to select or create an activity.")

    if "extraction_result" not in st.session_state or st.session_state.extraction_result is None:
        if not already_showed_activity_warning:
            st.warning("⚠️ No extraction data available. Please upload and process a PDF first on the main page.")
    else:
        result = st.session_state.extraction_result
        status = result.get('status', 'unknown')

        if status == 'loading':
            st.info(f"⏳ Processing in progress... Activity ID: **{result.get('activity_id', 'N/A')}**")
            st.markdown("**Showing data loaded so far:**")
        elif status == 'complete':
            st.success(f"✅ Viewing extraction for Activity ID: **{result['activity_id']}**")
            if 'output_dir' in result:
                st.markdown(f"📁 **Output directory:** `{result['output_dir']}`")
        else:
            st.warning(f"⚠️ Status: {status}")

        # -------------------------------------------------------------------------
        # Phase 0: Metadata
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📋 Phase 0: Metadata")

        metadata = result.get('metadata', {})
        if metadata:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Title:**")
                st.info(metadata.get('title', 'N/A'))

                st.markdown("**Locations:**")
                st.info(metadata.get('country_location', 'N/A'))

            with col2:
                st.markdown("**Participating Organizations:**")
                st.info(metadata.get('participating_orgs', 'N/A'))

                st.markdown("**Dates:**")
                st.info(f"Start: {metadata.get('planned_start_date', 'N/A')}\nEnd: {metadata.get('planned_end_date', 'N/A')}")

            with st.expander("View raw metadata JSON"):
                st.json(metadata)
        else:
            st.info("⏳ Metadata not yet loaded...")

        # -------------------------------------------------------------------------
        # Phase 1: Page Categories
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Phase 1: Page Categorization")

        page_categories = result.get('page_categories', [])
        if page_categories:
            st.markdown(f"**Total pages:** {len(page_categories)}")
            # Summary statistics
            col1, col2 = st.columns(2)
            with col1:
                avg_score = sum(p.get('score', 0) for p in page_categories) / len(page_categories)
                st.metric("Average Score", f"{avg_score:.1f}/10")
            with col2:
                high_score_pages = sum(1 for p in page_categories if p.get('score', 0) >= 7)
                st.metric("High-Score Pages", f"{high_score_pages} (≥7)")

            # Page table
            with st.expander("📊 View all page categories"):
                page_df = pd.DataFrame([
                    {
                        'Page': p.get('page_start', 'N/A'),
                        'Section': p.get('section', 'N/A'),
                        'Subcategory A': p.get('subcategory_a', 'N/A'),
                        'Subcategory B': p.get('subcategory_b', 'N/A'),
                        'Score': p.get('score', 0),
                    }
                    for p in page_categories
                ])
                st.dataframe(page_df, width='stretch', hide_index=True)
        else:
            st.info("⏳ Page categories not yet loaded...")

        # -------------------------------------------------------------------------
        # Phase 2: Summary
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Phase 2: Activity Summary")

        summary = result.get('summary', '')
        if summary and summary != '':
            st.markdown(f"**Length:** {len(summary)} characters")
            st.markdown("**Description:**")
            st.text_area("Activity Description", summary, height=1000, disabled=True)
        else:
            st.info("⏳ Summary not yet loaded...")

        # -------------------------------------------------------------------------
        # Phase 3: Finance
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("💰 Phase 3: Finance Breakdown")

        finance = result.get('finance', {})
        if finance and len(finance) > 0:
            total_alloc = finance.get('total_allocation', {})
            st.markdown(f"**Total Allocation:** {total_alloc.get('amount', 'N/A')} {total_alloc.get('currency', '')}")

            allocations = finance.get('quantitative_outcome_allocations', [])
            if allocations:
                st.markdown(f"**Sector Allocations:** {len(allocations)} entries")

                with st.expander("📊 View sector breakdown"):
                    alloc_df = pd.DataFrame([
                        {
                            'Outcome': a.get('outcome', 'N/A'),
                            'Custom': a.get('custom_outcome', ''),
                            'Type': a.get('grant_or_loan', 'N/A'),
                            'Amount': a.get('amount_allocated', 0),
                            'Currency': a.get('currency', 'N/A'),
                        }
                        for a in allocations
                    ])
                    st.dataframe(alloc_df, width='stretch', hide_index=True)

            with st.expander("View raw finance JSON"):
                st.json(finance)
        else:
            st.info("⏳ Finance data not yet loaded...")

        # -------------------------------------------------------------------------
        # Phase 4: Features
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔍 Phase 4: Baseline Features")

        features = result.get('features', {})
        if features and len(features) > 0:
            for feature_name, feature_text in features.items():
                with st.expander(f"{feature_name.replace('_', ' ').title()} ({len(feature_text)} chars)"):
                    if feature_text == "NO RESPONSE":
                        st.warning("No information found in PDF")
                    else:
                        st.text_area(f"{feature_name}_text", feature_text, height=750, disabled=True, label_visibility="collapsed")
        else:
            st.info("⏳ Features not yet loaded...")

        # -------------------------------------------------------------------------
        # Embedding Results
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Embedding Results")

        embedding_results = st.session_state.get('embedding_results', {})
        if embedding_results:
            st.markdown("**UMAP Coordinates:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("UMAP X", f"{embedding_results.get('umap3_x', 0):.4f}")
            with col2:
                st.metric("UMAP Y", f"{embedding_results.get('umap3_y', 0):.4f}")
            with col3:
                st.metric("UMAP Z", f"{embedding_results.get('umap3_z', 0):.4f}")

            st.markdown("**Distances:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sector Distance", f"{embedding_results.get('sector_distance', 0):.4f}",
                         help=f"Distance to {embedding_results.get('sector', 'N/A')} centroid")
            with col2:
                st.metric("Country Distance", f"{embedding_results.get('country_distance', 0):.4f}")

            if embedding_results.get('sector'):
                sector_info = f"**Sector:** {embedding_results['sector']}"
                if embedding_results.get('sector_auto_detected'):
                    sector_info += " (auto-detected)"
                st.info(sector_info)

            with st.expander("View raw embedding JSON"):
                st.json(embedding_results)
        else:
            st.info("⏳ No embedding results yet. Embeddings are computed from targets text.")

        # -------------------------------------------------------------------------
        # Phase 5: Grades
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Phase 5: Feature Grades")

        grades = st.session_state.get('feature_grades', {})
        if grades and len(grades) > 0:
            st.markdown("**LLM-generated grades (0-100 scale):**")

            # Display in a nice table format
            grade_data = []
            for feature_name, grade_value in sorted(grades.items()):
                grade_data.append({
                    'Feature': feature_name.replace('_', ' ').title(),
                    'Grade': f"{grade_value:.1f}",
                    'Status': '✅ Applied' if not st.session_state.field_locks.get(feature_name, False) else '🔒 Locked'
                })

            grade_df = pd.DataFrame(grade_data)
            st.dataframe(grade_df, width='stretch', hide_index=True)

            # Show distribution
            with st.expander("📊 View grade distribution"):
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[g['Feature'] for g in grade_data],
                    y=[float(g['Grade']) for g in grade_data],
                    marker_color='lightblue'
                ))
                fig.update_layout(
                    xaxis_title="Feature",
                    yaxis_title="Grade (0-100)",
                    height=400,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("⏳ Grades not yet generated. Click 'Confirm and Extract Feature Grades' on the main page.")
