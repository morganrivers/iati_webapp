import logging
import traceback

import streamlit as st

from ui_components import render_histogram, render_shap_annotation
from targets_embedder import process_new_activity

from .common import EXTRACTED_PDF_DIR

logger = logging.getLogger(__name__)


def render_targets_embeddings_subsection(model_metadata: dict, training_data, _shap, start_date):
    # Params:
    #   model_metadata  — for train-median fallback values
    # Returns: (umap3_x, umap3_y, umap3_z, sector_distance, country_distance)
    # ---- REFACTOR → render_targets_embeddings_subsection(model_metadata)
    #         -> (umap3_x, umap3_y, umap3_z, sector_distance, country_distance) ----
    # ============================================================================
    # TARGETS EMBEDDINGS
    # ============================================================================

    st.markdown("---")
    st.markdown("### Targets Embeddings")
    st.markdown("Semantic similarity based on activity targets text.")

    # Initialize with session state embeddings or train medians
    if 'embedding_results' in st.session_state and st.session_state.embedding_results:
        umap3_x = float(st.session_state.embedding_results.get('umap3_x', model_metadata["train_medians"]["umap3_x"]))
        umap3_y = float(st.session_state.embedding_results.get('umap3_y', model_metadata["train_medians"]["umap3_y"]))
        umap3_z = float(st.session_state.embedding_results.get('umap3_z', model_metadata["train_medians"]["umap3_z"]))
        sector_distance = float(st.session_state.embedding_results.get('sector_distance', model_metadata["train_medians"]["sector_distance"]))
        country_distance = float(st.session_state.embedding_results.get('country_distance', model_metadata["train_medians"]["country_distance"]))
    else:
        umap3_x = float(model_metadata["train_medians"]["umap3_x"])
        umap3_y = float(model_metadata["train_medians"]["umap3_y"])
        umap3_z = float(model_metadata["train_medians"]["umap3_z"])
        sector_distance = float(model_metadata["train_medians"]["sector_distance"])
        country_distance = float(model_metadata["train_medians"]["country_distance"])

    # Initialize flag
    embedding_computed = False

    # Check if we have extraction results with features
    if (st.session_state.extraction_result is not None and
        st.session_state.extraction_result.get('features') and
        st.session_state.extraction_result.get('metadata')):

        # Get targets text from extracted features
        extracted_features = st.session_state.extraction_result.get('features', {})
        metadata_dict = st.session_state.extraction_result.get('metadata', {})
        activity_id = st.session_state.extraction_result.get('activity_id', '')

        targets_text = extracted_features.get('targets', '')

        # Add button to force recomputation
        if 'force_embedding_recompute' not in st.session_state:
            st.session_state.force_embedding_recompute = False

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Targets text available:** {len(targets_text.strip())} characters")
        with col2:
            if st.button(" 🛠️ Recompute Embeddings", help="Force recalculation of embeddings from targets text", width='stretch'):
                st.session_state.force_embedding_recompute = True
                st.session_state.embedding_results = {}  # Clear cached results
                st.rerun()

        # Check if we should compute (either sufficient text OR user forced it)
        should_compute = (targets_text and len(targets_text.strip()) > 100) or st.session_state.force_embedding_recompute

        if should_compute and targets_text:
            try:
                # Get necessary inputs from metadata
                dac_codes = set(metadata_dict.get('dac5_codes', []))
                countries_str = metadata_dict.get('recipient_iso3_fractions', '')

                # Allow manual override of DAC codes
                st.markdown("#### 🏷️ Sector Classification")

                if not dac_codes:
                    st.info("No DAC codes found in metadata. Sector will be auto-detected from embedding similarity.")

                # Store DAC codes in session state for editing
                if 'dac_codes_override' not in st.session_state:
                    st.session_state.dac_codes_override = None

                with st.expander("✏️ Override DAC Sector Codes (Optional)", expanded=False):
                    st.markdown("Enter 5-digit DAC sector codes separated by pipes (e.g., `23110|23181|41010`)")
                    st.markdown("[DAC Sector Codes Reference](https://reference.codeforiati.org/codelists/Sector/)")

                    dac_input = st.text_input(
                        "DAC5 Codes",
                        value="|".join(sorted(dac_codes)) if dac_codes else "",
                        help="5-digit codes like 23110 (Energy policy), 23181 (Energy education), 41010 (Environmental policy)"
                    )

                    if st.button("Update Sector Codes"):
                        if dac_input.strip():
                            # Parse input
                            new_codes = set()
                            for code in dac_input.split("|"):
                                code = code.strip()
                                if code and code.isdigit() and len(code) == 5:
                                    new_codes.add(code)

                            if new_codes:
                                st.session_state.dac_codes_override = new_codes
                                st.success(f"✓ Updated codes: {sorted(new_codes)}")
                                st.rerun()
                            else:
                                st.error("❌ Invalid format. Enter 5-digit codes separated by pipes.")
                        else:
                            st.session_state.dac_codes_override = None
                            st.info("Cleared override - will use auto-detection")
                            st.rerun()

                # Use override if available
                if st.session_state.dac_codes_override:
                    dac_codes = st.session_state.dac_codes_override
                    st.success(f"🔧 Using manual override: {sorted(dac_codes)}")

                emb_result = process_new_activity(
                    activity_text=targets_text,
                    dac_codes=dac_codes,
                    recipient_iso3_fractions=countries_str,
                    actual_start_date=metadata_dict.get('start_date') or start_date,
                    original_planned_start_date=None,
                    txn_first_date=None,
                    output_dir=EXTRACTED_PDF_DIR / activity_id
                )

                # Update variables and save to session state
                sector_distance = emb_result['sector_distance']
                country_distance = emb_result['country_distance']
                umap3_x = emb_result['umap3_x']
                umap3_y = emb_result['umap3_y']
                umap3_z = emb_result['umap3_z']

                # Save embedding results to session state for persistence
                st.session_state.embedding_results = {
                    'sector_distance': sector_distance,
                    'country_distance': country_distance,
                    'umap3_x': umap3_x,
                    'umap3_y': umap3_y,
                    'umap3_z': umap3_z,
                    'sector': emb_result.get('sector'),
                    'sector_auto_detected': emb_result.get('sector_auto_detected', False),
                    'sector_candidates': emb_result.get('sector_candidates', [])
                }

                # Show detected sector
                detected_sector = emb_result['sector']
                was_auto_detected = emb_result.get('sector_auto_detected', False)
                sector_candidates = emb_result.get('sector_candidates', [])

                if was_auto_detected:
                    st.warning(f"**Auto-detected sector**: {detected_sector} (inferred from embedding similarity)")

                    # Show top candidates with distances
                    if sector_candidates:
                        with st.expander("🔍 View all sector candidates (ranked by similarity)", expanded=False):
                            st.markdown("**Top candidates based on embedding distance:**")
                            for i, (candidate_sector, distance) in enumerate(sector_candidates, 1):
                                emoji = "" if i == 1 else ""
                                st.markdown(f"{emoji} **{i}. {candidate_sector}** — distance: {distance:.4f}")

                    st.info("💡 If this is incorrect, use the 'Override DAC Sector Codes' section above to manually specify sector codes.")
                else:
                    st.success(f"✓ **Detected sector**: {detected_sector} (from DAC codes)")

                # Display metrics with histograms
                st.markdown(f"**Sector Distance:** {sector_distance:.3f} &nbsp; **Country Distance:** {country_distance:.3f} &nbsp; **UMAP 3D:** ({umap3_x:.2f}, {umap3_y:.2f}, {umap3_z:.2f})", unsafe_allow_html=True)
                _hcol3, _hcol4, _hcol5 = st.columns(3)

                with _hcol3:
                    _fig = render_histogram("umap3_x", training_data["umap3_x"].dropna(), umap3_x, height=180, subtitle="Low: forestry/water · High: energy/financing")
                    if _fig:
                        st.plotly_chart(_fig, width="stretch")
                    render_shap_annotation(_shap('umap3_x'), label="Semantic similarity (UMAP x-axis)")
                with _hcol4:
                    _fig = render_histogram("umap3_y", training_data["umap3_y"].dropna(), umap3_y, height=180, subtitle="Low: energy · High: biodiversity/conservation")
                    if _fig:
                        st.plotly_chart(_fig, width="stretch")
                    render_shap_annotation(_shap('umap3_y'), label="Semantic similarity (UMAP y-axis)")
                with _hcol5:
                    _fig = render_histogram("umap3_z", training_data["umap3_z"].dropna(), umap3_z, height=180, subtitle="Low: rural/wildlife · High: urban/sanitation")
                    if _fig:
                        st.plotly_chart(_fig, width="stretch")
                    render_shap_annotation(_shap('umap3_z'), label="Semantic similarity (UMAP z-axis)")

                _hcol1, _hcol2 = st.columns(2)
                with _hcol1:
                    _fig = render_histogram("sector_distance", training_data["sector_distance"].dropna(), sector_distance, height=180, subtitle="Higher is a more customized, tailored project objective (in its sector)")
                    if _fig:
                        st.plotly_chart(_fig, width="stretch")
                    render_shap_annotation(_shap('sector_distance'), label="Sector distance (semantic similarity to successful activities)")
                with _hcol2:
                    _fig = render_histogram("country_distance", training_data["country_distance"].dropna(), country_distance, height=180, subtitle="Higher is a more customized, tailored project objective (for these countries)")
                    if _fig:
                        st.plotly_chart(_fig, width="stretch")
                    render_shap_annotation(_shap('country_distance'), label="Country distance (semantic similarity to successful activities in same country)")
                # Show success message with warning if text was short
                if len(targets_text.strip()) < 100:
                    st.success("✅ Embeddings computed from your targets text")
                    st.warning(f"⚠️ Note: Short targets text ({len(targets_text.strip())} chars). Embeddings may be less accurate.")
                else:
                    st.success("✅ Embeddings computed from your targets text")

                embedding_computed = True

                # Reset force flag after successful computation
                st.session_state.force_embedding_recompute = False

            except Exception as e:
                full_traceback = traceback.format_exc()

                # Print to stdout (terminal) for debugging
                logger.exception("ERROR IN TARGETS EMBEDDINGS:")

                # Show in UI
                st.error(f"❌ Error computing embeddings: {e}")
                st.warning("⚠️ Using database medians")
                with st.expander("🔍 Show full error details", expanded=True):
                    st.code(full_traceback)
                embedding_computed = False

                # Reset force flag after error
                st.session_state.force_embedding_recompute = False
        else:
            # Not enough text and user didn't force computation
            if len(targets_text.strip()) < 100:
                st.warning(f"⚠️ Insufficient targets text ({len(targets_text.strip())} chars, need >100). Using database medians for embedding features.")
                st.info("💡 Click 'Recompute Embeddings' above to force computation with short text, or extract more detailed targets from your PDF.")
            else:
                st.info("⚠️ No targets text available. Using database medians for embedding features.")

            # Reset force flag
            st.session_state.force_embedding_recompute = False
    else:
        st.info("⚠️ No extraction data available. Upload a PDF and extract features first. Using database medians for embedding features.")

    return umap3_x, umap3_y, umap3_z, sector_distance, country_distance
