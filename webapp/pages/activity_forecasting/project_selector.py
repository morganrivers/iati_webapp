import logging
from datetime import datetime

import streamlit as st

from project_manager import (get_available_projects, load_project_name,
                             load_project_state, save_project_state,
                             create_new_project, delete_project, get_state_hash)
from state_manager import clear_project_state

from .common import EXTRACTED_PDF_DIR

logger = logging.getLogger(__name__)


def render_project_selector(_llm_running: bool) -> None:
    # Params:
    #   _llm_running  — disables save/name-edit widgets while background work runs
    # Returns: None  (all changes written to st.session_state)
    
    st.markdown("---")
    st.markdown("### Activity Selection")

    col_select, col_name = st.columns([1, 1], gap="large")

    with col_select:
        # Get available projects and create friendly name mapping
        available_folders = get_available_projects()

        # Create mapping: friendly_name -> folder_name
        name_to_folder = {}
        folder_to_name = {}
        name_counts = {}  # Track how many times each name appears

        for folder in available_folders:
            friendly_name = load_project_name(folder)

            # Handle duplicate names by appending folder ID
            original_name = friendly_name
            if friendly_name in name_to_folder:
                # Duplicate detected! Append folder suffix to both
                # First, update the existing entry
                existing_folder = name_to_folder[friendly_name]
                suffix = existing_folder.split('_')[-1][:8]  # First 8 chars of folder ID
                unique_existing_name = f"{friendly_name} ({suffix})"
                del name_to_folder[friendly_name]
                name_to_folder[unique_existing_name] = existing_folder
                folder_to_name[existing_folder] = unique_existing_name

                # Now add the new one with its suffix
                new_suffix = folder.split('_')[-1][:8]
                friendly_name = f"{original_name} ({new_suffix})"

            name_to_folder[friendly_name] = folder
            folder_to_name[folder] = friendly_name

        # Create dropdown options with friendly names
        friendly_names = list(name_to_folder.keys())
        project_options = ["➕ Create New Activity"] + friendly_names

        # Option that reflects the currently-selected project. Used to keep the
        # (keyed) selectbox in sync with *programmatic* changes to
        # selected_project_folder (create / save / delete).
        if st.session_state.selected_project_folder:
            _current_option = folder_to_name.get(st.session_state.selected_project_folder) or "➕ Create New Activity"
        else:
            _current_option = "➕ Create New Activity"
        if _current_option not in project_options:
            _current_option = "➕ Create New Activity"

        # Push the current selection into the widget's session_state value BEFORE
        # instantiating it. After a *user* change this is a no-op (the on_change
        # callback has already updated selected_project_folder to match), so it
        # never fights the user. A dynamic `index=` on a keyless selectbox was the
        # cause of the selection "popping back" and needing multiple clicks.
        if st.session_state.get("project_selector") != _current_option:
            st.session_state["project_selector"] = _current_option

        def _on_project_select():
            sel = st.session_state.project_selector
            if sel == "➕ Create New Activity":
                st.session_state.creating_new_project = True
                if not st.session_state.pending_project_name:
                    st.session_state.pending_project_name = f"Untitled - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                if st.session_state.selected_project_folder is not None:
                    logger.info(f">>> USER SELECTED: Create New Project (was: {st.session_state.selected_project_folder})")
                    clear_project_state()
                    st.session_state.selected_project_folder = None
                    st.session_state.project_name = None
                return
            st.session_state.creating_new_project = False
            st.session_state.pending_project_name = None
            folder = name_to_folder.get(sel)
            if folder and st.session_state.selected_project_folder != folder:
                clear_project_state()
                st.session_state.selected_project_folder = folder
                st.session_state.project_name = sel
                load_project_state(folder)

        selected_display = st.selectbox(
            "Select or create an activity:",
            project_options,
            key="project_selector",
            on_change=_on_project_select,
            help="Choose an existing activity to continue working on it, or create a new one"
        )

        # Creation mode must follow the *current* selection, not only the
        # on_change event. "➕ Create New Activity" is the default option, so on
        # first load selecting it fires no change and the callback never runs;
        # deriving the flag here makes the name input + Save button appear.
        if selected_display == "➕ Create New Activity" and not st.session_state.selected_project_folder:
            if not st.session_state.creating_new_project:
                st.session_state.creating_new_project = True
            if not st.session_state.pending_project_name:
                st.session_state.pending_project_name = f"Untitled - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # Navigating back to this page can drop widget keys while the project is
        # still selected; restore them.
        if selected_display != "➕ Create New Activity":
            _folder = name_to_folder.get(selected_display)
            if _folder and 'input_planned_expenditure' not in st.session_state:
                logger.info(">>> Widget keys missing - reloading state to restore widget values")
                load_project_state(_folder)

    with col_name:
        if st.session_state.selected_project_folder:
            # Show editable project name
            if st.session_state.project_name is None:
                st.session_state.project_name = load_project_name(st.session_state.selected_project_folder)

            # Source badge
            _name_source = st.session_state.get('project_name_source')
            if _name_source == 'llm':
                _name_badge = '<span style="background-color: #cce5ff; color: #004085; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; margin-left: 5px;">⚛︎ LLM</span>'
            elif _name_source == 'human':
                _name_badge = '<span style="background-color: #d4edda; color: #155724; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; margin-left: 5px;">✏️ Edited</span>'
            else:
                _name_badge = ''

            st.markdown(f"<b>Activity Name</b> {_name_badge}", unsafe_allow_html=True)

            # Input row - lock, checkbox, help, input
            col_lock, col_input = st.columns([0.2, 0.8])

            with col_lock:
                def _on_name_lock_change():
                    st.session_state.field_locks['project_name'] = st.session_state['lock_project_name']
                _lock_icon = "🔒" if st.session_state.field_locks.get('project_name', False) else "🔓"

                st.checkbox(
                    _lock_icon,
                    value=st.session_state.field_locks.get('project_name', False),
                    key="lock_project_name",
                    help="Lock to prevent LLM from overwriting the activity name",
                    on_change=_on_name_lock_change,
                )
            with col_input:
                new_name = st.text_input(
                    "Activity Name",
                    value=st.session_state.project_name,
                    help="Edit the activity name (auto-saves on change)",
                    key=f"project_name_input_{st.session_state.selected_project_folder}",
                    disabled=st.session_state.field_locks.get('project_name', False) or _llm_running,
                    label_visibility="collapsed"
                )

            if new_name != st.session_state.project_name:
                # st.session_state.project_name = new_name
                # st.session_state.project_name_source = 'human'
                # save_project_name(st.session_state.selected_project_folder, new_name)
                # st.success("✓ Activity name saved")
                st.session_state.project_name = new_name
                st.session_state.project_name_source = 'human'
                st.session_state.has_unsaved_changes = True

        elif st.session_state.creating_new_project:
            new_pending = st.text_input(
                "Activity Name:",
                value=st.session_state.pending_project_name or "",
                help="Name for the new activity"
            )
            if new_pending != st.session_state.pending_project_name:
                st.session_state.pending_project_name = new_pending
        else:
            st.info("👆 Select or create an activity to begin")

    # Add "Create Project" button when in creation mode
    if st.session_state.creating_new_project:
        col_create, col_spacer = st.columns([1, 3])
        with col_create:
            if st.button("Save New Activity", type="primary", width='stretch', disabled=_llm_running):
                if not st.session_state.pending_project_name or st.session_state.pending_project_name.strip() == "":
                    st.error("❌ Please enter an activity name")
                else:
                    # Create the new project
                    folder_name = create_new_project(st.session_state.pending_project_name)
                    if folder_name:
                        # Set it as the selected project
                        st.session_state.selected_project_folder = folder_name
                        st.session_state.project_name = st.session_state.pending_project_name
                        st.session_state.creating_new_project = False
                        st.session_state.pending_project_name = None
                        # CRITICAL: Save any widget values that were entered before creating project
                        save_project_state(folder_name)
                        # FIX: Load the state back immediately so widget keys are restored before rerun
                        load_project_state(folder_name)
                        st.success(f"✓ Activity '{st.session_state.project_name}' created successfully!")
                        st.rerun()

    # Show project info if one is selected
    if st.session_state.selected_project_folder:
        col_info, col_delete, col_save, col_pdf = st.columns([2.5, 1, 1, 1])
        with col_info:
            # Check for unsaved changes by comparing hash of current state vs hash at last save/load
            has_unsaved = get_state_hash() != st.session_state.get('saved_state_hash', None)

            status_emoji = "⚠️" if has_unsaved else "✓"
            status_text = "UNSAVED CHANGES" if has_unsaved else "Working on"
            status_color = "orange" if has_unsaved else "green"

            st.markdown(f":{status_color}[{status_emoji} {status_text}: **{st.session_state.project_name}**]")

            # Show last save time if available
            last_saved = st.session_state.get('last_saved_timestamp', '')
            if last_saved:
                try:
                    saved_dt = datetime.fromisoformat(last_saved)
                    st.caption(f"Last saved: {saved_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception:
                    pass

        with col_delete:
            st.write("")  # vertical spacer to align with status text
            if st.button("🗑️ Delete Activity", type="secondary", width='stretch'):
                st.session_state.confirm_delete = True

        with col_save:
            st.write("")  # vertical spacer to align with status text
            if st.button("Save Activity State", type="primary", width='stretch', disabled=_llm_running):
                logger.info("DEBUG SAVE BUTTON CLICKED")
                logger.info(f"Time: {datetime.now().isoformat()}")
                logger.info(f"Project: {st.session_state.selected_project_folder}")
                logger.info(f"st.session_state.get('input_planned_expenditure'): {st.session_state.get('input_planned_expenditure', 'NOT_IN_SESSION_STATE')}")
                logger.info(f"st.session_state.extracted_values.get('planned_expenditure'): {st.session_state.extracted_values.get('planned_expenditure', 'NOT_SET')}")
                logger.info(f"st.session_state.field_edited.get('planned_expenditure'): {st.session_state.field_edited.get('planned_expenditure', False)}")

                if save_project_state(st.session_state.selected_project_folder):
                    # FIX: Load the state back immediately so widget keys are restored before rerun
                    load_project_state(st.session_state.selected_project_folder)
                    st.success("✓ Saved!")
                    st.rerun()

        with col_pdf:
            st.write("")  # vertical spacer to align with status text
            pdf_path = EXTRACTED_PDF_DIR / st.session_state.selected_project_folder / "uploaded.pdf"
            if pdf_path.exists():
                with open(pdf_path, "rb") as _f:
                    st.download_button(
                        "📄 Download PDF",
                        data=_f.read(),
                        file_name=f"{st.session_state.project_name or 'activity'}.pdf",
                        mime="application/pdf",
                        width='stretch',
                    )

        # Confirmation dialog for delete
        if st.session_state.get('confirm_delete', False):
            st.warning(f"⚠️ Are you sure you want to delete **{st.session_state.project_name}**? This cannot be undone!")
            col_yes, col_no, col_space = st.columns([1, 1, 3])
            with col_yes:
                if st.button("Yes, delete it", type="primary", width='stretch'):
                    folder_to_delete = st.session_state.selected_project_folder
                    project_name_deleted = st.session_state.project_name
                    if delete_project(folder_to_delete):
                        st.session_state.selected_project_folder = None
                        st.session_state.project_name = None
                        st.session_state.confirm_delete = False
                        for _key in ['feature_table', 'field_edited', 'features', 'extracted_values',
                                     'embedding_results', 'sector_percentages', 'location_countries',
                                     'prediction', 'base', 'ens_delta',
                                     'feature_vector', 'feature_vector_imputed', 'shap_result',
                                     'prediction_explanation', 'extraction_result', 'tag_predictions']:
                            st.session_state.pop(_key, None)
                        st.success(f"✅ Activity '{project_name_deleted}' deleted successfully")
                        st.rerun()
                    else:
                        st.session_state.confirm_delete = False
            with col_no:
                if st.button("Cancel", width='stretch'):
                    st.session_state.confirm_delete = False
                    st.rerun()

        st.info("💡 **Auto-save:** Changes are temporarily saved when switching pages (preserved during navigation). Click **Save Activity State** to permanently save your work (persists after browser refresh).")
    else:
        st.warning("⚠️ No activity selected. Create a new activity or select an existing one to continue.")
