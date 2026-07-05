import streamlit as st
import json
import hashlib
import traceback
from pathlib import Path
from datetime import datetime
import secrets

# Import state management utilities
from state_manager import validate_and_sync_field_edited, clear_project_state
from debug_utils import _loc_debug, _gdp_debug

import logging

logger = logging.getLogger(__name__)

from webapp_paths import EXTRACTED_PDF_DIR, TRAIN_MEDIANS_PATH


def get_state_hash() -> str:
    """Hash the current saveable state for unsaved-changes detection. No disk I/O."""
    relevant = {
        'project_name': st.session_state.get('project_name'),
        'location_countries': st.session_state.get('location_countries', []),
        'sector_percentages': st.session_state.get('sector_percentages', {}),
        'extracted_values': st.session_state.get('extracted_values', {}),
        'field_edited': st.session_state.get('field_edited', {}),
        **{k: st.session_state[k] for k in st.session_state
           if k.startswith(('input_', 'select_', 'lock_')) and not k.startswith('lock_all_')}
    }
    return hashlib.md5(json.dumps(relevant, default=str, sort_keys=True).encode()).hexdigest()


def create_new_project(project_name: str) -> str:
    """Create a new project folder and save the project name

    Args:
        project_name: Display name for the project

    Returns:
        The folder name of the created project (e.g., 'webapp_abc123')
    """
    # Generate unique folder name
    folder_name = f"webapp_{secrets.token_hex(6)}"
    project_path = EXTRACTED_PDF_DIR / folder_name

    try:
        # Create project directory
        project_path.mkdir(parents=True, exist_ok=True)

        # Save project name
        name_file = project_path / "project_name.txt"
        with open(name_file, 'w') as f:
            f.write(project_name)

        # Initialize empty app state
        state_file = project_path / "app_state.json"
        initial_state = {
            'timestamp': datetime.now().isoformat(),
            'extracted_values': {},
            'field_edited': {},
            'field_locks': {},
            'features': {},
            'sector_percentages': {},
            'location_countries': [],
            'phases_0_3_complete': False,
            'ready_for_phase_4': False,
            'confirmed_metadata': {},
            'embedding_results': {},
            'widget_state': {},
            'feature_table': None,
        }
        with open(state_file, 'w') as f:
            json.dump(initial_state, f, indent=2)

        logger.info(f"Created new project: {project_name} {folder_name}")
        return folder_name
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.exception("ERROR creating project:")
        st.error(f"Error creating project: {str(e)}")
        return None


def get_available_projects():
    """Get list of available project folders from extracted_pdf_data/"""
    data_dir = EXTRACTED_PDF_DIR
    if not data_dir.exists():
        return []

    all_items = list(data_dir.iterdir())
    projects = [p.name for p in all_items if p.is_dir()]

    # Sort by modification time (most recent first)
    projects.sort(key=lambda p: (data_dir / p).stat().st_mtime, reverse=True)
    return projects


def delete_project(project_folder: str):
    """Delete a project folder and all its contents

    Args:
        project_folder: Name of the project folder to delete

    Returns:
        True if successful, False otherwise
    """
    import shutil

    project_path = EXTRACTED_PDF_DIR / project_folder
    if not project_path.exists():
        st.warning(f"Project folder not found: {project_folder}")
        return False

    try:
        logger.info(f"DELETING PROJECT: {project_folder}")
        logger.info(f"Path: {project_path}")

        # Delete the entire folder
        shutil.rmtree(project_path)

        logger.info("Project deleted successfully")
        return True
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.exception("ERROR deleting project:")
        st.error(f"Error deleting project: {str(e)}")
        return False


def load_project_data(project_folder: str, override_existing: bool = True):
    """Load all data from a project folder into session state

    Args:
        project_folder: Name of the project folder
        override_existing: If False, don't override values already in session_state
    """
    logger.info("!!! load project data called")
    project_path = EXTRACTED_PDF_DIR / project_folder
    if not project_path.exists():
        logger.error(f"Project folder not found: {project_folder}")
        st.error(f"Project folder not found: {project_folder}")
        return False

    logger.info(f"LOAD_PROJECT_DATA: {project_folder} (override_existing={override_existing})")

    try:
        # Load metadata
        metadata_path = project_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                st.session_state.extraction_result = {
                    'activity_id': project_folder,
                    'status': 'loaded',
                    'metadata': metadata,
                    'pdf_path': str(project_path / "uploaded.pdf"),
                    'output_dir': str(project_path),
                    'features': {},
                    'page_categories': [],
                    'summary': '',
                    'finance': {}
                }
                st.session_state.confirmed_metadata = metadata

                # Extract values for auto-fill
                if 'planned_start_date' in metadata and 'planned_end_date' in metadata:
                    try:
                        start = datetime.fromisoformat(metadata['planned_start_date'])
                        end = datetime.fromisoformat(metadata['planned_end_date'])
                        duration_years = (end - start).days / 365.25
                        if override_existing or 'planned_duration' not in st.session_state.extracted_values:
                            st.session_state.extracted_values['planned_duration'] = duration_years
                            st.session_state.extracted_values['start_date'] = metadata['planned_start_date']
                            st.session_state.field_edited['planned_duration'] = True
                            st.session_state.field_edited['start_date'] = True
                    except Exception as e:
                        logger.warning(f"Could not parse dates from metadata to compute duration: {e}")

        # Load page categories (JSONL format - one JSON per line)
        page_cat_path = project_path / "page_categories.jsonl"
        if page_cat_path.exists():
            try:
                page_categories = []
                with open(page_cat_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            page_categories.append(json.loads(line))
                st.session_state.extraction_result['page_categories'] = page_categories
                logger.info(f"Loaded {len(page_categories)} page categories from disk")
            except Exception as e:
                logger.warning(f"Warning: Could not load page categories: {e}")

        # Load summary
        summary_path = project_path / "summary.jsonl"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)
                    # Try different field names for summary text
                    summary_text = summary_data.get('response',
                                   summary_data.get('chatgpt_description',
                                   summary_data.get('summary',
                                   summary_data.get('description', ''))))
                    st.session_state.extraction_result['summary'] = summary_text
                    logger.info(f"Loaded summary ({len(summary_text)} chars) from disk")
            except Exception as e:
                logger.warning(f"Warning: Could not load summary: {e}")


        logger.info("!!! # Load feature grades if available")
        # Load feature grades if available
        grades_path = project_path / "feature_grades.jsonl"
        if grades_path.exists():
            try:
                with open(grades_path, 'r') as f:
                    grades_data = json.load(f)
                    loaded_grades = grades_data.get('grades', {})
                    if loaded_grades:
                        logger.info("!!! setting the feature grades state from lodaed json")
                        st.session_state.feature_grades = loaded_grades
                        # Also load into features for use (but don't override locks)
                        for feature, grade in loaded_grades.items():
                            if feature not in st.session_state.features:
                                st.session_state.features[feature] = grade
                                st.session_state.field_edited[feature] = True
                        logger.info(f"Loaded {len(loaded_grades)} grades from disk")
            except Exception as e:
                logger.warning(f"Warning: Could not load grades: {e}")

        # Load finance data
        finance_path = project_path / "finance_breakdown.jsonl"
        if finance_path.exists():
            try:
                with open(finance_path, 'r') as f:
                    finance_wrapper = json.load(f)

                    # Parse response_text which contains the actual finance JSON as a string
                    response_text = finance_wrapper.get('response_text', '{}')
                    finance_data = json.loads(response_text) if isinstance(response_text, str) else response_text

                    # Store parsed finance data in extraction_result for viewer
                    st.session_state.extraction_result['finance'] = finance_data
                    logger.info("Loaded finance breakdown from disk")

                    # Extract planned expenditure for model inputs
                    total_alloc = finance_data.get('total_allocation', {})
                    amount = total_alloc.get('amount')
                    currency = total_alloc.get('currency', '').lower()

                    if amount is not None:
                        if 'million' in currency:
                            amount_usd = float(amount) * 1_000_000
                        elif 'billion' in currency:
                            amount_usd = float(amount) * 1_000_000_000
                        else:
                            amount_usd = float(amount)

                        logger.info(f">>> LOAD_PROJECT_DATA: Extracted planned_expenditure from finance: {amount_usd}")
                        logger.info(f"override_existing={override_existing}")
                        logger.info(f"'planned_expenditure' in extracted_values: {'planned_expenditure' in st.session_state.extracted_values}")
                        logger.info(f"Current extracted_values['planned_expenditure']: {st.session_state.extracted_values.get('planned_expenditure', 'NOT_SET')}")

                        if override_existing or 'planned_expenditure' not in st.session_state.extracted_values:
                            logger.info(f">>> OVERWRITING extracted_values['planned_expenditure'] with {amount_usd}")
                            st.session_state.extracted_values['planned_expenditure'] = amount_usd
                            st.session_state.field_edited['planned_expenditure'] = True
                        else:
                            logger.info(">>> NOT OVERWRITING (override_existing=False and value already exists)")
            except Exception as e:
                logger.warning(f"Warning: Could not load finance data: {e}")

        # Load feature grades (targets, context, risks, etc.)
        feature_files = {
            'targets': 'targets.jsonl',
            'context': 'context.jsonl',
            'risks': 'risks.jsonl',
            'implementer_performance': 'implementer_performance.jsonl',
            'finance': 'finance_qualitative.jsonl',
            'integratedness': 'integratedness.jsonl',
            'complexity': 'complexity.jsonl'
        }

        for feature_name, filename in feature_files.items():
            feature_path = project_path / filename
            if feature_path.exists():
                try:
                    with open(feature_path, 'r') as f:
                        data = json.load(f)  # Load as single JSON, not JSONL

                        # Get the feature text from response_text
                        feature_text = data.get('response_text', data.get('text', ''))

                        # Store the extracted text in extraction_result
                        if 'extraction_result' in st.session_state:
                            if 'features' not in st.session_state.extraction_result:
                                st.session_state.extraction_result['features'] = {}
                            st.session_state.extraction_result['features'][feature_name] = feature_text

                        # Also check for grade if present (from older format)
                        grade = data.get('grade')
                        if grade is not None:
                            if override_existing or feature_name not in st.session_state.features:
                                st.session_state.features[feature_name] = float(grade)
                                st.session_state.field_edited[feature_name] = True

                        logger.info(f"Loaded {feature_name}: {len(feature_text)} chars")
                except Exception as e:
                    logger.warning(f"Warning: Could not load {feature_name} from {filename}: {e}")

        st.session_state.phases_0_3_complete = True
        logger.info(f"load_project_data complete. st.session_state.features = {st.session_state.features}")
        return True

    except Exception as e:
        logger.exception(f"ERROR in load_project_data: {str(e)}")
        st.error(f"Error loading project data: {str(e)}")
        return False


def save_project_name(project_folder: str, new_name: str):
    """Save project name to a metadata file"""
    project_path = EXTRACTED_PDF_DIR / project_folder
    name_file = project_path / "project_name.txt"
    try:
        with open(name_file, 'w') as f:
            f.write(new_name)
        return True
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.exception("ERROR saving project name:")
        st.error(f"Error saving project name: {str(e)}")
        return False


def load_project_name(project_folder: str) -> str:
    """Load project name from metadata, or generate default"""
    project_path = EXTRACTED_PDF_DIR / project_folder
    name_file = project_path / "project_name.txt"

    # Try to load saved name from project_name.txt
    saved_name = None
    if name_file.exists():
        try:
            with open(name_file, 'r') as f:
                saved_name = f.read().strip()
        except Exception:
            pass

    # Try to get title from metadata.json
    metadata_title = None
    metadata_path = project_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                title = metadata.get('title', '')
                if title and title != 'NOT FOUND':
                    metadata_title = title
        except Exception:
            pass

    # Prefer metadata title over "Untitled" saved names
    if metadata_title and (not saved_name or saved_name.startswith('Untitled')):
        return metadata_title
    elif saved_name:
        return saved_name

    # Default: use folder name or generate timestamp-based name
    if project_folder.startswith('webapp_'):
        # Generate friendly name from current time
        return f"Untitled - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    return project_folder


def save_project_state_temp(project_folder: str):
    """Save temporary auto-save state (for page navigation within same session).
    This is NOT a permanent save - only used to preserve widget state when switching pages.
    If user refreshes browser, temp state is ignored and last manual save is used.

    Only creates temp file if there are unsaved changes compared to permanent save."""
    project_path = EXTRACTED_PDF_DIR / project_folder
    temp_file = project_path / "app_state_temp.json"
    state_file = project_path / "app_state.json"

    try:
        # Collect current widget state
        widget_state = {}
        for key in st.session_state.keys():
            if key.startswith('_'):
                continue
            should_save = key.startswith(('input_', 'select_'))
            if key.startswith('lock_') and not key.startswith('lock_all_'):
                should_save = True
            if should_save:
                val = st.session_state[key]
                if val is not None:
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    try:
                        json.dumps(val)
                        widget_state[key] = val
                        if key == 'input_gdp_percap':
                            _gdp_debug(f"save_project_state_temp saving input_gdp_percap={val!r}")
                    except (TypeError, ValueError):
                        pass

        # Compare with permanent save to check if there are changes
        has_changes = False
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                saved_widget_state = saved_state.get('widget_state', {})

                # Check if any widget differs
                all_keys = set(widget_state.keys()) | set(saved_widget_state.keys())
                for key in all_keys:
                    if widget_state.get(key) != saved_widget_state.get(key):
                        has_changes = True
                        break
            except Exception:
                has_changes = True  # If can't read save file, assume changes
        else:
            has_changes = True  # No save file yet, so everything is unsaved

        # Only create temp file if there are actual changes
        if has_changes:
            temp_state = {
                'session_id': st.session_state.get('session_id'),
                'timestamp': datetime.now().isoformat(),
                'widget_state': widget_state,
                'project_name': st.session_state.get('project_name'),
                'location_countries': st.session_state.get('location_countries', []),
                'extracted_values': st.session_state.get('extracted_values', {}),
                'field_edited': st.session_state.get('field_edited', {}),
                'sector_percentages': st.session_state.get('sector_percentages', {}),
            }

            with open(temp_file, 'w') as f:
                json.dump(temp_state, f, indent=2)

            logger.info("Temp state auto-saved (unsaved changes detected)")
            return True
        else:
            # No changes, delete temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
                logger.info("No changes detected, temp file deleted")
            else:
                logger.info("No changes detected, no temp file needed")
            return False

    except Exception as e:
        logger.warning(f"Could not save temp state: {str(e)}")
        return False


def save_project_state(project_folder: str):
    """Save complete app state to project folder"""
    project_path = EXTRACTED_PDF_DIR / project_folder
    state_file = project_path / "app_state.json"

    try:
        # Collect embedding results if they exist
        embedding_results = st.session_state.get('embedding_results', {})

        # Snapshot widget values dynamically (input_, select_, lock_ keys)
        # This ensures we capture sector widgets, LLM grade widgets, etc.
        # IMPORTANT: Exclude button keys (like 'lock_all_*') which cannot be set in session_state
        widget_state = {}

        for key in st.session_state.keys():
            # Skip internal Streamlit keys
            if key.startswith('_'):
                continue

            # Save input and select widget keys
            should_save = key.startswith(('input_', 'select_'))

            # Save lock checkbox keys, but NOT lock button keys (lock_all_*)
            if key.startswith('lock_') and not key.startswith('lock_all_'):
                should_save = True

            if should_save:
                val = st.session_state[key]
                if val is not None:
                    # Convert non-JSON-serializable types
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    try:
                        json.dumps(val)  # Test if serializable
                        widget_state[key] = val
                    except (TypeError, ValueError):
                        logger.warning(f"Skipping non-serializable widget {key}: {type(val)}")

        # Collect all relevant state
        state = {
            'timestamp': datetime.now().isoformat(),
            'extracted_values': st.session_state.get('extracted_values', {}),
            'field_edited': st.session_state.get('field_edited', {}),
            'field_locks': st.session_state.get('field_locks', {}),
            'features': st.session_state.get('features', {}),
            'sector_percentages': st.session_state.get('sector_percentages', {}),
            'location_countries': st.session_state.get('location_countries', []),
            'phases_0_3_complete': st.session_state.get('phases_0_3_complete', False),
            'ready_for_phase_4': st.session_state.get('ready_for_phase_4', False),
            'confirmed_metadata': st.session_state.get('confirmed_metadata', {}),
            'embedding_results': embedding_results,
            'widget_state': widget_state,
            'feature_table': st.session_state.get('feature_table', None),
            'feature_grades': st.session_state.get('feature_grades', {}),
            'extraction_result': st.session_state.get('extraction_result'),
        }

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        st.session_state.saved_state_hash = get_state_hash()
        st.session_state.last_saved_timestamp = state['timestamp']

        # --- Use existing function to save project_name.txt ---
        project_name = st.session_state.get('project_name')
        if project_name:
            save_project_name(project_folder, project_name)

        # Delete temp file after successful permanent save
        temp_file = project_path / "app_state_temp.json"
        if temp_file.exists():
            try:
                temp_file.unlink()
                logger.info("Deleted temp auto-save (promoted to permanent save)")
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")

        return True
    except Exception as e:
        logger.exception(f"ERROR saving state: {str(e)}")
        st.error(f"Error saving state: {str(e)}")
        return False


def load_project_state(project_folder: str):
    """Load complete app state from project folder"""
    project_path = EXTRACTED_PDF_DIR / project_folder
    state_file = project_path / "app_state.json"

    st.session_state.saved_state_hash = None  # reset; set after successful load
    if not state_file.exists():
        return False

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)

        widget_state = state.get('widget_state', {})

        # Restore all session state
        st.session_state.extracted_values = state.get('extracted_values', {})
        st.session_state.field_edited = state.get('field_edited', {})
        st.session_state.field_locks = state.get('field_locks', {})
        st.session_state.features = state.get('features', {})
        st.session_state.sector_percentages = state.get('sector_percentages', {})
        st.session_state.location_countries = state.get('location_countries', [])
        _loc_debug(f"load_project_state PERM {project_folder}: location_countries={st.session_state.location_countries!r}")
        st.session_state.phases_0_3_complete = state.get('phases_0_3_complete', False)
        st.session_state.ready_for_phase_4 = state.get('ready_for_phase_4', False)
        st.session_state.confirmed_metadata = state.get('confirmed_metadata', {})
        st.session_state.embedding_results = state.get('embedding_results', {})
        if state.get('feature_table'):
            st.session_state.feature_table = state['feature_table']
        st.session_state.feature_grades = state.get('feature_grades', {})
        st.session_state.extraction_result = state.get('extraction_result')

        _GRADE_KEYS = {'input_finance', 'input_integratedness', 'input_implementer_performance',
                       'input_targets', 'input_context', 'input_risks', 'input_complexity'}

        # Restore all input widget values directly into session_state so widgets pick them up
        # Skip button keys (lock_all_*) which cannot be set
        # print(f"[SLIDER_DEBUG] load_project_state: restoring widget_state ({len(widget_state)} keys)")
        for key, val in widget_state.items():
            if key.startswith('lock_all_'):
                continue

            # Convert date strings back to date objects
            if 'date' in key.lower() and isinstance(val, str):
                try:
                    val = datetime.fromisoformat(val).date()
                except Exception:
                    pass

            try:
                if key in _GRADE_KEYS:
                    prev = st.session_state.get(key, '<unset>')
                    # print(f"[SLIDER_DEBUG] load_project_state widget_state: {key} = {val}  (was {prev})")
                if key == 'input_gdp_percap':
                    _gdp_debug(f"load_project_state PERM restoring input_gdp_percap={val!r} "
                               f"(was {st.session_state.get(key, '<unset>')!r})")
                st.session_state[key] = val
            except Exception as e:
                logger.warning(f"Could not restore widget key {key}: {e}")

        # Resync field_edited from widget key values vs training medians.
        if TRAIN_MEDIANS_PATH.exists():
            with open(TRAIN_MEDIANS_PATH, 'r') as f:
                train_medians = json.load(f)
            validate_and_sync_field_edited(train_medians)

        # Capture hash of permanent save state before any temp overrides
        _perm_hash = get_state_hash()
        st.session_state.last_saved_timestamp = state.get('timestamp', '')

        # Check for temporary auto-save from same session
        temp_file = project_path / "app_state_temp.json"
        if temp_file.exists():
            try:
                with open(temp_file, 'r') as f:
                    temp_state = json.load(f)
                temp_session_id = temp_state.get('session_id')
                current_session_id = st.session_state.get('session_id')

                _loc_debug(f"load_project_state TEMP {project_folder}: temp_exists=True "
                           f"temp_session={temp_session_id!r} current_session={current_session_id!r} match={temp_session_id == current_session_id} "
                           f"temp_location_countries={temp_state.get('location_countries')!r}")
                if temp_session_id == current_session_id:
                    # Restore widget state from temp (in-session changes not yet saved)
                    temp_widget_state = temp_state.get('widget_state', {})
                    # print(f"[SLIDER_DEBUG] load_project_state: restoring temp widget_state ({len(temp_widget_state)} keys)")
                    for key, val in temp_widget_state.items():
                        if key.startswith('lock_all_'):
                            continue
                        if 'date' in key.lower() and isinstance(val, str):
                            try:
                                val = datetime.fromisoformat(val).date()
                            except Exception:
                                pass
                        try:
                            if key in _GRADE_KEYS:
                                prev = st.session_state.get(key, '<unset>')
                                # print(f"[SLIDER_DEBUG] load_project_state temp_widget_state: {key} = {val}  (was {prev})")
                            if key == 'input_gdp_percap':
                                _gdp_debug(f"load_project_state TEMP restoring input_gdp_percap={val!r} "
                                           f"(was {st.session_state.get(key, '<unset>')!r})")
                            st.session_state[key] = val
                        except Exception as e:
                            logger.warning(f"Could not restore temp widget {key}: {e}")

                    # Restore other session state from temp
                    if 'location_countries' in temp_state:
                        st.session_state.location_countries = temp_state['location_countries']
                    if 'extracted_values' in temp_state:
                        st.session_state.extracted_values.update(temp_state['extracted_values'])
                    if 'field_edited' in temp_state:
                        st.session_state.field_edited.update(temp_state['field_edited'])
                    if 'sector_percentages' in temp_state:
                        st.session_state.sector_percentages = temp_state['sector_percentages']
                else:
                    # Delete stale temp file from previous session
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Could not load temp file: {e}")

        st.session_state.saved_state_hash = _perm_hash
        return True
    except Exception as e:
        logger.exception(f"ERROR loading state: {str(e)}")
        st.error(f"Error loading state: {str(e)}")
        return False
