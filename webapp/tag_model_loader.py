from debug_utils import _print_ram

import streamlit as st
import pickle


import logging

logger = logging.getLogger(__name__)

from webapp_paths import DATA_DIR
TAG_MODELS_PATH = DATA_DIR / "outcome_tags" / "tag_models.pkl"


@st.cache_resource
def load_tag_models():
    """Load saved outcome tag prediction models from pkl.

    Returns the dict from tag_models.pkl, or None if the file is missing.
    """
    _print_ram("load_tag_models START")
    if not TAG_MODELS_PATH.exists():
        logger.warning(f"[tag_model_loader] WARNING: tag_models.pkl not found at {TAG_MODELS_PATH}")
        return None
    with open(TAG_MODELS_PATH, "rb") as f:
        tag_data = pickle.load(f)
    _print_ram("load_tag_models END")
    return tag_data
