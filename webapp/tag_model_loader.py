import logging
import pickle

import streamlit as st

from webapp_paths import DATA_DIR

logger = logging.getLogger(__name__)

TAG_MODELS_PATH = DATA_DIR / "outcome_tags" / "tag_models.pkl"


@st.cache_resource
def load_tag_models():
    """Load saved outcome tag prediction models from pkl.

    Returns the dict from tag_models.pkl, or None if the file is missing.
    """
    if not TAG_MODELS_PATH.exists():
        logger.warning(f"tag_models.pkl not found at {TAG_MODELS_PATH}")
        return None
    with open(TAG_MODELS_PATH, "rb") as f:
        return pickle.load(f)
