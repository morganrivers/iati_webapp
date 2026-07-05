from debug_utils import _print_ram


import logging

logger = logging.getLogger(__name__)

_print_ram("before model loader imports")
import streamlit as st
_print_ram("after import streamlit as st")
import pickle
_print_ram("after import pickle")
import json
_print_ram("after import json")
import pandas as pd
_print_ram("after import pandas as pd")

from webapp_paths import MODEL_DIR as BASE_PATH

_SECTOR_CLUSTER_PREFIX = "sector_cluster_"


def sector_clusters_from_feature_names(feature_names):
    """Single source of truth for the model's sector-cluster names.

    Strips the ``sector_cluster_`` prefix from the model feature list. Excludes
    the ``sector_clusters_missing`` flag (plural ``clusters`` fails the prefix).
    """
    clusters = [
        f[len(_SECTOR_CLUSTER_PREFIX):]
        for f in feature_names
        if f.startswith(_SECTOR_CLUSTER_PREFIX)
    ]
    assert clusters, "no sector_cluster_* features found in feature_names"
    return clusters


def load_feature_names():
    with open(BASE_PATH / "feature_names.json", "r") as f:
        return json.load(f)


def get_sector_clusters():
    """Derive sector-cluster names directly from the model artifact."""
    return sector_clusters_from_feature_names(load_feature_names())


@st.cache_resource
def load_model_and_data():
    """Load the RF + ExtraTrees ensemble, per-org baseline, start-year correction, and feature matrix."""
    import sys

    def _obj_mb(obj):
        try:
            return sys.getsizeof(obj) / 1024 / 1024
        except Exception:
            return float('nan')

    _print_ram("load_model_and_data START")

    with open(BASE_PATH / "model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    _print_ram("after load model.pkl")
    logger.debug(f"[SIZE] rf_model sys.getsizeof = {_obj_mb(rf_model):.1f} MB (note: RF trees not counted by getsizeof)")

    with open(BASE_PATH / "extra_model.pkl", "rb") as f:
        extra_model = pickle.load(f)
    _print_ram("after load extra_model.pkl")
    logger.debug(f"[SIZE] extra_model sys.getsizeof = {_obj_mb(extra_model):.1f} MB")

    with open(BASE_PATH / "per_org_baseline.json", "r") as f:
        per_org_baseline = json.load(f)
    _print_ram("after load per_org_baseline.json")

    with open(BASE_PATH / "start_year_correction.json", "r") as f:
        correction = json.load(f)
    _print_ram("after load start_year_correction.json")

    with open(BASE_PATH / "train_medians.json", "r") as f:
        train_medians = json.load(f)
    _print_ram("after load train_medians.json")

    with open(BASE_PATH / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    _print_ram("after load feature_names.json")

    _print_ram("before read train_features.csv")
    training_features = pd.read_csv(BASE_PATH / "train_features.csv")
    _print_ram("after read train_features.csv")
    logger.debug(f"[SIZE] training_features DataFrame: {training_features.shape}, "
                 f"memory_usage = {training_features.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    assert "__overall__" in per_org_baseline, "per_org_baseline.json missing __overall__"
    assert "slope" in correction and "intercept" in correction, "start_year_correction.json malformed"

    metadata = {
        'train_medians': train_medians,
        'feature_names': feature_names,
        'sector_clusters': sector_clusters_from_feature_names(feature_names),
    }

    return rf_model, extra_model, per_org_baseline, correction, metadata, training_features
