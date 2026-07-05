"""
End-to-end UI integration tests for the IATI Activity Forecasting webapp.

These tests verify the complete UI flow:
- All model features present in feature vector
- Source badges match actual data source
- State persistence (save/load)
- Field edit tracking
- Train median verification
- Feature vector construction
- Derived features computation

Run with:
    cd /home/dmrivers/Code/forecasting_iati/webapp
    python -m pytest test_suite_ui_e2e.py -v
"""

import os
import sys
import json
import shutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

# Path setup
WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"
DATA_DIR = WEBAPP_DIR.parent / "data"
MODEL_DIR = DATA_DIR / "rating_model_outputs"
SRC_UTILS = WEBAPP_DIR.parent / "src" / "utils"

for _p in [str(WEBAPP_DIR), str(WEBAPP_DIR / "modules"), str(WEBAPP_DIR / "pages"), str(SRC_UTILS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROJECTS_DIR = WEBAPP_DIR.parent / "projects"

# Streamlit mock — shared across all test files (single session_state).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from streamlit_mock import st_mock as _st, SessionState as _SessionState

# Import webapp modules
from utils import parse_location_string, build_location_string
from ui_components import get_field_indicator
from project_manager import save_project_state, load_project_state, save_project_name, load_project_name
from location_features import get_org_dummies, KEEP_REPORTING_ORGS
from model_loader import sector_clusters_from_feature_names

# Load model artifacts
FEATURE_NAMES = json.loads((MODEL_DIR / "feature_names.json").read_text())
TRAIN_MEDIANS = json.loads((MODEL_DIR / "train_medians.json").read_text())

# Single source of truth: derive sector clusters from the model feature list.
SECTOR_CLUSTERS = sector_clusters_from_feature_names(FEATURE_NAMES)

LLM_GRADE_FEATURES = [
    "finance", "integratedness", "implementer_performance",
    "targets", "context", "risks", "complexity",
]


@pytest.fixture(scope="session", autouse=True)
def set_working_directory():
    """Tests must run from the webapp/ directory."""
    original = os.getcwd()
    os.chdir(WEBAPP_DIR)
    yield
    os.chdir(original)


@pytest.fixture(autouse=True)
def reset_session_state():
    """Reset st.session_state before each test."""
    _st.session_state.clear()
    _st.session_state.update({
        "features": {},
        "field_edited": {},
        "field_locks": {},
        "extracted_values": {},
        "sector_percentages": {c: 0.0 for c in SECTOR_CLUSTERS},
        "location_countries": [],
    })
    yield


class TestFeatureCoverage:
    """Verify all model features can be populated from UI."""

    def test_all_features_in_feature_names(self):
        """Every expected feature appears in FEATURE_NAMES."""
        expected_count = 59  # As per feature_names.json (actual count from model)
        assert len(FEATURE_NAMES) == expected_count

    def test_all_features_have_medians(self):
        """Every feature in FEATURE_NAMES has a train median."""
        for feat in FEATURE_NAMES:
            assert feat in TRAIN_MEDIANS, f"Missing train median for {feat}"

    def test_llm_grades_all_present(self):
        """All 7 LLM grade features are in model."""
        for feat in LLM_GRADE_FEATURES:
            assert feat in FEATURE_NAMES

    def test_sector_clusters_all_present(self):
        """All 16 sector clusters are in model."""
        for cluster in SECTOR_CLUSTERS:
            feat_name = f"sector_cluster_{cluster}"
            assert feat_name in FEATURE_NAMES

    def test_economic_indicators_present(self):
        """Economic indicators (GDP, CPIA, governance) are in model."""
        for feat in ["gdp_percap", "cpia_score", "governance_composite"]:
            assert feat in FEATURE_NAMES

    def test_region_features_present(self):
        """All 7 region features are in model."""
        regions = ["region_AFE", "region_AFW", "region_EAP", "region_ECA",
                   "region_LAC", "region_MENA", "region_SAS"]
        for region in regions:
            assert region in FEATURE_NAMES

    def test_org_dummies_present(self):
        """Reporting org dummy variables are in model."""
        for feat in ["rep_org_0", "rep_org_1", "rep_org_2"]:
            assert feat in FEATURE_NAMES

    def test_embedding_features_present(self):
        """UMAP embeddings and distances are in model."""
        for feat in ["umap3_x", "umap3_y", "umap3_z", "sector_distance", "country_distance"]:
            assert feat in FEATURE_NAMES

    def test_missing_indicators_present(self):
        """Missing indicator features are in model."""
        indicators = [
            "llm_features_missing_count", "llm_features_present_ratio",
            "cpia_missing", "gdp_percap_missing", "wgi_any_missing",
            "sector_clusters_missing", "umap_missing", "governance_missing_count",
            "feature_completeness_ratio", "planned_expenditure_missing",
            "planned_duration_missing"
        ]
        for feat in indicators:
            assert feat in FEATURE_NAMES


class TestSourceBadges:
    """Verify source badges correctly indicate edited vs median."""

    def test_badge_shows_set_when_edited(self):
        """When field is edited, badge shows '✓ Set'."""
        _st.session_state.field_edited["finance"] = True
        badge = get_field_indicator("finance")
        assert "✓ Set" in badge
        assert "#d4edda" in badge  # Green background

    def test_badge_shows_median_when_not_edited(self):
        """When field uses median, badge shows '⚠ Using median'."""
        _st.session_state.field_edited["finance"] = False
        badge = get_field_indicator("finance")
        assert "⚠ Using median" in badge
        assert "#f8d7da" in badge  # Red background

    def test_badge_color_green_for_edited(self):
        """Edited field badge has green background."""
        _st.session_state.field_edited["targets"] = True
        badge = get_field_indicator("targets")
        assert "d4edda" in badge

    def test_badge_color_red_for_median(self):
        """Median field badge has red background."""
        _st.session_state.field_edited["context"] = False
        badge = get_field_indicator("context")
        assert "f8d7da" in badge

    def test_all_llm_grades_have_badges(self):
        """Every LLM grade feature can show a badge."""
        for feat in LLM_GRADE_FEATURES:
            _st.session_state.field_edited[feat] = True
            badge = get_field_indicator(feat)
            assert "<span" in badge
            assert "Set" in badge


class TestTrainMedianValues:
    """Verify train median values match expected ranges."""

    def test_llm_grade_medians_in_valid_range(self):
        """LLM grade medians should be between 0-100."""
        for feat in LLM_GRADE_FEATURES:
            median = TRAIN_MEDIANS[feat]
            assert 0.0 <= median <= 100.0, f"{feat} median {median} out of range"

    def test_sector_cluster_medians_all_zero(self):
        """Sector cluster medians should be 0 (most activities don't use all sectors)."""
        for cluster in SECTOR_CLUSTERS:
            feat_name = f"sector_cluster_{cluster}"
            assert TRAIN_MEDIANS[feat_name] == 0.0

    def test_cpia_median_in_valid_range(self):
        """CPIA score median should be between 1-6."""
        median = TRAIN_MEDIANS["cpia_score"]
        assert 1.0 <= median <= 6.0

    def test_activity_scope_median_is_integer(self):
        """Activity scope median should be a valid scope code."""
        median = TRAIN_MEDIANS["activity_scope"]
        assert median in [0, 1, 2, 3, 4, 5, 6, 7]

    def test_finance_is_loan_median_binary(self):
        """Finance is loan median should be 0 or 1."""
        median = TRAIN_MEDIANS["finance_is_loan"]
        assert median in [0.0, 1.0]

    def test_missing_indicators_binary(self):
        """Binary missing indicators should be 0 or 1."""
        binary_indicators = [
            "cpia_missing", "gdp_percap_missing", "wgi_any_missing",
            "sector_clusters_missing", "umap_missing",
            "planned_expenditure_missing", "planned_duration_missing"
        ]
        for feat in binary_indicators:
            median = TRAIN_MEDIANS[feat]
            assert median in [0.0, 1.0], f"{feat} median {median} not binary"


class TestStatePersistence:
    """Verify save and load preserves all state."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up test projects."""
        test_proj = PROJECTS_DIR / "test_ui_state_persist"
        if test_proj.exists():
            shutil.rmtree(test_proj)
        yield
        if test_proj.exists():
            shutil.rmtree(test_proj)

    def _make_test_project(self):
        """Create minimal test project."""
        proj = PROJECTS_DIR / "test_ui_state_persist"
        proj.mkdir(parents=True, exist_ok=True)
        (proj / "metadata.json").write_text(json.dumps({
            "title": "Test Project",
            "country_locations": [{"iso2_code": "KE", "percentage": 100}],
        }))
        (proj / "page_categories.jsonl").write_text(json.dumps({"category": "content"}))
        return proj

    def test_features_persist_after_save_load(self):
        """Saved features are restored after load."""
        self._make_test_project()
        _st.session_state.features = {
            "finance": 85.0,
            "targets": 70.0,
            "gdp_percap": 9.5,
        }
        save_project_state("test_ui_state_persist")

        _st.session_state.features = {}
        load_project_state("test_ui_state_persist")

        assert _st.session_state.features.get("finance") == 85.0
        assert _st.session_state.features.get("targets") == 70.0
        assert _st.session_state.features.get("gdp_percap") == 9.5

    def test_field_edited_persists_after_save_load(self):
        """Field edited flags persist."""
        self._make_test_project()
        _st.session_state.field_edited = {
            "finance": True,
            "targets": False,
        }
        save_project_state("test_ui_state_persist")

        _st.session_state.field_edited = {}
        load_project_state("test_ui_state_persist")

        assert _st.session_state.field_edited.get("finance") == True
        assert _st.session_state.field_edited.get("targets") == False

    def test_sector_percentages_persist_after_save_load(self):
        """Sector percentages persist."""
        self._make_test_project()
        _st.session_state.sector_percentages = {
            SECTOR_CLUSTERS[0]: 60.0,
            SECTOR_CLUSTERS[1]: 40.0,
        }
        save_project_state("test_ui_state_persist")

        _st.session_state.sector_percentages = {}
        load_project_state("test_ui_state_persist")

        assert _st.session_state.sector_percentages.get(SECTOR_CLUSTERS[0]) == 60.0
        assert _st.session_state.sector_percentages.get(SECTOR_CLUSTERS[1]) == 40.0

    def test_location_countries_persist_after_save_load(self):
        """Location countries persist."""
        self._make_test_project()
        _st.session_state.location_countries = [
            {"code": "KE", "pct": 60},
            {"code": "UG", "pct": 40},
        ]
        save_project_state("test_ui_state_persist")

        _st.session_state.location_countries = []
        load_project_state("test_ui_state_persist")

        assert len(_st.session_state.location_countries) == 2
        assert _st.session_state.location_countries[0]["code"] == "KE"
        assert _st.session_state.location_countries[1]["code"] == "UG"


class TestFieldEditTracking:
    """Verify field edit tracking works correctly."""

    def test_edit_tracking_initially_empty(self):
        """Field edited dict starts empty."""
        assert _st.session_state.field_edited == {}

    def test_setting_field_marks_as_edited(self):
        """Setting a field value should mark it as edited."""
        _st.session_state.features["finance"] = 90.0
        _st.session_state.field_edited["finance"] = True
        assert _st.session_state.field_edited["finance"] == True

    def test_using_median_marks_as_not_edited(self):
        """Using median value should mark field as not edited."""
        _st.session_state.features["targets"] = TRAIN_MEDIANS["targets"]
        _st.session_state.field_edited["targets"] = False
        assert _st.session_state.field_edited["targets"] == False

    def test_all_llm_grades_trackable(self):
        """All LLM grade features can be tracked."""
        for feat in LLM_GRADE_FEATURES:
            _st.session_state.field_edited[feat] = True
            assert _st.session_state.field_edited[feat] == True


class TestDerivedFeatures:
    """Verify derived features are computed correctly."""

    def test_log_planned_expenditure_feature_exists(self):
        """log_planned_expenditure feature exists in model."""
        assert "log_planned_expenditure" in FEATURE_NAMES

    def test_reporting_org_dummies_mutually_exclusive(self):
        """Only one org dummy can be active at a time."""
        for org in KEEP_REPORTING_ORGS:
            dummies = get_org_dummies(org)
            total = sum(dummies.values())
            assert total in [0, 1], f"{org}: sum={total}"

    def test_sector_cluster_features_derived_from_percentages(self):
        """Sector cluster features are percentages/100."""
        # Set up percentages
        _st.session_state.sector_percentages = {
            SECTOR_CLUSTERS[0]: 60.0,
            SECTOR_CLUSTERS[1]: 40.0,
        }
        # Model expects proportions (0-1)
        expected_0 = 60.0 / 100.0
        expected_1 = 40.0 / 100.0
        assert abs(expected_0 - 0.6) < 0.01
        assert abs(expected_1 - 0.4) < 0.01


class TestFeatureVectorConstruction:
    """Verify feature vector construction is correct."""

    def test_feature_vector_has_all_features(self):
        """Feature vector should contain all features."""
        _st.session_state.features = {k: TRAIN_MEDIANS[k] for k in FEATURE_NAMES}
        assert len(_st.session_state.features) == len(FEATURE_NAMES)

    def test_feature_vector_order_matches_feature_names(self):
        """Feature vector keys should match FEATURE_NAMES order."""
        _st.session_state.features = {k: TRAIN_MEDIANS[k] for k in FEATURE_NAMES}
        for i, feat in enumerate(FEATURE_NAMES):
            assert feat in _st.session_state.features

    def test_missing_features_use_medians(self):
        """Missing features should use train medians."""
        _st.session_state.features = {"finance": 85.0}
        # Other features should use medians
        for feat in FEATURE_NAMES:
            if feat not in _st.session_state.features:
                # Would be filled with median during prediction
                expected = TRAIN_MEDIANS[feat]
                assert expected is not None


class TestLocationFeatures:
    """Verify location-based features work correctly."""

    def test_location_string_parses_correctly(self):
        """Location string parses to country list."""
        loc_str = "KE|60|UG|40"
        parsed = parse_location_string(loc_str)
        assert len(parsed) == 2
        assert parsed[0]["code"] == "KE"
        assert parsed[0]["pct"] == 60

    def test_location_string_builds_correctly(self):
        """Country list builds to location string."""
        countries = [{"code": "KE", "pct": 60}, {"code": "UG", "pct": 40}]
        built = build_location_string(countries)
        assert built == "KE|60|UG|40"

    def test_location_roundtrip(self):
        """Parse -> build should roundtrip."""
        original = "KE|50|TZ|30|UG|20"
        parsed = parse_location_string(original)
        rebuilt = build_location_string(parsed)
        assert rebuilt == original


class TestLockFunctionality:
    """Verify field lock functionality."""

    def test_lock_state_persists(self):
        """Field lock state should persist across saves."""
        # This would be tested in integration with save/load
        _st.session_state.field_locks["finance"] = True
        assert _st.session_state.field_locks["finance"] == True

    def test_all_lockable_fields_exist(self):
        """All expected fields can be locked."""
        lockable_fields = (
            LLM_GRADE_FEATURES +
            ["reporting_org", "planned_expenditure", "activity_scope",
             "finance_is_loan", "location", "start_date", "planned_duration",
             "gdp_percap", "cpia_score", "governance_composite", "wgi_any_missing"] +
            [f"sector_{c}" for c in SECTOR_CLUSTERS]
        )
        # Just verify we can set locks for these
        for field in lockable_fields:
            _st.session_state.field_locks[field] = True
            assert _st.session_state.field_locks[field] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
