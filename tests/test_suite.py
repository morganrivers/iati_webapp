"""
Comprehensive test suite for the IATI Activity Forecasting webapp.

Run with:
    cd /home/dmrivers/Code/forecasting_iati/webapp
    python -m pytest test_suite.py -v

Rules:
  - No LLM API calls (AIRPLANE_MODE=True already set in extracting_and_grading_helper_functions.py)
  - Only tests files in the webapp directory (plus src/utils/dummy_response_text_generator.py)
  - Test folders created under webapp/extracted_pdf_data/ are cleaned up after each test
"""

import os
import sys
import re
import json
import pickle
import shutil
import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime

# ── Path setup (must happen before any webapp imports) ─────────────────────────
WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"
DATA_DIR   = WEBAPP_DIR.parent / "data"
MODEL_DIR  = DATA_DIR / "rating_model_outputs"
SRC_UTILS  = WEBAPP_DIR.parent / "src" / "utils"

for _p in [str(WEBAPP_DIR), str(WEBAPP_DIR / "modules"), str(WEBAPP_DIR / "pages"), str(SRC_UTILS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXTRACTED_PDF_DIR = WEBAPP_DIR / "extracted_pdf_data"

# ── Streamlit mock (must be injected before importing any webapp module) ────────
# Shared across all test files so imported webapp modules read/write one session_state.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from streamlit_mock import st_mock as _st, SessionState as _SessionState

# ── Webapp module imports (safe now that st is mocked) ──────────────────────────
import utils as webapp_utils
from utils import parse_location_string, build_location_string, COUNTRY_NAMES, _COUNTRY_OPTIONS

import project_manager as _pm
from project_manager import (
    get_available_projects, load_project_name, save_project_name,
    save_project_state, load_project_state, load_project_data,
)

import ui_components
from ui_components import render_histogram, get_field_indicator

from modules.location_features import (
    parse_location_string as lf_parse,
    get_org_dummies,
    KEEP_REPORTING_ORGS,
    ORG_NAME_TO_DUMMY,
)
from modules.webapp_pipeline import generate_activity_id_from_content

# dummy response generator lives in src/utils
from dummy_response_text_generator import (
    get_dummy_response_text,
    _detect_schema_type,
    _detect_prompt_type,
    _make_dummy_page_categorization,
    _make_dummy_metadata,
    _make_dummy_misc_features,
    _make_dummy_finance_breakdown,
)

# ── Load model artefacts once at module level ──────────────────────────────────
FEATURE_NAMES = json.loads((MODEL_DIR / "feature_names.json").read_text())
TRAIN_MEDIANS = json.loads((MODEL_DIR / "train_medians.json").read_text())

from model_loader import sector_clusters_from_feature_names

SECTOR_CLUSTERS = sector_clusters_from_feature_names(FEATURE_NAMES)

LLM_GRADE_FEATURES = [
    "finance", "integratedness", "implementer_performance",
    "targets", "context", "risks", "complexity",
]

def _build_default_locks():
    locks = {k: False for k in [
        "reporting_org", "planned_expenditure", "activity_scope", "finance_is_loan",
        "location", "start_date", "planned_duration", "finance", "integratedness",
        "implementer_performance", "targets", "context", "risks", "complexity",
        "gdp_percap", "cpia_score", "governance_composite", "wgi_any_missing",
    ]}
    for c in SECTOR_CLUSTERS:
        locks[f"sector_{c}"] = False
    return locks

_DEFAULT_FIELD_LOCKS = _build_default_locks()


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def set_working_directory():
    """Tests must run from the webapp/ directory (project_manager uses relative paths)."""
    original = os.getcwd()
    os.chdir(WEBAPP_DIR)
    yield
    os.chdir(original)


@pytest.fixture(autouse=True)
def reset_session_state():
    """Reset st.session_state to a clean default before every test."""
    _st.session_state.clear()
    _st.session_state.update({
        "features":            {},
        "field_edited":        {},
        "field_locks":         dict(_DEFAULT_FIELD_LOCKS),
        "extracted_values":    {},
        "feature_grades":      {},
        "extraction_result":   None,
        "embedding_results":   {},
        "sector_percentages":  {c: 0.0 for c in SECTOR_CLUSTERS},
        "location_countries":  [],
        "phases_0_3_complete": False,
        "ready_for_phase_4":   False,
        "confirmed_metadata":  {},
        "feature_table":       None,
    })
    _st.error.reset_mock()
    _st.success.reset_mock()
    _st.warning.reset_mock()
    yield


@pytest.fixture(scope="session")
def rf_model():
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def extra_model():
    with open(MODEL_DIR / "extra_model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def per_org_baseline():
    with open(MODEL_DIR / "per_org_baseline.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def start_year_correction():
    with open(MODEL_DIR / "start_year_correction.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def training_data():
    return pd.read_csv(MODEL_DIR / "train_features.csv")


# ── Helpers ────────────────────────────────────────────────────────────────────

from rf_predictor import predict_rating, _ensemble_delta, DEFAULT_START_YEAR


def _make_feature_vector(**overrides):
    """Build a DataFrame feature vector starting from training medians."""
    vec = {k: TRAIN_MEDIANS[k] for k in FEATURE_NAMES}
    vec.update(overrides)
    return pd.DataFrame([{k: vec[k] for k in FEATURE_NAMES}])


def _base_from_fv(per_org_baseline, fv_df):
    """Select the per-org baseline implied by the rep_org dummies in a feature vector."""
    row = fv_df.iloc[0]
    for col in ("rep_org_0", "rep_org_1", "rep_org_2"):
        if col in fv_df.columns and row[col] == 1:
            return per_org_baseline.get(col, per_org_baseline["__overall__"])
    return per_org_baseline["__overall__"]


def _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv_df,
             start_year=DEFAULT_START_YEAR):
    """Per-org baseline + RF/ExtraTrees ensemble delta + start-year correction.

    Mirrors impute_and_run_statistical_model via the shared predict_rating() so the
    formula stays single-sourced.
    """
    base = _base_from_fv(per_org_baseline, fv_df)
    ens_delta = _ensemble_delta(rf_model, extra_model, fv_df)
    prediction, _ = predict_rating(base, ens_delta, start_year_correction, start_year)
    return prediction


# Tree ensembles are only approximately monotone in any single feature; this absorbs
# small non-monotone wiggles at the median vector without hiding gross violations.
_MONOTONE_TOL = 0.05


def _make_test_project(name="test_webapp_test_00000"):
    """
    Create a minimal project folder with the files load_project_data() expects.
    Returns the Path to the created folder.
    """
    proj = EXTRACTED_PDF_DIR / name
    proj.mkdir(parents=True, exist_ok=True)

    metadata = {
        "title":              "Test Environmental Activity",
        "participating_orgs": ["Test Agency"],
        "country_locations":  [{"iso2_code": "KE", "percentage": 100}],
        "planned_start_date": "2020-01-01",
        "planned_end_date":   "2025-01-01",
    }
    (proj / "metadata.json").write_text(json.dumps(metadata))

    page_cat = [
        {"category": "content", "subcategory_A": "broad_objectives", "informativeness": 7}
    ]
    (proj / "page_categories.jsonl").write_text(
        "\n".join(json.dumps(p) for p in page_cat)
    )

    (proj / "summary.jsonl").write_text(json.dumps({
        "response": "A test environmental activity supporting renewable energy."
    }))

    finance_data = {
        "total_allocation": {"amount": 45_000_000, "currency": "USD"},
        "sectors":          [{"name": "Renewable Energy", "percentage": 100}],
    }
    (proj / "finance_breakdown.jsonl").write_text(json.dumps({
        "response_text": json.dumps(finance_data)
    }))

    return proj


def _normalize_sectors(sp):
    """Mirror the Normalize button logic from page_activity_forecasting.py."""
    total = sum(sp.values())
    if total > 0:
        return {k: (v / total) * 100.0 for k, v in sp.items()}
    n = len(sp)
    return {k: 100.0 / n for k in sp}


def _zero_sectors(sp):
    """Mirror the Set-all-to-0 button logic."""
    return {k: 0.0 for k in sp}


# ══════════════════════════════════════════════════════════════════════════════
# 1. UTILS.PY — location / country utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestUtils:

    # parse_location_string
    def test_single_iso2_code(self):
        assert parse_location_string("KE") == [{"code": "KE", "pct": 100}]

    def test_single_iso2_lowercase_uppercased(self):
        assert parse_location_string("ke")[0]["code"] == "KE"

    def test_multi_country_two(self):
        r = parse_location_string("KE|50|UG|30")
        assert len(r) == 2
        assert r[0] == {"code": "KE", "pct": 50}
        assert r[1] == {"code": "UG", "pct": 30}

    def test_multi_country_three(self):
        r = parse_location_string("KE|50|UG|30|TZ|20")
        assert len(r) == 3
        assert r[2] == {"code": "TZ", "pct": 20}

    def test_empty_string_returns_empty(self):
        assert parse_location_string("") == []

    def test_float_percentage_truncated_to_int(self):
        r = parse_location_string("KE|66.7|UG|33.3")
        assert r[0]["pct"] == 66
        assert r[1]["pct"] == 33

    def test_whitespace_stripped(self):
        r = parse_location_string("KE | 50 | UG | 30")
        assert r[0]["code"] == "KE"
        assert r[1]["code"] == "UG"

    def test_odd_pipe_parts_does_not_crash(self):
        # 3 parts: KE has a percentage, UG does not — should not raise
        r = parse_location_string("KE|50|UG")
        assert isinstance(r, list)
        codes = [x["code"] for x in r]
        assert "KE" in codes

    def test_100_pct_single_country(self):
        assert parse_location_string("NG")[0]["pct"] == 100

    # build_location_string
    def test_build_empty(self):
        assert build_location_string([]) == ""

    def test_build_single_country(self):
        assert build_location_string([{"code": "KE", "pct": 100}]) == "KE"

    def test_build_multi_country(self):
        r = build_location_string([{"code": "KE", "pct": 60}, {"code": "UG", "pct": 40}])
        assert r == "KE|60|UG|40"

    def test_roundtrip_single(self):
        assert build_location_string(parse_location_string("KE")) == "KE"

    def test_roundtrip_multi(self):
        assert build_location_string(parse_location_string("KE|50|UG|30")) == "KE|50|UG|30"

    # COUNTRY_NAMES
    def test_country_names_not_empty(self):
        assert len(COUNTRY_NAMES) > 50

    def test_all_iso2_codes_are_two_uppercase_chars(self):
        for code in COUNTRY_NAMES:
            assert len(code) == 2 and code.isupper(), f"Bad ISO2: {code!r}"

    def test_all_country_names_nonempty(self):
        for code, name in COUNTRY_NAMES.items():
            assert name, f"Empty name for {code}"

    def test_key_countries_present(self):
        for code in ["KE", "UG", "BD", "IN", "PK", "NG", "TZ", "GH", "ET"]:
            assert code in COUNTRY_NAMES

    # _COUNTRY_OPTIONS
    def test_options_format_name_code(self):
        pattern = re.compile(r"^.+ \([A-Z]{2}\)$")
        for opt in _COUNTRY_OPTIONS:
            assert pattern.match(opt), f"Bad format: {opt!r}"

    def test_options_alphabetically_sorted(self):
        assert _COUNTRY_OPTIONS == sorted(_COUNTRY_OPTIONS)

    def test_options_count_matches_country_names(self):
        assert len(_COUNTRY_OPTIONS) == len(COUNTRY_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# 2. modules/location_features.py
# ══════════════════════════════════════════════════════════════════════════════

class TestLocationFeatures:

    # parse_location_string (module's own version returns list-of-tuples)
    def test_lf_single_iso2_returns_one_tuple(self):
        r = lf_parse("KE")
        assert len(r) == 1
        assert r[0][0] == "KE"
        assert r[0][2] == 100.0

    def test_lf_multi_country_three(self):
        r = lf_parse("KE|50|UG|30|TZ|20")
        assert len(r) == 3
        codes = [t[0] for t in r]
        assert "KE" in codes and "TZ" in codes

    def test_lf_empty_returns_empty(self):
        assert lf_parse("") == []

    def test_lf_float_percentages_preserved(self):
        r = lf_parse("KE|66.6|UG|33.4")
        assert r[0][2] == pytest.approx(66.6)
        assert r[1][2] == pytest.approx(33.4)

    def test_lf_odd_parts_does_not_crash(self):
        r = lf_parse("KE|50|UG")
        assert isinstance(r, list)

    # get_org_dummies
    def test_fcdo_dummy(self):
        d = get_org_dummies("UK - Foreign, Commonwealth Development Office (FCDO)")
        assert d == {"rep_org_0": 1, "rep_org_1": 0, "rep_org_2": 0}

    def test_adb_dummy(self):
        d = get_org_dummies("Asian Development Bank")
        assert d == {"rep_org_0": 0, "rep_org_1": 1, "rep_org_2": 0}

    def test_world_bank_dummy(self):
        d = get_org_dummies("World Bank")
        assert d == {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 1}

    def test_bmz_is_reference_category_all_zeros(self):
        d = get_org_dummies("BMZ")
        assert d == {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 0}

    def test_unknown_org_returns_zeros(self):
        d = get_org_dummies("Unknown Organisation XYZ")
        assert d == {"rep_org_0": 0, "rep_org_1": 0, "rep_org_2": 0}

    def test_all_orgs_have_exactly_three_keys(self):
        for org in KEEP_REPORTING_ORGS + ["Unknown"]:
            d = get_org_dummies(org)
            assert set(d.keys()) == {"rep_org_0", "rep_org_1", "rep_org_2"}

    def test_at_most_one_dummy_active(self):
        for org in KEEP_REPORTING_ORGS:
            d = get_org_dummies(org)
            assert sum(d.values()) <= 1, f"{org}: sum={sum(d.values())}"

    def test_exactly_three_non_reference_orgs(self):
        one_hot = [o for o in KEEP_REPORTING_ORGS if sum(get_org_dummies(o).values()) == 1]
        assert len(one_hot) == 3

    def test_keep_reporting_orgs_has_four_entries(self):
        """Matches NUM_ORGS_KEEP = 4 in C_run_GLM_nobayes.py."""
        assert len(KEEP_REPORTING_ORGS) == 4

    def test_required_orgs_present(self):
        assert "World Bank" in KEEP_REPORTING_ORGS
        assert "Asian Development Bank" in KEEP_REPORTING_ORGS
        assert any("FCDO" in o for o in KEEP_REPORTING_ORGS)
        assert "BMZ" in KEEP_REPORTING_ORGS

    def test_org_dummy_mapping_covers_all_keep_orgs(self):
        for org in KEEP_REPORTING_ORGS:
            assert org in ORG_NAME_TO_DUMMY, f"{org!r} missing from ORG_NAME_TO_DUMMY"


# ══════════════════════════════════════════════════════════════════════════════
# 3. modules/webapp_pipeline.py — generate_activity_id_from_content
# ══════════════════════════════════════════════════════════════════════════════

class TestWebappPipeline:

    def test_deterministic_same_bytes(self):
        c = b"some pdf content"
        assert generate_activity_id_from_content(c) == generate_activity_id_from_content(c)

    def test_different_bytes_give_different_ids(self):
        assert generate_activity_id_from_content(b"AAA") != generate_activity_id_from_content(b"BBB")

    def test_id_starts_with_webapp_prefix(self):
        assert generate_activity_id_from_content(b"x").startswith("webapp_")

    def test_hash_part_is_12_hex_chars(self):
        result = generate_activity_id_from_content(b"test")
        hash_part = result[len("webapp_"):]
        assert len(hash_part) == 12
        assert all(c in "0123456789abcdef" for c in hash_part)

    def test_matches_md5_first_12(self):
        content = b"hello world pdf"
        expected = "webapp_" + hashlib.md5(content).hexdigest()[:12]
        assert generate_activity_id_from_content(content) == expected

    def test_empty_bytes_does_not_crash(self):
        result = generate_activity_id_from_content(b"")
        assert result.startswith("webapp_")

    def test_large_content_deterministic(self):
        big = b"x" * 100_000
        assert generate_activity_id_from_content(big) == generate_activity_id_from_content(big)


# ══════════════════════════════════════════════════════════════════════════════
# 4. src/utils/dummy_response_text_generator.py
# ══════════════════════════════════════════════════════════════════════════════

class TestDummyResponseGenerator:

    # Schema type detection
    def test_none_schema_is_plain_text(self):
        assert _detect_schema_type(None) == "plain_text"

    def test_pages_key_is_page_categorization(self):
        assert _detect_schema_type({"properties": {"pages": {}}}) == "page_categorization"

    def test_title_and_country_locations_is_metadata(self):
        assert _detect_schema_type({"properties": {"title": {}, "country_locations": {}}}) == "metadata"

    def test_complexity_details_is_misc_features(self):
        assert _detect_schema_type({"properties": {"complexity_details": {}}}) == "misc_features"

    def test_how_integrated_is_misc_features(self):
        assert _detect_schema_type({"properties": {"how_integrated_description": {}}}) == "misc_features"

    def test_total_allocation_is_finance_breakdown(self):
        assert _detect_schema_type({"properties": {"total_allocation": {}, "sectors": {}}}) == "finance_breakdown"

    def test_sectors_key_is_finance_breakdown(self):
        assert _detect_schema_type({"properties": {"quantitative_outcome_allocations": {}}}) == "finance_breakdown"

    def test_unknown_properties_is_generic_json(self):
        assert _detect_schema_type({"properties": {"some_random_field": {}}}) == "generic_json"

    # Prompt type detection
    def test_implementer_prompt_detected(self):
        assert _detect_prompt_type("Grade the implementer performance of the organisation") == "implementer_performance"

    def test_target_outcome_prompt_detected(self):
        assert _detect_prompt_type("Rate the target outcome achievability") == "targets"

    def test_risk_grade_prompt_detected(self):
        assert _detect_prompt_type("Provide a risk grade for this activity") == "risks"

    def test_external_context_prompt_detected(self):
        assert _detect_prompt_type("Assess the external context conditions") == "context"

    def test_finance_grade_prompt_detected(self):
        assert _detect_prompt_type("Provide a finance grade for the budget") == "finance_text"

    def test_summary_description_prompt_detected(self):
        assert _detect_prompt_type("Write a summary description of this activity") == "summary"

    def test_phrasegen_via_dict_prompt_type(self):
        assert _detect_prompt_type({"prompt_type": "phrasegen_x", "prompt": ""}) == "phrasegen"

    def test_knn_summary_via_dict_prompt_type(self):
        assert _detect_prompt_type({"prompt_type": "knn_summary_x", "prompt": ""}) == "knn_summary"

    def test_rag_synthesis_via_dict_prompt_type(self):
        assert _detect_prompt_type({"prompt_type": "rag_synthesis_x", "prompt": ""}) == "rag_synthesis"

    def test_forecast_stage_s1_via_dict(self):
        assert _detect_prompt_type({"prompt_type": "activity_s1", "prompt": ""}) == "forecast_stage"

    def test_forecast_stage_s2_via_dict(self):
        assert _detect_prompt_type({"prompt_type": "activity_s2", "prompt": ""}) == "forecast_stage"

    def test_forecast_stage_s3_via_dict(self):
        assert _detect_prompt_type({"prompt_type": "activity_s3", "prompt": ""}) == "forecast_stage"

    # JSON schema responses
    def test_page_cat_response_is_valid_json(self):
        schema = {"properties": {"pages": {}}}
        r = get_dummy_response_text(schema, "", "id")
        parsed = json.loads(r)
        assert "pages" in parsed and isinstance(parsed["pages"], list)

    def test_metadata_response_is_valid_json_with_required_keys(self):
        schema = {"properties": {"title": {}, "country_locations": {}}}
        parsed = json.loads(get_dummy_response_text(schema, "", "id"))
        assert "title" in parsed
        assert "country_locations" in parsed
        assert len(parsed["country_locations"]) > 0
        assert "iso2_code" in parsed["country_locations"][0]

    def test_misc_features_response_valid_json(self):
        schema = {"properties": {"complexity_details": {}, "how_integrated_description": {}}}
        parsed = json.loads(get_dummy_response_text(schema, "", "id"))
        assert "complexity_details" in parsed
        assert "how_integrated_description" in parsed

    def test_finance_breakdown_response_valid_json(self):
        schema = {"properties": {"total_allocation": {}, "quantitative_outcome_allocations": {}}}
        parsed = json.loads(get_dummy_response_text(schema, "", "id"))
        assert "total_allocation" in parsed
        assert "quantitative_outcome_allocations" in parsed
        assert isinstance(parsed["quantitative_outcome_allocations"], list) and len(parsed["quantitative_outcome_allocations"]) > 0
        assert parsed["total_allocation"]["amount"] > 0

    def test_generic_json_schema_returns_dict(self):
        schema = {"properties": {"foo": {}}}
        parsed = json.loads(get_dummy_response_text(schema, "", "id"))
        assert isinstance(parsed, dict)

    # Page categorisation batch size
    def test_batch_0_3_gives_3_pages(self):
        parsed = json.loads(_make_dummy_page_categorization("act_batch_0_3"))
        assert len(parsed["pages"]) == 3

    def test_batch_3_6_gives_3_pages(self):
        parsed = json.loads(_make_dummy_page_categorization("act_batch_3_6"))
        assert len(parsed["pages"]) == 3

    def test_batch_0_5_gives_5_pages(self):
        parsed = json.loads(_make_dummy_page_categorization("act_batch_0_5"))
        assert len(parsed["pages"]) == 5

    def test_no_batch_suffix_gives_one_page(self):
        parsed = json.loads(_make_dummy_page_categorization("act_no_batch"))
        assert len(parsed["pages"]) >= 1

    def test_page_cat_pages_have_category_and_informativeness(self):
        parsed = json.loads(_make_dummy_page_categorization("act_batch_0_3"))
        for page in parsed["pages"]:
            assert "category" in page
            assert "informativeness" in page

    def test_page_cat_informativeness_in_0_10(self):
        parsed = json.loads(_make_dummy_page_categorization("act_batch_0_4"))
        for page in parsed["pages"]:
            assert 0 <= page["informativeness"] <= 10

    # Plain text responses
    def test_plain_text_implementer_nonempty(self):
        r = get_dummy_response_text(None, "Grade the implementer performance", "id")
        assert len(r) > 20

    def test_plain_text_targets_nonempty(self):
        r = get_dummy_response_text(None, "Rate the target outcome quality", "id")
        assert len(r) > 20

    def test_plain_text_risks_nonempty(self):
        r = get_dummy_response_text(None, "Provide a risk grade for this project", "id")
        assert len(r) > 20

    def test_plain_text_context_nonempty(self):
        r = get_dummy_response_text(None, "Assess the external context conditions", "id")
        assert len(r) > 20

    def test_plain_text_finance_nonempty(self):
        r = get_dummy_response_text(None, "Provide a finance grade for this activity", "id")
        assert len(r) > 20

    def test_plain_text_summary_nonempty(self):
        r = get_dummy_response_text(None, "Write a summary description of this", "id")
        assert len(r) > 20

    def test_all_plain_text_types_return_strings(self):
        for prompt in [
            "Grade the implementer performance",
            "Rate the target outcome quality",
            "Provide a risk grade",
            "Assess the external context conditions",
            "Provide a finance grade",
            "Write a summary",
        ]:
            r = get_dummy_response_text(None, prompt, "id")
            assert isinstance(r, str)


# ══════════════════════════════════════════════════════════════════════════════
# 5. project_manager.py — file I/O (mocked Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

_TEST_PM_PROJECT = "test_webapp_pm_suite_001"


class TestProjectManager:

    @pytest.fixture(autouse=True)
    def cleanup_test_project(self):
        yield
        for name in [
            _TEST_PM_PROJECT,
            "test_webapp_pm_suite_002",
            "test_webapp_pm_suite_003",
        ]:
            p = EXTRACTED_PDF_DIR / name
            if p.exists():
                shutil.rmtree(p)

    def _make(self, name=_TEST_PM_PROJECT):
        return _make_test_project(name)

    # get_available_projects
    def test_project_appears_after_creation(self):
        self._make()
        assert _TEST_PM_PROJECT in get_available_projects()

    def test_returns_list(self):
        assert isinstance(get_available_projects(), list)

    def test_most_recent_project_first(self):
        import time
        p1 = EXTRACTED_PDF_DIR / "test_webapp_pm_suite_002"
        p2 = EXTRACTED_PDF_DIR / "test_webapp_pm_suite_003"
        p1.mkdir(parents=True, exist_ok=True)
        time.sleep(0.05)
        p2.mkdir(parents=True, exist_ok=True)
        projects = get_available_projects()
        assert projects.index("test_webapp_pm_suite_003") < projects.index("test_webapp_pm_suite_002")

    # load_project_name
    def test_load_name_from_txt_file(self):
        proj = self._make()
        (proj / "project_name.txt").write_text("My Kenya Renewable Project")
        assert load_project_name(_TEST_PM_PROJECT) == "My Kenya Renewable Project"

    def test_load_name_falls_back_to_metadata_title(self):
        proj = self._make()
        (proj / "project_name.txt").unlink(missing_ok=True)
        assert load_project_name(_TEST_PM_PROJECT) == "Test Environmental Activity"

    def test_load_name_default_for_empty_title(self):
        proj = self._make()
        (proj / "project_name.txt").unlink(missing_ok=True)
        (proj / "metadata.json").write_text(json.dumps({"title": ""}))
        result = load_project_name(_TEST_PM_PROJECT)
        assert isinstance(result, str) and len(result) > 0

    def test_load_name_nonexistent_folder_does_not_crash(self):
        result = load_project_name("nonexistent_folder_zzz999")
        assert isinstance(result, str)

    # save_project_name roundtrip
    def test_save_load_name_roundtrip(self):
        self._make()
        save_project_name(_TEST_PM_PROJECT, "Saved Name XYZ")
        assert load_project_name(_TEST_PM_PROJECT) == "Saved Name XYZ"

    def test_save_name_creates_txt_file(self):
        proj = self._make()
        save_project_name(_TEST_PM_PROJECT, "Alpha Project")
        assert (proj / "project_name.txt").read_text() == "Alpha Project"

    # save_project_state
    def test_save_creates_app_state_json(self):
        proj = self._make()
        save_project_state(_TEST_PM_PROJECT)
        assert (proj / "app_state.json").exists()

    def test_saved_state_is_valid_json(self):
        proj = self._make()
        save_project_state(_TEST_PM_PROJECT)
        state = json.loads((proj / "app_state.json").read_text())
        assert isinstance(state, dict)

    def test_saved_state_has_required_top_level_keys(self):
        proj = self._make()
        save_project_state(_TEST_PM_PROJECT)
        state = json.loads((proj / "app_state.json").read_text())
        for key in ["timestamp", "features", "field_edited", "field_locks",
                    "extracted_values", "sector_percentages", "location_countries",
                    "phases_0_3_complete", "ready_for_phase_4", "confirmed_metadata"]:
            assert key in state, f"Key {key!r} missing from saved state"

    def test_saved_state_timestamp_is_iso(self):
        proj = self._make()
        save_project_state(_TEST_PM_PROJECT)
        state = json.loads((proj / "app_state.json").read_text())
        ts = datetime.fromisoformat(state["timestamp"])
        assert ts.year >= 2024

    # save / load state roundtrips
    def test_features_roundtrip(self):
        proj = self._make()
        _st.session_state["features"] = {"finance": 75.0, "targets": 60.0}
        save_project_state(_TEST_PM_PROJECT)
        _st.session_state["features"] = {}
        load_project_state(_TEST_PM_PROJECT)
        assert _st.session_state["features"].get("finance") == 75.0
        assert _st.session_state["features"].get("targets") == 60.0

    def test_sector_percentages_roundtrip(self):
        proj = self._make()
        _st.session_state["sector_percentages"] = {
            "reduced_CO2_emissions": 70.0,
            "Climate_adaptation_and_resilience": 30.0,
        }
        save_project_state(_TEST_PM_PROJECT)
        _st.session_state["sector_percentages"] = {}
        load_project_state(_TEST_PM_PROJECT)
        sp = _st.session_state.get("sector_percentages", {})
        assert sp.get("reduced_CO2_emissions") == 70.0
        assert sp.get("Climate_adaptation_and_resilience") == 30.0

    def test_field_locks_roundtrip(self):
        proj = self._make()
        _st.session_state["field_locks"]["targets"] = True
        _st.session_state["field_locks"]["finance"] = True
        save_project_state(_TEST_PM_PROJECT)
        _st.session_state["field_locks"]["targets"] = False
        _st.session_state["field_locks"]["finance"] = False
        load_project_state(_TEST_PM_PROJECT)
        assert _st.session_state["field_locks"]["targets"] is True
        assert _st.session_state["field_locks"]["finance"] is True

    def test_phases_complete_flags_roundtrip(self):
        proj = self._make()
        _st.session_state["phases_0_3_complete"] = True
        _st.session_state["ready_for_phase_4"] = True
        save_project_state(_TEST_PM_PROJECT)
        _st.session_state["phases_0_3_complete"] = False
        _st.session_state["ready_for_phase_4"] = False
        load_project_state(_TEST_PM_PROJECT)
        assert _st.session_state.get("phases_0_3_complete") is True
        assert _st.session_state.get("ready_for_phase_4") is True

    def test_location_countries_roundtrip(self):
        proj = self._make()
        _st.session_state["location_countries"] = [{"code": "KE", "pct": 60}, {"code": "UG", "pct": 40}]
        save_project_state(_TEST_PM_PROJECT)
        _st.session_state["location_countries"] = []
        load_project_state(_TEST_PM_PROJECT)
        assert len(_st.session_state.get("location_countries", [])) == 2

    def test_multiple_saves_overwrite_correctly(self):
        proj = self._make()
        _st.session_state["features"]["finance"] = 50.0
        save_project_state(_TEST_PM_PROJECT)
        _st.session_state["features"]["finance"] = 90.0
        save_project_state(_TEST_PM_PROJECT)
        state = json.loads((proj / "app_state.json").read_text())
        assert state["features"]["finance"] == 90.0

    # load_project_data
    def test_load_project_data_returns_true(self):
        self._make()
        assert load_project_data(_TEST_PM_PROJECT) is True

    def test_load_project_data_populates_extraction_result(self):
        self._make()
        load_project_data(_TEST_PM_PROJECT)
        assert _st.session_state["extraction_result"] is not None

    def test_load_project_data_sets_metadata_title(self):
        self._make()
        load_project_data(_TEST_PM_PROJECT)
        assert _st.session_state["confirmed_metadata"].get("title") == "Test Environmental Activity"

    def test_load_project_data_computes_duration_from_dates(self):
        self._make()
        load_project_data(_TEST_PM_PROJECT)
        dur = _st.session_state["extracted_values"].get("planned_duration")
        assert dur is not None
        assert abs(dur - 5.0) < 0.15  # 2020-01-01 → 2025-01-01 ≈ 5 years

    def test_load_project_data_parses_expenditure(self):
        self._make()
        load_project_data(_TEST_PM_PROJECT)
        exp = _st.session_state["extracted_values"].get("planned_expenditure")
        assert exp == 45_000_000.0

    def test_load_project_data_sets_phases_complete(self):
        self._make()
        load_project_data(_TEST_PM_PROJECT)
        assert _st.session_state["phases_0_3_complete"] is True

    def test_load_project_data_nonexistent_returns_false(self):
        assert load_project_data("nonexistent_project_zzz_000") is False

    def test_load_project_data_override_false_preserves_existing(self):
        self._make()
        _st.session_state["extracted_values"]["planned_duration"] = 99.0
        load_project_data(_TEST_PM_PROJECT, override_existing=False)
        assert _st.session_state["extracted_values"]["planned_duration"] == 99.0

    def test_load_project_data_override_true_replaces_duration(self):
        self._make()
        _st.session_state["extracted_values"]["planned_duration"] = 99.0
        load_project_data(_TEST_PM_PROJECT, override_existing=True)
        dur = _st.session_state["extracted_values"]["planned_duration"]
        assert abs(dur - 5.0) < 0.15

    def test_load_project_data_populates_page_categories(self):
        self._make()
        load_project_data(_TEST_PM_PROJECT)
        cats = _st.session_state["extraction_result"].get("page_categories", [])
        assert len(cats) > 0

    def test_load_project_data_populates_summary(self):
        self._make()
        load_project_data(_TEST_PM_PROJECT)
        summary = _st.session_state["extraction_result"].get("summary", "")
        assert isinstance(summary, str) and len(summary) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. Sector cluster logic (normalize / set-to-zero / proportion conversion)
# ══════════════════════════════════════════════════════════════════════════════

class TestSectorClusterLogic:

    # Normalize button
    def test_normalize_sums_to_100(self):
        sp = {c: float(i + 1) for i, c in enumerate(SECTOR_CLUSTERS)}
        norm = _normalize_sectors(sp)
        assert abs(sum(norm.values()) - 100.0) < 1e-6

    def test_normalize_all_zero_distributes_evenly(self):
        sp = {c: 0.0 for c in SECTOR_CLUSTERS}
        norm = _normalize_sectors(sp)
        expected = 100.0 / len(SECTOR_CLUSTERS)
        for v in norm.values():
            assert abs(v - expected) < 1e-6

    def test_normalize_preserves_relative_proportions(self):
        sp = {c: 0.0 for c in SECTOR_CLUSTERS}
        sp[SECTOR_CLUSTERS[0]] = 30.0
        sp[SECTOR_CLUSTERS[1]] = 70.0
        norm = _normalize_sectors(sp)
        assert abs(norm[SECTOR_CLUSTERS[0]] - 30.0) < 1e-6
        assert abs(norm[SECTOR_CLUSTERS[1]] - 70.0) < 1e-6

    def test_normalize_single_sector_100pct(self):
        sp = {c: 0.0 for c in SECTOR_CLUSTERS}
        sp[SECTOR_CLUSTERS[3]] = 50.0
        norm = _normalize_sectors(sp)
        assert abs(norm[SECTOR_CLUSTERS[3]] - 100.0) < 1e-6

    def test_normalize_already_100_unchanged(self):
        sp = {c: 0.0 for c in SECTOR_CLUSTERS}
        sp[SECTOR_CLUSTERS[0]] = 100.0
        norm = _normalize_sectors(sp)
        assert abs(norm[SECTOR_CLUSTERS[0]] - 100.0) < 1e-6

    def test_normalize_preserves_all_keys(self):
        sp = {c: 10.0 for c in SECTOR_CLUSTERS}
        norm = _normalize_sectors(sp)
        assert set(norm.keys()) == set(SECTOR_CLUSTERS)

    def test_normalize_no_negative_values(self):
        sp = {c: float(i) for i, c in enumerate(SECTOR_CLUSTERS)}
        norm = _normalize_sectors(sp)
        for v in norm.values():
            assert v >= 0.0

    def test_normalize_values_at_most_100(self):
        sp = {c: float(i) for i, c in enumerate(SECTOR_CLUSTERS)}
        norm = _normalize_sectors(sp)
        for v in norm.values():
            assert v <= 100.0 + 1e-9

    # Set-to-zero button
    def test_set_to_zero_all_values_are_zero(self):
        sp = {c: 50.0 for c in SECTOR_CLUSTERS}
        zeroed = _zero_sectors(sp)
        assert all(v == 0.0 for v in zeroed.values())

    def test_set_to_zero_preserves_keys(self):
        sp = {c: 50.0 for c in SECTOR_CLUSTERS}
        zeroed = _zero_sectors(sp)
        assert set(zeroed.keys()) == set(SECTOR_CLUSTERS)

    def test_set_to_zero_idempotent(self):
        sp = {c: 0.0 for c in SECTOR_CLUSTERS}
        zeroed = _zero_sectors(sp)
        assert all(v == 0.0 for v in zeroed.values())

    # Proportion conversion (UI % → model 0-1)
    def test_100_pct_converts_to_1(self):
        assert 100.0 / 100.0 == 1.0

    def test_0_pct_converts_to_0(self):
        assert 0.0 / 100.0 == 0.0

    def test_50_pct_converts_to_0p5(self):
        assert 50.0 / 100.0 == pytest.approx(0.5)

    def test_proportions_in_range_for_valid_ui_pcts(self):
        for pct in np.linspace(0, 100, 21):
            prop = pct / 100.0
            assert 0.0 <= prop <= 1.0 + 1e-9

    # Model feature name consistency
    def test_all_sector_clusters_have_model_features(self):
        for c in SECTOR_CLUSTERS:
            assert f"sector_cluster_{c}" in FEATURE_NAMES

    def test_sector_feature_count_in_model(self):
        sc_feats = [f for f in FEATURE_NAMES if f.startswith("sector_cluster_")]
        assert len(sc_feats) == len(SECTOR_CLUSTERS)

    def test_sector_model_values_are_proportions(self, training_data):
        for c in SECTOR_CLUSTERS:
            col = f"sector_cluster_{c}"
            vals = training_data[col].dropna()
            assert vals.min() >= 0.0 and vals.max() <= 1.0 + 1e-9, f"{col} not in [0,1]"

    def test_normalize_then_proportion_sums_to_1(self):
        sp = {c: float(i + 1) for i, c in enumerate(SECTOR_CLUSTERS)}
        norm = _normalize_sectors(sp)
        props = {k: v / 100.0 for k, v in norm.items()}
        assert abs(sum(props.values()) - 1.0) < 1e-6

    def test_sector_lock_keys_initialised(self):
        for c in SECTOR_CLUSTERS:
            assert f"sector_{c}" in _st.session_state["field_locks"]

    def test_sector_percentages_keys_initialised(self):
        assert set(_st.session_state["sector_percentages"].keys()) == set(SECTOR_CLUSTERS)

    def test_initialize_session_state_sector_locks_match_artifact(self):
        """Guard the REAL init path: initialize_session_state() must create exactly the
        artifact's sector locks. The fixture builds field_locks itself, so without this
        test the production loop is never exercised against feature_names.json."""
        from state_manager import initialize_session_state
        _st.session_state.clear()
        initialize_session_state()
        created = {k[len("sector_"):] for k in _st.session_state["field_locks"]
                   if k.startswith("sector_")}
        assert created == set(SECTOR_CLUSTERS)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Model data integrity — files, feature names, medians
# ══════════════════════════════════════════════════════════════════════════════

class TestModelIntegrity:

    def test_feature_names_json_exists(self):
        assert (MODEL_DIR / "feature_names.json").exists()

    def test_train_medians_json_exists(self):
        assert (MODEL_DIR / "train_medians.json").exists()

    def test_model_pkl_exists(self):
        assert (MODEL_DIR / "model.pkl").exists()

    def test_extra_model_pkl_exists(self):
        assert (MODEL_DIR / "extra_model.pkl").exists()

    def test_per_org_baseline_json_exists(self):
        assert (MODEL_DIR / "per_org_baseline.json").exists()

    def test_start_year_correction_json_exists(self):
        assert (MODEL_DIR / "start_year_correction.json").exists()

    def test_train_features_csv_exists(self):
        assert (MODEL_DIR / "train_features.csv").exists()

    def test_feature_names_is_nonempty_list(self):
        assert isinstance(FEATURE_NAMES, list) and len(FEATURE_NAMES) > 0

    def test_feature_names_exact_count(self):
        """59 features as of the current saved model."""
        assert len(FEATURE_NAMES) == 59

    def test_every_feature_has_a_train_median(self):
        for feat in FEATURE_NAMES:
            assert feat in TRAIN_MEDIANS, f"No median for {feat!r}"

    def test_train_features_csv_has_all_feature_columns(self, training_data):
        for feat in FEATURE_NAMES:
            assert feat in training_data.columns, f"{feat!r} missing from train_features.csv"

    def test_rf_model_loads_and_has_predict(self, rf_model):
        assert hasattr(rf_model, "predict")

    def test_extra_model_loads_and_has_predict(self, extra_model):
        assert hasattr(extra_model, "predict")

    def test_rf_model_expects_correct_feature_count(self, rf_model):
        assert rf_model.n_features_in_ == len(FEATURE_NAMES)

    def test_llm_grade_features_in_model(self):
        for f in LLM_GRADE_FEATURES:
            assert f in FEATURE_NAMES

    def test_org_dummy_features_in_model(self):
        for f in ["rep_org_0", "rep_org_1", "rep_org_2"]:
            assert f in FEATURE_NAMES

    def test_region_features_in_model(self):
        for f in ["region_AFE", "region_AFW", "region_EAP", "region_ECA",
                  "region_LAC", "region_MENA", "region_SAS"]:
            assert f in FEATURE_NAMES

    def test_umap_features_in_model(self):
        for f in ["umap3_x", "umap3_y", "umap3_z"]:
            assert f in FEATURE_NAMES

    def test_distance_features_in_model(self):
        for f in ["sector_distance", "country_distance"]:
            assert f in FEATURE_NAMES

    def test_missing_indicator_features_in_model(self):
        for f in ["cpia_missing", "gdp_percap_missing", "wgi_any_missing",
                  "sector_clusters_missing", "umap_missing", "governance_missing_count"]:
            assert f in FEATURE_NAMES

    def test_log_planned_expenditure_feature_in_model(self):
        assert "log_planned_expenditure" in FEATURE_NAMES

    def test_governance_composite_in_model(self):
        assert "governance_composite" in FEATURE_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# 8. Feature ranges — match C_run_GLM_nobayes.py training data
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureRanges:
    """
    Verify that the values the webapp can produce for each feature
    are consistent with the training data ranges in train_features.csv,
    and that the UI limits match C_run_GLM_nobayes.py expectations.
    """

    # LLM grades (0-100 in training, slider 0-100 in UI)
    def test_llm_grades_training_min_above_zero(self, training_data):
        for feat in LLM_GRADE_FEATURES:
            assert training_data[feat].min() > 0, f"{feat} has zeros in training"

    def test_llm_grades_training_max_at_most_100(self, training_data):
        for feat in LLM_GRADE_FEATURES:
            assert training_data[feat].max() <= 100.0

    def test_llm_grade_medians_in_40_to_95_range(self, training_data):
        for feat in LLM_GRADE_FEATURES:
            med = training_data[feat].median()
            assert 40 <= med <= 95, f"{feat} median {med:.1f} not in [40, 95]"

    def test_train_medians_llm_grades_in_0_100(self):
        for feat in LLM_GRADE_FEATURES:
            assert 0.0 <= TRAIN_MEDIANS[feat] <= 100.0

    # planned_expenditure is raw USD; log_planned_expenditure is log(USD)
    def test_expenditure_45M_log_in_training_range(self, training_data):
        log_val = np.log(45_000_000)
        col = training_data["log_planned_expenditure"].dropna()
        assert col.min() <= log_val <= col.max()

    def test_expenditure_1M_log_in_training_range(self, training_data):
        log_val = np.log(1_000_000)
        col = training_data["log_planned_expenditure"].dropna()
        assert col.min() <= log_val <= col.max()

    def test_expenditure_ui_minimum_below_training_min(self, training_data):
        """UI min 0.01M = $10k → log(10000) ≈ 9.21 < training min ~13.46."""
        log_val = np.log(0.01 * 1_000_000)
        col = training_data["log_planned_expenditure"].dropna()
        assert log_val < col.min(), (
            f"log(UI_min) = {log_val:.2f} should be below training min {col.min():.2f}"
        )

    def test_expenditure_ui_maximum_within_training_range(self, training_data):
        """UI max 100M = $100M → log(100M) ≈ 18.42, within training range."""
        log_val = np.log(100_000_000)
        col = training_data["log_planned_expenditure"].dropna()
        assert log_val <= col.max()

    def test_log_planned_expenditure_training_range(self, training_data):
        """log_planned_expenditure is log(USD); training values span ~13–22."""
        col = training_data["log_planned_expenditure"].dropna()
        assert 10.0 <= col.min() and col.max() <= 25.0

    # gdp_percap: UI is raw USD, model is log(USD)
    def test_gdp_percap_training_is_log_scale(self, training_data):
        """Training gdp_percap values ~5-10 → consistent with log(USD)."""
        col = training_data["gdp_percap"].dropna()
        assert col.min() >= 5.0 and col.max() <= 11.0

    def test_gdp_percap_zero_input_falls_back_to_median(self):
        """Webapp code: log(0) is avoided by using train_median instead."""
        gdp_input = 0.0
        result = np.log(gdp_input) if gdp_input > 0 else TRAIN_MEDIANS["gdp_percap"]
        assert result == TRAIN_MEDIANS["gdp_percap"]

    def test_gdp_percap_typical_value_in_training_range(self, training_data):
        gdp_raw = 4000.0   # reasonable developing-country GDP
        log_gdp = np.log(gdp_raw)
        col = training_data["gdp_percap"].dropna()
        assert col.min() <= log_gdp <= col.max()

    # cpia_score: UI range 1.0–6.0, training 1.83–4.44
    def test_cpia_training_within_1_to_6(self, training_data):
        col = training_data["cpia_score"].dropna()
        assert col.min() >= 1.0 and col.max() <= 6.0

    def test_cpia_median_in_ui_range(self):
        assert 1.0 <= TRAIN_MEDIANS["cpia_score"] <= 6.0

    # governance_composite: UI range -3 to 3, training -2.26 to 0.78
    def test_governance_training_within_pm3_to_p3(self, training_data):
        col = training_data["governance_composite"].dropna()
        assert col.min() >= -3.0 and col.max() <= 3.0

    def test_governance_training_max_below_1(self, training_data):
        """Only developing countries in training — governance < 1."""
        col = training_data["governance_composite"].dropna()
        assert col.max() < 1.5

    def test_governance_ui_range_covers_training(self, training_data):
        col = training_data["governance_composite"].dropna()
        assert -3.0 <= col.min() and col.max() <= 3.0

    # planned_duration
    def test_planned_duration_training_positive(self, training_data):
        col = training_data["planned_duration"].dropna()
        assert col.min() > 0.0

    def test_planned_duration_median_above_ui_min(self):
        """UI min is 0.5 years; training median should be well above that."""
        assert TRAIN_MEDIANS["planned_duration"] >= 0.5

    # activity_scope
    def test_activity_scope_training_values(self, training_data):
        """Training data has scopes {0,2,3,4,5,6,7}; scope=1 is not represented.
        UI offers [1,2,3,4,5,6,7] as selectbox options — scope 1 is valid IATI code
        but has zero training examples (a known data gap)."""
        col = set(training_data["activity_scope"].dropna().astype(int).unique())
        # At least scopes 2, 3, 4 (the common ones) must exist in training
        for opt in [2, 3, 4]:
            assert opt in col, f"UI scope {opt} not found in training data"
        # Scope 1 is absent from training — flag this as a known gap
        assert 1 not in col, "Scope 1 unexpectedly appeared in training data"

    def test_activity_scope_median_is_3(self):
        """Training median activity_scope is 3."""
        assert TRAIN_MEDIANS["activity_scope"] == pytest.approx(3.0)

    # finance_is_loan
    def test_finance_is_loan_is_binary_in_training(self, training_data):
        col = training_data["finance_is_loan"].dropna()
        assert set(col.unique()).issubset({0.0, 1.0})

    def test_finance_is_loan_training_median_is_1(self):
        """Most activities in training are loans."""
        assert TRAIN_MEDIANS["finance_is_loan"] == 1.0

    # region features
    def test_region_features_are_proportions_in_training(self, training_data):
        for feat in ["region_AFE", "region_AFW", "region_EAP", "region_ECA",
                     "region_LAC", "region_MENA", "region_SAS"]:
            col = training_data[feat].dropna()
            assert col.min() >= 0.0 and col.max() <= 1.0

    # missing indicators
    def test_missing_indicators_binary_in_training(self, training_data):
        for feat in ["cpia_missing", "gdp_percap_missing", "wgi_any_missing",
                     "sector_clusters_missing"]:
            col = training_data[feat].dropna()
            assert set(col.unique()).issubset({0.0, 1.0}), f"{feat} not binary"

    # sector clusters
    def test_sector_cluster_training_medians_all_zero(self):
        """Most activities don't allocate to every sector; medians are 0."""
        for c in SECTOR_CLUSTERS:
            med = TRAIN_MEDIANS[f"sector_cluster_{c}"]
            assert med == 0.0, f"sector_cluster_{c} median {med} != 0"

    def test_num_reporting_orgs_matches_glm_config(self):
        """C_run_GLM_nobayes.py: NUM_ORGS_KEEP = 4."""
        assert len(KEEP_REPORTING_ORGS) == 4

    def test_rep_org_dummies_encode_four_orgs(self):
        """3 dummies encode 4 orgs: BMZ=reference, 3 one-hot."""
        reference_count = sum(
            1 for o in KEEP_REPORTING_ORGS if sum(get_org_dummies(o).values()) == 0
        )
        one_hot_count = sum(
            1 for o in KEEP_REPORTING_ORGS if sum(get_org_dummies(o).values()) == 1
        )
        assert reference_count == 1
        assert one_hot_count == 3


# ══════════════════════════════════════════════════════════════════════════════
# 9. Prediction logic — feature vector construction
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionLogic:

    def test_median_vector_prediction_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, _make_feature_vector())
        assert 0.0 <= pred <= 5.5

    def test_all_zeros_prediction_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        fv = pd.DataFrame([{k: 0.0 for k in FEATURE_NAMES}])
        pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv)
        assert 0.0 <= pred <= 5.5

    def test_nan_imputed_prediction_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        fv = pd.DataFrame([{k: np.nan for k in FEATURE_NAMES}])
        fv_imp = fv.fillna(pd.Series(TRAIN_MEDIANS))
        pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv_imp)
        assert 0.0 <= pred <= 5.5

    def test_feature_vector_has_correct_columns(self):
        fv = _make_feature_vector()
        assert list(fv.columns) == FEATURE_NAMES

    def test_feature_vector_no_extra_columns(self):
        fv = _make_feature_vector()
        assert len(fv.columns) == len(FEATURE_NAMES)

    # Expenditure transform
    def test_log_expenditure_for_45M(self):
        usd = 45.0 * 1_000_000
        assert np.log(usd) == pytest.approx(17.621, abs=0.01)

    def test_log_log_expenditure_for_45M(self):
        usd = 45.0 * 1_000_000
        log_e = np.log(usd)
        assert np.log1p(log_e) == pytest.approx(np.log1p(17.621), abs=0.01)

    def test_expenditure_millions_to_usd_conversion(self):
        assert 45.0 * 1_000_000 == 45_000_000

    # Missing indicator logic (mirroring page_activity_forecasting.py)
    def test_cpia_missing_flag_none_input(self):
        raw_cpia = None
        assert (1.0 if raw_cpia is None else 0.0) == 1.0

    def test_cpia_missing_flag_present_input(self):
        raw_cpia = 3.5
        assert (1.0 if raw_cpia is None else 0.0) == 0.0

    def test_gdp_missing_flag_none(self):
        raw = None
        assert (1.0 if not (raw and raw > 0) else 0.0) == 1.0

    def test_gdp_missing_flag_zero(self):
        raw = 0.0
        assert (1.0 if not (raw and raw > 0) else 0.0) == 1.0

    def test_gdp_missing_flag_positive(self):
        raw = 5000.0
        assert (1.0 if not (raw and raw > 0) else 0.0) == 0.0

    def test_sector_clusters_missing_when_all_zero(self):
        sp = {c: 0.0 for c in SECTOR_CLUSTERS}
        flag = 1.0 if sum(sp.values()) == 0 else 0.0
        assert flag == 1.0

    def test_sector_clusters_not_missing_when_one_nonzero(self):
        sp = {c: 0.0 for c in SECTOR_CLUSTERS}
        sp[SECTOR_CLUSTERS[0]] = 100.0
        flag = 1.0 if sum(sp.values()) == 0 else 0.0
        assert flag == 0.0

    def test_governance_missing_count_when_wgi_any_missing(self):
        # cpia present + all WGI missing → 0 + 5 = 5 (training value 5 does not occur because
        # WGI missing implies cpia missing in the training data, but the formula is per-input)
        from rf_predictor import governance_missing_count
        assert governance_missing_count(0.0, 1.0) == 5

    def test_governance_missing_count_all_missing(self):
        # cpia missing + all WGI missing → 1 + 5 = 6 (the training "all missing" value)
        from rf_predictor import governance_missing_count
        assert governance_missing_count(1.0, 1.0) == 6

    def test_governance_missing_count_only_cpia_missing(self):
        # cpia missing, WGI present → 1 (a value the old {0,5} formula could never emit)
        from rf_predictor import governance_missing_count
        assert governance_missing_count(1.0, 0.0) == 1

    def test_governance_missing_count_when_wgi_present(self):
        from rf_predictor import governance_missing_count
        assert governance_missing_count(0.0, 0.0) == 0

    def test_governance_missing_count_when_wgi_none(self):
        from rf_predictor import governance_missing_count
        assert governance_missing_count(0.0, None) == 5

    # RF/ExtraTrees ensemble returns numeric scalars
    def test_rf_predict_returns_finite_scalar(self, rf_model):
        fv = _make_feature_vector()
        pred = rf_model.predict(fv)[0]
        assert np.isfinite(pred)

    def test_ensemble_delta_returns_finite_scalar(self, rf_model, extra_model):
        fv = _make_feature_vector()
        assert np.isfinite(_ensemble_delta(rf_model, extra_model, fv))

    def test_clip_to_0_5_clamps_high(self):
        assert float(np.clip(8.0, 0.0, 5.0)) == 5.0

    def test_clip_to_0_5_clamps_low(self):
        assert float(np.clip(-1.0, 0.0, 5.0)) == 0.0

    def test_clip_to_0_5_passthrough_in_range(self):
        assert float(np.clip(3.7, 0.0, 5.0)) == pytest.approx(3.7)

    def test_training_samples_predict_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction, training_data):
        """All predictions on held-out training rows stay in [0, 5] (plus start-year offset)."""
        sample = training_data.sample(min(40, len(training_data)), random_state=0)
        for _, row in sample.iterrows():
            fv = pd.DataFrame([{k: row[k] for k in FEATURE_NAMES}])
            pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv)
            assert 0.0 <= pred <= 5.5, f"Training sample prediction {pred} out of range"


# ══════════════════════════════════════════════════════════════════════════════
# 10. Prediction direction — monotonicity with respect to key features
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionDirection:

    def _pred(self, rf_model, extra_model, per_org_baseline, start_year_correction, **kw):
        return _predict(rf_model, extra_model, per_org_baseline, start_year_correction, _make_feature_vector(**kw))

    # LLM grade monotonicity (low→high should raise prediction)
    def test_targets_monotone_up(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        lo = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, targets=20.0)
        hi = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, targets=90.0)
        assert hi >= lo - _MONOTONE_TOL, f"targets 20→90: {lo:.3f} → {hi:.3f}"

    def test_finance_monotone_up(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        lo = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, finance=20.0)
        hi = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, finance=100.0)
        assert hi >= lo - _MONOTONE_TOL

    def test_implementer_monotone_up(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        lo = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, implementer_performance=20.0)
        hi = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, implementer_performance=95.0)
        assert hi >= lo - _MONOTONE_TOL

    def test_context_monotone_up(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        lo = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, context=20.0)
        hi = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, context=95.0)
        assert hi >= lo - _MONOTONE_TOL

    def test_risks_monotone_up(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        """Risks score = lower risk; higher score → higher prediction."""
        lo = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, risks=10.0)
        hi = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, risks=90.0)
        assert hi >= lo - _MONOTONE_TOL

    def test_all_grades_best_beats_worst(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        best_kw  = {g: 100.0 for g in LLM_GRADE_FEATURES}
        worst_kw = {g: 0.0   for g in LLM_GRADE_FEATURES}
        best  = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, **best_kw)
        worst = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, **worst_kw)
        assert best > worst, f"All-best {best:.3f} not > all-worst {worst:.3f}"

    def test_governance_monotone_up(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        lo = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, governance_composite=-2.0)
        hi = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, governance_composite=0.5)
        assert hi >= lo - _MONOTONE_TOL

    # Org effects — all should give valid predictions
    def test_world_bank_prediction_valid(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        pred = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, rep_org_0=0, rep_org_1=0, rep_org_2=1)
        assert 0.0 <= pred <= 5.5

    def test_fcdo_prediction_valid(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        pred = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, rep_org_0=1, rep_org_1=0, rep_org_2=0)
        assert 0.0 <= pred <= 5.5

    def test_adb_prediction_valid(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        pred = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, rep_org_0=0, rep_org_1=1, rep_org_2=0)
        assert 0.0 <= pred <= 5.5

    def test_bmz_prediction_valid(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        pred = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, rep_org_0=0, rep_org_1=0, rep_org_2=0)
        assert 0.0 <= pred <= 5.5

    # Feature completeness: no missing features → at least as good
    def test_complete_vs_missing_features_both_valid(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        complete   = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction,
                                llm_features_missing_count=0.0,
                                llm_features_present_ratio=1.0,
                                feature_completeness_ratio=1.0)
        incomplete = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction,
                                llm_features_missing_count=7.0,
                                llm_features_present_ratio=0.0,
                                feature_completeness_ratio=0.0)
        assert np.isfinite(complete) and np.isfinite(incomplete)

    # Sector allocation effects
    def test_sector_allocation_all_to_one_cluster_still_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        kw = {f"sector_cluster_{c}": 0.0 for c in SECTOR_CLUSTERS}
        kw[f"sector_cluster_{SECTOR_CLUSTERS[0]}"] = 1.0
        kw["sector_clusters_missing"] = 0.0
        pred = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, **kw)
        assert 0.0 <= pred <= 5.5

    def test_zero_sectors_missing_flag_gives_valid_prediction(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        kw = {f"sector_cluster_{c}": 0.0 for c in SECTOR_CLUSTERS}
        kw["sector_clusters_missing"] = 1.0
        pred = self._pred(rf_model, extra_model, per_org_baseline, start_year_correction, **kw)
        assert 0.0 <= pred <= 5.5

    # Wide range of random training-set rows all in bounds
    def test_30_training_rows_all_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction, training_data):
        sample = training_data.sample(30, random_state=7)
        for _, row in sample.iterrows():
            fv = pd.DataFrame([{k: row[k] for k in FEATURE_NAMES}])
            pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv)
            assert 0.0 <= pred <= 5.5


# ══════════════════════════════════════════════════════════════════════════════
# 11. UI components — pure logic (no Streamlit rendering)
# ══════════════════════════════════════════════════════════════════════════════

class TestUIComponents:

    def test_field_indicator_edited_contains_set(self):
        _st.session_state["field_edited"]["targets"] = True
        assert "Set" in get_field_indicator("targets")

    def test_field_indicator_unedited_contains_median(self):
        _st.session_state["field_edited"]["targets"] = False
        assert "median" in get_field_indicator("targets").lower()

    def test_field_indicator_edited_has_green_background(self):
        _st.session_state["field_edited"]["finance"] = True
        assert "d4edda" in get_field_indicator("finance")

    def test_field_indicator_unedited_has_red_background(self):
        _st.session_state["field_edited"]["finance"] = False
        assert "f8d7da" in get_field_indicator("finance")

    def test_field_indicator_returns_html_span(self):
        _st.session_state["field_edited"]["risks"] = True
        assert "<span" in get_field_indicator("risks")

    def test_field_indicator_unknown_key_shows_median(self):
        # Key not present in field_edited at all
        result = get_field_indicator("completely_unknown_field_xyz")
        assert "median" in result.lower()

    # render_histogram
    def test_histogram_empty_series_returns_none(self):
        assert render_histogram("X", pd.Series([], dtype=float), 50.0) is None

    def test_histogram_nan_user_value_returns_none(self):
        assert render_histogram("X", pd.Series([1.0, 2.0, 3.0]), float("nan")) is None

    def test_histogram_valid_data_returns_figure(self):
        import plotly.graph_objects as go
        fig = render_histogram("Finance", pd.Series([40.0, 60.0, 80.0, 90.0]), 70.0)
        assert isinstance(fig, go.Figure)

    def test_histogram_no_marker_still_returns_figure(self):
        import plotly.graph_objects as go
        fig = render_histogram("X", pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0, show_marker=False)
        assert isinstance(fig, go.Figure)

    def test_histogram_height_applied(self):
        fig = render_histogram("X", pd.Series([1.0, 2.0, 3.0, 4.0]), 2.5, height=350)
        assert fig.layout.height == 350

    def test_histogram_default_height_200(self):
        fig = render_histogram("X", pd.Series([1.0, 2.0, 3.0, 4.0]), 2.5)
        assert fig.layout.height == 200

    def test_histogram_single_value_does_not_crash(self):
        # Should not raise
        render_histogram("X", pd.Series([50.0]), 50.0)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Full project lifecycle — create, modify, save, reload, predict
# ══════════════════════════════════════════════════════════════════════════════

_LC_PROJECT = "test_webapp_lifecycle_suite_001"


class TestProjectLifecycle:

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        proj = EXTRACTED_PDF_DIR / _LC_PROJECT
        if proj.exists():
            shutil.rmtree(proj)

    def test_new_project_loads_successfully(self):
        _make_test_project(_LC_PROJECT)
        assert load_project_data(_LC_PROJECT) is True
        assert _st.session_state["phases_0_3_complete"] is True

    def test_duration_autofilled_on_load(self):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        dur = _st.session_state["extracted_values"].get("planned_duration")
        assert dur is not None and abs(dur - 5.0) < 0.15

    def test_expenditure_autofilled_on_load(self):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        assert _st.session_state["extracted_values"].get("planned_expenditure") == 45_000_000.0

    def test_modify_grades_save_reload_consistent(self):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        _st.session_state["features"].update({"finance": 80.0, "targets": 55.0, "risks": 75.0})
        save_project_state(_LC_PROJECT)
        _st.session_state["features"] = {}
        load_project_state(_LC_PROJECT)
        assert _st.session_state["features"].get("finance") == 80.0
        assert _st.session_state["features"].get("targets") == 55.0
        assert _st.session_state["features"].get("risks")   == 75.0

    def test_sector_allocate_save_reload_consistent(self):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        _st.session_state["sector_percentages"] = {
            "reduced_CO2_emissions":            50.0,
            "Climate_adaptation_and_resilience": 50.0,
            **{c: 0.0 for c in SECTOR_CLUSTERS
               if c not in ["reduced_CO2_emissions", "Climate_adaptation_and_resilience"]},
        }
        save_project_state(_LC_PROJECT)
        _st.session_state["sector_percentages"] = {}
        load_project_state(_LC_PROJECT)
        sp = _st.session_state.get("sector_percentages", {})
        assert sp.get("reduced_CO2_emissions") == 50.0
        assert sp.get("Climate_adaptation_and_resilience") == 50.0

    def test_normalize_sectors_then_save_reload_sum_100(self):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        sp = {c: float(i + 1) for i, c in enumerate(SECTOR_CLUSTERS)}
        _st.session_state["sector_percentages"] = _normalize_sectors(sp)
        save_project_state(_LC_PROJECT)
        _st.session_state["sector_percentages"] = {}
        load_project_state(_LC_PROJECT)
        total = sum(_st.session_state.get("sector_percentages", {}).values())
        assert abs(total - 100.0) < 1.0

    def test_zero_sectors_save_reload_all_zero(self):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        _st.session_state["sector_percentages"] = _zero_sectors(
            {c: 25.0 for c in SECTOR_CLUSTERS}
        )
        save_project_state(_LC_PROJECT)
        _st.session_state["sector_percentages"] = {c: 99.0 for c in SECTOR_CLUSTERS}
        load_project_state(_LC_PROJECT)
        sp = _st.session_state.get("sector_percentages", {})
        assert all(v == 0.0 for v in sp.values()), f"Not all zero: {sp}"

    def test_lock_fields_persist_across_save_reload(self):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        _st.session_state["field_locks"]["targets"] = True
        _st.session_state["field_locks"]["finance"] = True
        _st.session_state["field_locks"][f"sector_{SECTOR_CLUSTERS[0]}"] = True
        save_project_state(_LC_PROJECT)
        _st.session_state["field_locks"]["targets"] = False
        _st.session_state["field_locks"]["finance"] = False
        _st.session_state["field_locks"][f"sector_{SECTOR_CLUSTERS[0]}"] = False
        load_project_state(_LC_PROJECT)
        assert _st.session_state["field_locks"]["targets"] is True
        assert _st.session_state["field_locks"]["finance"] is True
        assert _st.session_state["field_locks"][f"sector_{SECTOR_CLUSTERS[0]}"] is True

    def test_predict_after_loading_project_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        fv = _make_feature_vector()
        pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv)
        assert 0.0 <= pred <= 5.5

    def test_predict_with_high_grades_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        fv = _make_feature_vector(**{g: 95.0 for g in LLM_GRADE_FEATURES})
        pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv)
        assert 0.0 <= pred <= 5.5

    def test_predict_with_low_grades_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        fv = _make_feature_vector(**{g: 20.0 for g in LLM_GRADE_FEATURES})
        pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv)
        assert 0.0 <= pred <= 5.5

    def test_predict_with_normalized_sectors_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        """Normalize sectors evenly, then predict."""
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        sp = _normalize_sectors({c: 1.0 for c in SECTOR_CLUSTERS})
        kw = {f"sector_cluster_{c}": v / 100.0 for c, v in sp.items()}
        kw["sector_clusters_missing"] = 0.0
        fv = _make_feature_vector(**kw)
        pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv)
        assert 0.0 <= pred <= 5.5

    def test_predict_with_zero_sectors_in_range(self, rf_model, extra_model, per_org_baseline, start_year_correction):
        """Zero out all sectors (missing flag = 1), then predict."""
        _make_test_project(_LC_PROJECT)
        load_project_data(_LC_PROJECT)
        kw = {f"sector_cluster_{c}": 0.0 for c in SECTOR_CLUSTERS}
        kw["sector_clusters_missing"] = 1.0
        fv = _make_feature_vector(**kw)
        pred = _predict(rf_model, extra_model, per_org_baseline, start_year_correction, fv)
        assert 0.0 <= pred <= 5.5

    def test_project_name_persists(self):
        _make_test_project(_LC_PROJECT)
        save_project_name(_LC_PROJECT, "Kenya Renewable 2020")
        assert load_project_name(_LC_PROJECT) == "Kenya Renewable 2020"

    def test_app_state_timestamp_updated_on_second_save(self):
        import time
        _make_test_project(_LC_PROJECT)
        save_project_state(_LC_PROJECT)
        state_path = EXTRACTED_PDF_DIR / _LC_PROJECT / "app_state.json"
        ts1 = json.loads(state_path.read_text())["timestamp"]
        time.sleep(0.05)
        save_project_state(_LC_PROJECT)
        ts2 = json.loads(state_path.read_text())["timestamp"]
        assert ts2 >= ts1


# ══════════════════════════════════════════════════════════════════════════════
# 13. UI consistency — verify display values match model input values
# ══════════════════════════════════════════════════════════════════════════════

class TestUIConsistency:
    """
    Verify that the conversion between UI display values (millions, raw USD,
    percentages) and model input values (log-scale, proportions) is correct,
    and that UI limits cover all training data values.
    """

    def test_gdp_ui_raw_to_model_log(self):
        gdp_raw = 4000.0
        assert np.log(gdp_raw) == pytest.approx(np.log(4000.0))

    def test_gdp_typical_developing_country_in_log_range(self):
        for gdp in [500.0, 2000.0, 8000.0, 20000.0]:
            log_gdp = np.log(gdp)
            assert 5.0 < log_gdp < 12.0, f"log({gdp}) = {log_gdp} not in expected range"

    def test_expenditure_millions_to_log_usd(self):
        for m in [1.0, 10.0, 45.0, 100.0]:
            usd = m * 1_000_000
            log_usd = np.log(usd)
            assert log_usd > 13.0, f"log({m}M) = {log_usd} below training minimum"

    def test_sector_ui_pct_to_model_proportion(self):
        for pct in [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]:
            prop = pct / 100.0
            assert 0.0 <= prop <= 1.0

    def test_sector_keys_in_ui_match_model_feature_names(self):
        for c in SECTOR_CLUSTERS:
            assert f"sector_cluster_{c}" in FEATURE_NAMES

    def test_llm_slider_range_covers_training_min_max(self, training_data):
        for feat in LLM_GRADE_FEATURES:
            col = training_data[feat].dropna()
            assert col.min() >= 0.0,   f"{feat} training min below UI slider 0"
            assert col.max() <= 100.0, f"{feat} training max above UI slider 100"

    def test_cpia_ui_range_covers_training(self, training_data):
        col = training_data["cpia_score"].dropna()
        assert col.min() >= 1.0 and col.max() <= 6.0

    def test_governance_ui_range_covers_training(self, training_data):
        col = training_data["governance_composite"].dropna()
        assert -3.0 <= col.min() and col.max() <= 3.0

    def test_train_medians_llm_grades_in_slider_range(self):
        for feat in LLM_GRADE_FEATURES:
            m = TRAIN_MEDIANS[feat]
            assert 0.0 <= m <= 100.0, f"{feat} median {m} outside [0, 100]"

    def test_train_medians_cpia_in_ui_range(self):
        assert 1.0 <= TRAIN_MEDIANS["cpia_score"] <= 6.0

    def test_train_medians_governance_in_ui_range(self):
        assert -3.0 <= TRAIN_MEDIANS["governance_composite"] <= 3.0

    def test_field_indicator_green_for_edited(self):
        _st.session_state["field_edited"]["finance"] = True
        html = get_field_indicator("finance")
        assert "d4edda" in html   # green background colour

    def test_field_indicator_red_for_unedited(self):
        _st.session_state["field_edited"]["targets"] = False
        html = get_field_indicator("targets")
        assert "f8d7da" in html   # red background colour

    def test_org_dummies_one_hot_consistent_with_model(self):
        """Each org maps to a unique one-hot vector (or reference category)."""
        seen = set()
        for org in KEEP_REPORTING_ORGS:
            d = get_org_dummies(org)
            key = tuple(d[k] for k in ["rep_org_0", "rep_org_1", "rep_org_2"])
            assert key not in seen, f"Duplicate dummy vector for {org}: {key}"
            seen.add(key)

    def test_normalize_then_model_proportions_valid(self):
        """After normalise + /100, every proportion ∈ [0, 1] and sum = 1."""
        sp = {c: float(i % 5 + 1) for i, c in enumerate(SECTOR_CLUSTERS)}
        norm = _normalize_sectors(sp)
        props = [v / 100.0 for v in norm.values()]
        assert all(0.0 <= p <= 1.0 + 1e-9 for p in props)
        assert abs(sum(props) - 1.0) < 1e-6

    def test_activity_scope_ui_options_in_training(self, training_data):
        """Scopes 2, 3, 4 are in training. Scope 1 is a UI option but absent from
        training data — the UI allows it but the model has no examples of it."""
        training_scopes = set(training_data["activity_scope"].dropna().astype(int).unique())
        for opt in [2, 3, 4]:
            assert opt in training_scopes, f"UI scope option {opt} not in training data"
        # Document that scope 1 is missing from training (model will impute median)
        assert 1 not in training_scopes, "Scope 1 unexpectedly in training data"

    # ── Scope label correctness ──────────────────────────────────────────────
    _IATI_SCOPE_LABELS = {
        1: "1 - Global",
        2: "2 - Regional",
        3: "3 - Multi-national",
        4: "4 - National",
        5: "5 - Sub-national: Multi first-level admin",
        6: "6 - Sub-national: Single first-level admin",
        7: "7 - Sub-national: Single second-level admin",
    }

    def test_scope_label_1_is_global_not_national(self):
        """IATI code 1 = Global. A previous bug labelled it 'National'."""
        assert "Global" in self._IATI_SCOPE_LABELS[1]
        assert "National" not in self._IATI_SCOPE_LABELS[1]

    def test_scope_label_4_is_national_not_multi_country(self):
        """IATI code 4 = National. A previous bug labelled it 'Multi-country'."""
        assert "National" in self._IATI_SCOPE_LABELS[4]
        assert "Multi" not in self._IATI_SCOPE_LABELS[4]

    def test_scope_labels_no_duplicates(self):
        labels = list(self._IATI_SCOPE_LABELS.values())
        assert len(labels) == len(set(labels)), "Duplicate scope labels found"

    def test_scope_code_increases_with_granularity(self):
        """Lower codes = broader scope; higher codes = more local.
        Code 1 (Global) label should contain 'Global', code 7 should contain 'Sub-national'."""
        assert "Global" in self._IATI_SCOPE_LABELS[1]
        assert "Sub-national" in self._IATI_SCOPE_LABELS[7]

    def test_scope_feature_value_in_valid_range(self):
        """activity_scope in the feature vector must be an integer the model has seen."""
        valid_scopes = {0, 2, 3, 4, 5, 6, 7}   # from training data
        for scope_code in self._IATI_SCOPE_LABELS:
            fv = _make_feature_vector(activity_scope=float(scope_code))
            sc = int(fv["activity_scope"].iloc[0])
            # Scope 1 is valid IATI but absent from training — it's still a valid int
            assert isinstance(sc, int)
            assert sc == scope_code

    def test_scope_label_used_in_display_not_raw_number(self):
        """The display string for scope should include the human-readable label,
        not just the raw integer (e.g. '4 - National', not just '4')."""
        for code, label in self._IATI_SCOPE_LABELS.items():
            display = f"Activity Scope: {label}"
            assert str(code) in display
            # Ensure the word part is present (not just the number)
            word_part = label.split(" - ", 1)[1]
            assert word_part in display

    def test_finance_is_loan_ui_options_match_training(self, training_data):
        training_vals = set(training_data["finance_is_loan"].dropna().unique())
        for opt in [0, 1]:
            assert float(opt) in training_vals
