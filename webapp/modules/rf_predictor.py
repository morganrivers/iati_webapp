import numpy as np
import pandas as pd

DEFAULT_START_YEAR = 2020


def _resolve_start_year(value):
    if value is None:
        return DEFAULT_START_YEAR
    if hasattr(value, "year"):
        return int(value.year)
    try:
        return int(str(value)[:4])
    except (ValueError, TypeError):
        return DEFAULT_START_YEAR


def governance_missing_count(cpia_missing, wgi_any_missing):
    """Missing count over {cpia_score} + 5 WGI columns, matching the training definition
    in explain_single_prediction.py (isna sum → values 0..6). WGI columns are all-or-nothing
    per country, so wgi_any_missing==1 contributes the full 5. Single-sourced for the predictor
    and its tests."""
    wgi_missing = 1.0 if (wgi_any_missing is None or wgi_any_missing) else 0.0
    return float(cpia_missing) + (5.0 if wgi_missing else 0.0)


def _active_rep_org_col(org_dummies):
    for col in ('rep_org_0', 'rep_org_1', 'rep_org_2'):
        if org_dummies.get(col) == 1:
            return col
    return '__overall__'


def _ensemble_delta(rf_model, extra_model, feature_vector_imputed):
    """Mean of the Random Forest and ExtraTrees residual predictions."""
    return (rf_model.predict(feature_vector_imputed)[0]
            + extra_model.predict(feature_vector_imputed)[0]) / 2.0


def predict_rating(base, ens_delta, start_year_correction, start_year):
    """Final rating = clip(base + ensemble delta, 0, 5) + start-year drift correction.

    Shared by the production path and the test suite so the formula stays single-sourced.
    """
    prediction = float(np.clip(base + ens_delta, 0.0, 5.0))
    return prediction + start_year_correction['intercept'] + start_year_correction['slope'] * start_year


def impute_and_run_statistical_model(rf_model, extra_model, per_org_baseline, start_year_correction,
                                     model_metadata, reporting_org, gdp_percap_input, fields_edited,
                                     session_state_values, location_features):
    """Compute the per-org-mode + RF/ExtraTrees delta ensemble prediction from widget state.

    session_state_values: dict with input_* widget keys (st.session_state in the webapp, or
                          app_state["widget_state"] merged with sector_percentages and
                          embedding_results for run_rag_forecast.py).
    location_features:    dict from extract_features_from_location() containing gdp_percap,
                          cpia_score, governance_composite, wgi_any_missing, region_* keys.
    """
    from location_features import get_org_dummies
    org_dummies = get_org_dummies(reporting_org)

    train_medians = model_metadata["train_medians"]

    # Expenditure: widget stores millions, model needs raw USD and log(USD)
    # New convention (after training script change):
    #   planned_expenditure      = raw USD
    #   log_planned_expenditure  = log(USD)
    # Old convention was: planned_expenditure=log(USD), log_planned_expenditure=log1p(log(USD))
    planned_exp_millions = session_state_values.get('input_planned_expenditure') or 0
    planned_expenditure = planned_exp_millions * 1_000_000
    log_planned_expenditure = np.log(planned_expenditure) if planned_expenditure > 0 else 0
    logger.debug(f"planned_expenditure (raw USD) = {planned_expenditure:.4g}, log = {log_planned_expenditure:.4f}")

    # expenditure_per_year_log: log(raw_USD / duration) when both are meaningful
    planned_duration_raw = session_state_values.get('input_planned_duration', train_medians.get("planned_duration", 3.0))
    if planned_expenditure >= 100_000 and planned_duration_raw is not None and planned_duration_raw >= 1:
        expenditure_per_year_log = np.log(planned_expenditure / planned_duration_raw)
    else:
        expenditure_per_year_log = train_medians.get("expenditure_per_year_log", 0.0)
    logger.debug(f"expenditure_per_year_log = {expenditure_per_year_log:.4f}")

    # GDP: raw USD passed as explicit param (from input_gdp_percap widget)
    gdp_percap = np.log(gdp_percap_input) if gdp_percap_input and gdp_percap_input > 0 else train_medians["gdp_percap"]
    gdp_percap_missing = 0.0 if (gdp_percap_input and gdp_percap_input > 0) or fields_edited.get('gdp_percap', False) else 1.0

    # Location widget values
    cpia_score = session_state_values.get('input_cpia_score')
    governance_composite = session_state_values.get('input_governance_composite')
    wgi_any_missing = session_state_values.get('input_wgi_any_missing')
    if cpia_score is None:
        cpia_score = train_medians.get("cpia_score")
    if governance_composite is None:
        governance_composite = train_medians.get("governance_composite")
    if wgi_any_missing is None:
        wgi_any_missing = train_medians.get("wgi_any_missing", 0)

    # Activity metadata
    activity_scope = session_state_values.get('input_activity_scope')
    finance_is_loan = session_state_values.get('input_finance_is_loan', 0)
    planned_duration = session_state_values.get('input_planned_duration', train_medians.get("planned_duration", 3.0))

    # LLM grades
    finance = session_state_values.get('input_finance', train_medians["finance"])
    integratedness = session_state_values.get('input_integratedness', train_medians["integratedness"])
    implementer_performance = session_state_values.get('input_implementer_performance', train_medians["implementer_performance"])
    targets = session_state_values.get('input_targets', train_medians["targets"])
    context = session_state_values.get('input_context', train_medians["context"])
    risks = session_state_values.get('input_risks', train_medians["risks"])
    complexity = session_state_values.get('input_complexity', train_medians["complexity"])

    # Embedding results (sub-dict in both st.session_state and merged app_state dict)
    emb = session_state_values.get('embedding_results') or {}
    umap3_x = emb.get('umap3_x', train_medians.get("umap3_x", 0.0))
    umap3_y = emb.get('umap3_y', train_medians.get("umap3_y", 0.0))
    umap3_z = emb.get('umap3_z', train_medians.get("umap3_z", 0.0))
    country_distance = emb.get('country_distance', train_medians.get("country_distance", 0.0))
    sector_distance = emb.get('sector_distance', train_medians.get("sector_distance", 0.0))

    # Sector percentages (sub-dict in both st.session_state and merged app_state dict)
    sector_percentages = session_state_values.get('sector_percentages') or {}

    # Missingness for governance features. Training defines governance_missing_count as the
    # isna count over {cpia_score} + 5 WGI columns (see explain_single_prediction.py), so its
    # values are 0..6. WGI columns are all-or-nothing per country, hence wgi_any_missing==1
    # contributes the full 5. Emitting the old {0,5} skipped cpia and never matched training.
    cpia_missing = 0.0 if session_state_values.get('input_cpia_score') is not None else 1.0
    gov_missing_count = governance_missing_count(cpia_missing, wgi_any_missing)

    # Sector clusters (convert 0-100 % to 0-1 proportion). Names sourced from the
    # model artifact so they stay single-sourced with the UI and extractor.
    from model_loader import get_sector_clusters
    sector_cluster_features = {
        f'sector_cluster_{c}': sector_percentages.get(c, 0.0) / 100.0
        for c in get_sector_clusters()
    }

    all_features = {
        # LLM grades
        'finance': finance,
        'integratedness': integratedness,
        'implementer_performance': implementer_performance,
        'targets': targets,
        'context': context,
        'risks': risks,
        'complexity': complexity,
        # Activity metadata
        'activity_scope': activity_scope,
        'finance_is_loan': finance_is_loan,
        'planned_duration': planned_duration,
        'planned_expenditure': planned_expenditure,       # raw USD (new convention)
        'log_planned_expenditure': log_planned_expenditure,  # log(USD) (new convention)
        'expenditure_per_year_log': expenditure_per_year_log,
        'expenditure_x_complexity': planned_expenditure * complexity,  # raw USD × complexity grade
        # Location
        'gdp_percap': gdp_percap,
        'cpia_score': cpia_score,
        'governance_composite': governance_composite,
        'wgi_any_missing': wgi_any_missing,
        # Regions (from location_features param)
        'region_AFE': location_features.get('region_AFE', 0.0),
        'region_AFW': location_features.get('region_AFW', 0.0),
        'region_EAP': location_features.get('region_EAP', 0.0),
        'region_ECA': location_features.get('region_ECA', 0.0),
        'region_LAC': location_features.get('region_LAC', 0.0),
        'region_MENA': location_features.get('region_MENA', 0.0),
        'region_SAS': location_features.get('region_SAS', 0.0),
        # Org dummies
        'rep_org_0': org_dummies['rep_org_0'],
        'rep_org_1': org_dummies['rep_org_1'],
        'rep_org_2': org_dummies['rep_org_2'],
        # UMAP + distances
        'umap3_x': umap3_x,
        'umap3_y': umap3_y,
        'umap3_z': umap3_z,
        'country_distance': country_distance,
        'sector_distance': sector_distance,
        **sector_cluster_features,
        # Missing indicators
        'cpia_missing': cpia_missing,
        'gdp_percap_missing': gdp_percap_missing,
        'planned_expenditure_missing': 0.0,
        'planned_duration_missing': 0.0,
        'sector_clusters_missing': 1.0
            if any(v is None for v in sector_percentages.values())
               or sum(v or 0 for v in sector_percentages.values()) == 0
            else 0.0,
        'umap_missing': 0.0,
        'governance_missing_count': gov_missing_count,
        # Completeness metrics
        'llm_features_missing_count': 0.0,
        'llm_features_present_ratio': 1.0,
        'feature_completeness_ratio': 1.0,
    }

    # ── DEBUG: check what the model expects vs what all_features provides ──────
    expected_features = model_metadata['feature_names']

    # Features the model needs but are NOT in all_features (will use train_median fallback)
    missing_from_all = [f for f in expected_features if f not in all_features]
    # Features computed in all_features but NOT used by the model (silently dropped)
    extra_in_all = [f for f in all_features if f not in expected_features]

    if missing_from_all:
        logger.warning(
            "FEATURES MISSING FROM all_features — USING TRAIN MEDIAN FALLBACK: %s",
            {f: train_medians.get(f, 0.0) for f in missing_from_all},
        )
    else:
        logger.debug("All expected features present in all_features. No fallbacks needed.")

    if extra_in_all:
        logger.debug(
            "all_features contains keys NOT used by the model (silently dropped): %s",
            {f: all_features[f] for f in extra_in_all},
        )

    # Reorder to match training feature order
    ordered_features = {fname: all_features.get(fname, train_medians.get(fname, 0.0)) for fname in model_metadata['feature_names']}
    feature_vector = pd.DataFrame([ordered_features])
    feature_vector_imputed = feature_vector.fillna(pd.Series(train_medians))

    active_rep_org_col = _active_rep_org_col(org_dummies)
    base = per_org_baseline.get(active_rep_org_col, per_org_baseline['__overall__'])

    ens_delta = _ensemble_delta(rf_model, extra_model, feature_vector_imputed)
    start_year = _resolve_start_year(session_state_values.get('input_start_date'))
    prediction = predict_rating(base, ens_delta, start_year_correction, start_year)

    return feature_vector, feature_vector_imputed, base, ens_delta, prediction
