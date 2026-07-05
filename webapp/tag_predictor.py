"""Predict outcome tag probabilities for a single activity.

Works with the pkl produced for outcome tag forecasting.
"""

from __future__ import annotations

import pandas as pd


def _build_input(model_dict: dict, feat_key: str, feature_vector_imputed: pd.DataFrame,
                 train_medians: dict) -> pd.DataFrame:
    """Build a single-row DataFrame for the given feature list key."""
    feat_cols = model_dict[feat_key]
    row = {}
    for col in feat_cols:
        if col in feature_vector_imputed.columns:
            val = feature_vector_imputed.iloc[0][col]
            row[col] = float(val) if not pd.isna(val) else float(train_medians.get(col, 0.0))
        else:
            row[col] = float(train_medians.get(col, 0.0))
    return pd.DataFrame([row], columns=feat_cols)


def predict_outcome_tags(
    tag_models_data: dict,
    feature_vector_imputed: pd.DataFrame,
) -> dict[str, float]:
    """Return {tag_name: probability} for all outcome tags.

    Args:
        tag_models_data: dict loaded from tag_models.pkl
        feature_vector_imputed: single-row DataFrame (from impute_and_run_statistical_model)

    Returns:
        dict mapping tag name → predicted probability [0, 1].
        Empty dict if tag_models_data is None.
    """
    if tag_models_data is None:
        return {}

    models: dict = tag_models_data["models"]
    global_feature_cols: list[str] = tag_models_data["feature_cols"]
    train_medians: dict = tag_models_data["train_medians"]

    # Ensure global features have a fallback row
    _global_row = {}
    for col in global_feature_cols:
        if col in feature_vector_imputed.columns:
            val = feature_vector_imputed.iloc[0][col]
            _global_row[col] = float(val) if not pd.isna(val) else float(train_medians.get(col, 0.0))
        else:
            _global_row[col] = float(train_medians.get(col, 0.0))

    results: dict[str, float] = {}

    for tag, model_dict in models.items():
        if not model_dict:
            continue  # legacy empty dict

        # ---- const_base ----
        if "base_rate" in model_dict and "rf" not in model_dict and "ridge" not in model_dict:
            results[tag] = float(model_dict["base_rate"])
            continue

        feat_key = "feature_cols" if "feature_cols" in model_dict else None

        # ---- RF + ET + Ridge average: ((RF+ET)/2 + Ridge) / 2 ----
        if "rf" in model_dict and "extra" in model_dict and "ridge" in model_dict and "ridge_feat" in model_dict:
            X_rf = _build_input(model_dict, "feature_cols", feature_vector_imputed, train_medians)
            X_ridge = _build_input(model_dict, "ridge_feat", feature_vector_imputed, train_medians)
            rf_et_prob = (model_dict["rf"].predict_proba(X_rf)[0, 1] +
                          model_dict["extra"].predict_proba(X_rf)[0, 1]) / 2.0
            ridge_prob = float(model_dict["ridge"].predict_proba(X_ridge.to_numpy(dtype=float))[0, 1])
            results[tag] = float((rf_et_prob + ridge_prob) / 2.0)

        # ---- RF + ET ensemble ----
        elif "rf" in model_dict and "extra" in model_dict:
            X_rf = _build_input(model_dict, "feature_cols", feature_vector_imputed, train_medians)
            rf_prob = model_dict["rf"].predict_proba(X_rf)[0, 1]
            et_prob = model_dict["extra"].predict_proba(X_rf)[0, 1]
            results[tag] = float((rf_prob + et_prob) / 2.0)

        # ---- Ridge only ----
        elif "ridge" in model_dict:
            X_ridge = _build_input(model_dict, "feature_cols", feature_vector_imputed, train_medians)
            results[tag] = float(model_dict["ridge"].predict_proba(X_ridge.to_numpy(dtype=float))[0, 1])

        # ---- RF only ----
        elif "rf" in model_dict:
            X_rf = _build_input(model_dict, "feature_cols", feature_vector_imputed, train_medians)
            results[tag] = float(model_dict["rf"].predict_proba(X_rf)[0, 1])

    return results


def get_model_type_label(model_dict: dict) -> str:
    """Return a short display label for the model type."""
    if "base_rate" in model_dict and "rf" not in model_dict and "ridge" not in model_dict:
        return "base rate"
    if "rf" in model_dict and "extra" in model_dict and "ridge" in model_dict:
        return "RF+ET+Ridge"
    if "rf" in model_dict and "extra" in model_dict:
        return "RF+ET"
    if "ridge" in model_dict:
        return "Ridge"
    if "rf" in model_dict:
        return "RF"
    return ""
