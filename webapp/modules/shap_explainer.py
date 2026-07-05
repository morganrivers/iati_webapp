"""
Feature contribution explainer for Activity Predictions.

Uses tree_contributions (a minimal treeinterpreter replacement) to decompose
each RF prediction into per-feature contributions. The public API is identical
to the old shap version so the rest of the webapp is unchanged.
"""
import numpy as np
from modules.tree_contributions import predict as ti_predict


def compute_shap_values(rf_model, extra_model, feature_vector_imputed, feature_names):
    """
    Compute per-feature contributions for a single prediction.

    Averages contributions from both RF and ExtraTrees so the values are
    consistent with the ensemble prediction shown in the UI.

    Returns dict with:
        - 'base_value': Ensemble baseline (mean of RF and ET root-node values)
        - 'shap_values': Array of averaged per-feature contributions
        - 'feature_names': List of feature names
        - 'feature_values': Array of feature values used
        - 'prediction': Ensemble prediction from the two models
    """
    X = feature_vector_imputed.values
    pred_rf, bias_rf, contribs_rf = ti_predict(rf_model, X)
    pred_et, bias_et, contribs_et = ti_predict(extra_model, X)

    return {
        'base_value': float((bias_rf[0] + bias_et[0]) / 2),
        'shap_values': ((contribs_rf[0] + contribs_et[0]) / 2).astype(float),
        'feature_names': feature_names,
        'feature_values': feature_vector_imputed.iloc[0].values,
        'prediction': float((pred_rf[0] + pred_et[0]) / 2),
    }


def get_sorted_contributions(shap_result):
    """
    Sort features by absolute contribution magnitude.

    Returns dict with:
        - 'feature_names': sorted by |contribution|
        - 'shap_values': signed contributions in sorted order
        - 'feature_values': feature values in same sorted order
        - 'abs_shap_values': absolute contributions
    """
    shap_vals     = shap_result['shap_values']
    feature_names = shap_result['feature_names']
    feature_vals  = shap_result['feature_values']

    sort_idx = np.argsort(-np.abs(shap_vals))

    return {
        'feature_names':  [feature_names[i] for i in sort_idx],
        'shap_values':    shap_vals[sort_idx],
        'feature_values': feature_vals[sort_idx],
        'abs_shap_values': np.abs(shap_vals[sort_idx]),
    }
