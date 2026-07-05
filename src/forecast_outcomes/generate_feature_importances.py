"""
Regenerate data/rating_model_outputs/feature_importances.csv from the saved
RF + ExtraTrees ensemble artifacts.

The webapp Model Performance page consumes columns: feature, importance,
delta_pred_1sd. These come from a model-based 1-SD local shift importance
evaluated at the TRAIN median point; no data pipeline or ablation is needed.

Run:
    micromamba run -n py311 python src/forecast_outcomes/generate_feature_importances.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).resolve().parents[2] / "data" / "rating_model_outputs"


def one_sd_shift_importance(*, model, X_train_imputed, sd, feature_cols):
    x0 = X_train_imputed.median(numeric_only=True).reindex(feature_cols).to_frame().T
    pred0 = float(model.predict(x0)[0])

    rows = []
    for f in feature_cols:
        s = float(sd.get(f, np.nan))
        if not np.isfinite(s):
            continue
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[f] = float(x_plus[f].iloc[0]) + s
        x_minus[f] = float(x_minus[f].iloc[0]) - s
        delta = (float(model.predict(x_plus)[0]) - float(model.predict(x_minus)[0])) / 2.0
        rows.append({
            "feature": f,
            "sd_train": s,
            "pred0": pred0,
            "delta_pred_1sd": delta,
            "importance_abs_1sd": abs(delta),
        })
    return pd.DataFrame(rows).set_index("feature")
