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


def main():
    with open(BASE_PATH / "model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open(BASE_PATH / "extra_model.pkl", "rb") as f:
        extra_model = pickle.load(f)
    with open(BASE_PATH / "feature_names.json", "r") as f:
        feature_cols = json.load(f)

    df = pd.read_csv(BASE_PATH / "train_features.csv")
    assert "split" in df.columns, "train_features.csv missing split column"
    train = df[df["split"] == "train"]
    assert len(train) > 0, "no train rows in train_features.csv"

    Xtr = train[feature_cols].astype(float)
    med = Xtr.median(numeric_only=True)
    Xtr_imp = Xtr.fillna(med)
    sd = Xtr_imp.std(numeric_only=True).replace(0.0, np.nan)

    imp_rf = one_sd_shift_importance(
        model=rf_model, X_train_imputed=Xtr_imp, sd=sd, feature_cols=feature_cols
    )
    imp_et = one_sd_shift_importance(
        model=extra_model, X_train_imputed=Xtr_imp, sd=sd, feature_cols=feature_cols
    )

    out = imp_rf.copy()
    out["delta_pred_1sd"] = (imp_rf["delta_pred_1sd"] + imp_et["delta_pred_1sd"]) / 2.0
    out["importance_abs_1sd"] = (imp_rf["importance_abs_1sd"] + imp_et["importance_abs_1sd"]) / 2.0
    out["importance_abs_1sd"] = out["importance_abs_1sd"].where(out["importance_abs_1sd"] > 1e-12, 0.0)

    out = (
        out.reset_index()
        .rename(columns={"importance_abs_1sd": "importance"})
        .sort_values("importance", ascending=False)
    )

    missingness = (df[feature_cols].isna().mean() * 100.0).to_dict()
    out["missingness_pct"] = out["feature"].map(missingness)

    dest = BASE_PATH / "feature_importances.csv"
    out.to_csv(dest, index=False)
    print(f"Wrote {dest} ({len(out)} features)")


if __name__ == "__main__":
    main()
