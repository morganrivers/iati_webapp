"""explain_single_prediction.py

Picks a random validation-set activity and prints a verbal explanation
of why the model made its prediction.

Usage:
    python explain_single_prediction.py            # random val activity
    python explain_single_prediction.py <activity_id>
"""

import json, pickle, sys, random
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# MODE FLAG
# ============================================================
# True  → concise: only duration, expenditure, qualifying LLM/CPIA/GDP drivers
#         no interactions, no LLM-correction section, ends with one-sentence summary
# False → original full output (interactions, LLM-correction section, all top-7 features)
CONCISE = False

# ============================================================
# PATHS  (relative to src/forecast_outcomes/)
# ============================================================
MODEL_PATH       = Path("rating_model_outputs/model.pkl")
META_PATH        = Path("rating_model_outputs/model_metadata.json")
PREDS_PATH       = Path("rating_model_outputs/predictions.csv")   # y_pred = pred_rf_llm_modded
RF_PREDS_PATH    = Path("../../data/random_forest_predictions.csv")
OUTCOMES_CACHE   = Path("prediction_outcome_cached_input_data.pkl")
INFO_CSV         = Path("../../data/info_for_activity_forecasting_old_transaction_types.csv")

LLM_VARIANT_PATHS = [
    Path("../../data/rag_prompts_and_responses/outputs_exactly_like_halawi_et_al_rag_added_deepseek_s3_call_1.jsonl"),
    Path("../../data/rag_prompts_and_responses/outputs_exactly_like_halawi_et_al_rag_added_deepseek_with_stages_s3_call_1.jsonl"),
    Path("../../data/rag_prompts_and_responses/outputs_exactly_like_halawi_et_al_rag_added_gemini3pro_val_s3_call_1.jsonl"),
]

KEEP_REPORTING_ORGS = [
    "UK - Foreign, Commonwealth Development Office (FCDO)",
    "Asian Development Bank",
    "World Bank",
    "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); "
    "Federal Ministry for Economic Cooperation and Development (BMZ)",
]
NUM_ORGS_KEEP = 4

# ---- import LLM forecast parser from utils (same setup as main script) ----
UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from helpers_for_ratings_and_final_activity_features import parse_last_line_label_after_forecast


# ============================================================
# CONCISE-MODE CONSTANTS
# ============================================================
LLM_GRADE_FEATS = {'finance', 'integratedness', 'implementer_performance',
                   'targets', 'context', 'risks', 'complexity'}
# features beyond duration+expenditure that are allowed in concise output (if |SHAP|>0.05)
CONCISE_CONDITIONAL = LLM_GRADE_FEATS | {'cpia_score', 'gdp_percap'}

_CONCISE_LABEL = {
    "planned_duration":         "the planned duration",
    "expenditure":              "how much money the project has planned to be committed from financers",
    "finance":                  "how well financed the project is",
    # "integratedness":           "how integrated the project is into a bro",
    "implementer_performance":  "how skilled those in the implementing organization are",
    "targets":                  None,  # direction-dependent: see _short_phrase
    "context":                  None,  # direction-dependent: see _short_phrase
    "risks":                    "the level of risk identified in the risk assessment",
    "complexity":               "the level of complexity of the project",
    "cpia_score":               "CPIA score",
    "gdp_percap":               "GDP per capita",
}


# ============================================================
# FEATURE RECONSTRUCTION
# ============================================================
def reconstruct_features():
    """Rebuild all 57 RF features from the outcomes-model cache + CSV."""
    with open(OUTCOMES_CACHE, 'rb') as f:
        data = pickle.load(f)['data'].copy()

    # planned_duration + reporting_orgs from the main CSV
    dates_df = pd.read_csv(INFO_CSV, usecols=[
        "activity_id", "reporting_orgs",
        "original_planned_start_date", "original_planned_close_date",
    ])
    for col in ["original_planned_start_date", "original_planned_close_date"]:
        dates_df[col] = pd.to_datetime(dates_df[col], errors="coerce")
    dates_df["planned_duration"] = (
        dates_df["original_planned_close_date"] - dates_df["original_planned_start_date"]
    ).dt.days / 365.25
    dates_df = dates_df.set_index("activity_id")
    data = data.join(dates_df[["planned_duration", "reporting_orgs"]], how="left")

    # rep_org one-hot dummies (same vocab as main script)
    BMZ_B = ("Federal Ministry for Economic Cooperation and Development (BMZ); "
             "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ)")
    BMZ_A = ("Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); "
             "Federal Ministry for Economic Cooperation and Development (BMZ)")
    vocab = {org: i for i, org in enumerate(KEEP_REPORTING_ORGS)}
    s = data["reporting_orgs"].fillna("").astype(str).str.strip().replace(BMZ_B, BMZ_A)
    idx_series = s.map(vocab)
    ohe = pd.get_dummies(idx_series, dtype=int)
    ohe = ohe.reindex(columns=range(NUM_ORGS_KEEP), fill_value=0)
    ohe.columns = [f"rep_org_{i}" for i in range(NUM_ORGS_KEEP)]
    data = pd.concat([data, ohe], axis=1)

    # governance_composite = mean of WGI columns
    wgi_cols = ['wgi_control_of_corruption_est', 'wgi_political_stability_est',
                'wgi_government_effectiveness_est', 'wgi_regulatory_quality_est',
                'wgi_rule_of_law_est']
    data['governance_composite'] = data[wgi_cols].mean(axis=1)

    # log_planned_expenditure
    data['log_planned_expenditure'] = np.log1p(np.maximum(data['planned_expenditure'].fillna(0), 0))

    # missingness indicators (same logic as add_enhanced_uncertainty_features)
    llm_feats = ['finance', 'integratedness', 'implementer_performance',
                 'targets', 'context', 'risks', 'complexity']
    data['llm_features_missing_count']  = data[llm_feats].isna().sum(axis=1).astype(float)
    data['llm_features_present_ratio']  = 1.0 - data['llm_features_missing_count'] / len(llm_feats)
    data['feature_completeness_ratio']  = data['llm_features_present_ratio']

    gov_feats = ['cpia_score'] + wgi_cols
    data['governance_missing_count']     = data[gov_feats].isna().sum(axis=1).astype(float)
    data['cpia_missing']                 = data['cpia_score'].isna().astype(float)

    sc_cols = [c for c in data.columns if c.startswith('sector_cluster_')]
    present = data[sc_cols].notna().any(axis=1) | (data[sc_cols].sum(axis=1) > 0)
    data['sector_clusters_missing']      = (~present).astype(float)
    data['gdp_percap_missing']           = data['gdp_percap'].isna().astype(float)
    data['planned_expenditure_missing']  = data['planned_expenditure'].isna().astype(float)
    data['planned_duration_missing']     = data['planned_duration'].isna().astype(float)
    data['wgi_any_missing']              = data[wgi_cols].isna().any(axis=1).astype(float)
    data['umap_missing']                 = data[['umap3_x', 'umap3_y', 'umap3_z']].isna().any(axis=1).astype(float)

    return data


# ============================================================
# LLM PREDICTION LOADING  (only used when CONCISE = False)
# ============================================================
def load_llm_predictions():
    """Load and average LLM forecast predictions across the active variant JNLs."""
    variant_series = []
    for path in LLM_VARIANT_PATHS:
        if not path.exists():
            print(f"  (skipping {path.name}: not found)")
            continue
        preds = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                aid = rec.get("activity_id")
                if not aid:
                    continue
                # extract content (ChatGPT-style or plain)
                content = None
                resp = rec.get("response")
                if isinstance(resp, dict):
                    content = resp.get("content") or resp.get("text")
                if not content:
                    content = rec.get("response_text")
                if not content:
                    continue
                val = parse_last_line_label_after_forecast(content, rec)
                if val is not None:
                    preds.setdefault(aid, []).append(val)
        # average duplicates within variant
        s = pd.Series({k: np.mean(v) for k, v in preds.items()})
        if len(s):
            variant_series.append(s)

    if not variant_series:
        return pd.Series(dtype=float)
    df = pd.concat(variant_series, axis=1)
    return df.mean(axis=1)


# ============================================================
# VERBAL DESCRIPTION HELPERS  (shared / verbose-mode)
# ============================================================
FEAT_DESC = {
    "finance":                  "financial planning quality (LLM-graded)",
    "integratedness":           "activity integration quality (LLM-graded)",
    "implementer_performance":  "implementer track record (LLM-graded)",
    "targets":                  "target-setting quality (LLM-graded)",
    "context":                  "contextual analysis quality (LLM-graded)",
    "risks":                    "risk assessment quality (LLM-graded)",
    "complexity":               "activity complexity (LLM-graded)",
    "activity_scope":           "activity scope (numeric code)",
    "gdp_percap":               "host-country GDP per capita",
    "cpia_score":               "CPIA institutional-quality score",
    "planned_duration":         "planned duration (years)",
    "planned_expenditure":      "planned expenditure (normalized)",
    "log_planned_expenditure":  "log planned expenditure",
    "governance_composite":     "average WGI governance score",
    "sector_distance":          "semantic distance to typical sector",
    "country_distance":         "semantic distance to typical country",
    "umap3_x":                  "UMAP embedding dim 1",
    "umap3_y":                  "UMAP embedding dim 2",
    "umap3_z":                  "UMAP embedding dim 3",
    "finance_is_loan":          "financing is a loan (vs. disbursement)",
}

# short names for the 4 reporting orgs
_ORG_SHORT = ["FCDO (UK)", "Asian Dev Bank", "World Bank", "BMZ (Germany)"]


def _desc(name):
    if name in FEAT_DESC:
        return FEAT_DESC[name]
    if name.startswith("rep_org_"):
        i = int(name.split("_")[-1])
        return f"org dummy: {_ORG_SHORT[i]}" if i < len(_ORG_SHORT) else name
    if name.startswith("region_"):
        return f"region: {name[7:]}"
    if name.startswith("sector_cluster_"):
        return f"sector cluster: {name[15:].replace('_', ' ')}"
    if name.endswith("_missing"):
        return f"missingness flag: {name[:-8].replace('_', ' ')}"
    if "missing" in name or "completeness" in name or "present" in name:
        return name.replace("_", " ")
    return name


def _fmt(v):
    if abs(v) < 0.005:
        return "0.00"
    if abs(v) > 10000:
        return f"{v:,.0f}"
    return f"{v:.2f}"


# ============================================================
# CONCISE-MODE HELPERS
# ============================================================
def _strength(abs_sv, max_abs):
    frac = abs_sv / max_abs if max_abs > 0 else 0
    if   frac > 0.9: return "strongly by"
    elif frac > 0.6: return "moderately by"
    elif frac > 0.3: return "somewhat by"
    else:            return "weakly by"

def _short_phrase(name, shap_val, val, strength=""):
    """One short clause for the summary sentence.  Only LLM grades show their value."""
    label = _CONCISE_LABEL.get(name, name)
    if name == "targets":
        label = "how easy the activity objectives are to achieve" if shap_val > 0 else "how ambitious the activity objectives are to achieve"
    elif name == "context":
        label = "the challenge of the activity context" if shap_val < 0 else "the ease of the activity context"
    if name == "planned_duration":
        label = f"{label} ({_fmt(val)} yr)"
    prefix = f"{strength} " if strength else ""
    return f"{prefix}{label}"

def _join_phrases(phrases):
    if len(phrases) == 1: return phrases[0]
    if len(phrases) == 2: return phrases[0] + " and " + phrases[1]
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


# ============================================================
# WEBAPP INTERFACE FUNCTION
# ============================================================
def generate_explanation_for_webapp(rf_model, feature_vector_imputed, feature_names, final_pred):
    """
    Generate a concise explanation for a prediction in the webapp.

    Args:
        rf_model: Trained RandomForest model
        feature_vector_imputed: DataFrame with 1 row containing imputed feature values (in correct order)
        feature_names: List of feature names (must match columns in feature_vector_imputed)
        final_pred: Final prediction value

    Returns:
        str: Concise explanation text
    """
    import sys as _sys, pathlib as _pl
    _webapp = _pl.Path(__file__).resolve().parents[2] / "webapp"
    if str(_webapp) not in _sys.path:
        _sys.path.insert(0, str(_webapp))
    from modules.tree_contributions import predict as _ti_predict
    _, _, _contribs = _ti_predict(rf_model, feature_vector_imputed.values)
    shap_vals = _contribs[0].astype(float)

    # --- combine expenditure SHAP ---
    exp_idx     = feature_names.index("planned_expenditure")
    log_exp_idx = feature_names.index("log_planned_expenditure")
    dur_idx     = feature_names.index("planned_duration")

    combined_exp_shap = float(shap_vals[exp_idx] + shap_vals[log_exp_idx])
    dur_shap          = float(shap_vals[dur_idx])

    # always-shown: (name, shap, display_value)
    effects = [
        ("planned_duration", dur_shap,          float(feature_vector_imputed.iloc[0, dur_idx])),
        ("expenditure",      combined_exp_shap, float(feature_vector_imputed.iloc[0, exp_idx])),
    ]

    # conditional: LLM grades + CPIA + GDP if |SHAP| > 0.05
    conditional = []
    for feat in CONCISE_CONDITIONAL:
        if feat not in feature_names:
            continue
        idx  = feature_names.index(feat)
        sv_  = float(shap_vals[idx])
        if abs(sv_) > 0.03:
            conditional.append((feat, sv_, float(feature_vector_imputed.iloc[0, idx])))
    conditional.sort(key=lambda x: -abs(x[1]))

    # merge: always-shown first (sorted by |SHAP| among themselves), then conditional; cap at 7
    effects.sort(key=lambda x: -abs(x[1]))
    effects = effects + conditional
    effects = effects[:7]

    # --- one-sentence summary ---
    max_abs = max((abs(e[1]) for e in effects), default=1.0)
    up   = [e for e in effects if e[1] >  0.005]
    down = [e for e in effects if e[1] < -0.005]

    parts = []
    if up:
        parts.append("increased " + _join_phrases([_short_phrase(n, sv, v, _strength(abs(sv), max_abs)) for n, sv, v in up[:3]]))
    if down:
        parts.append("decreased " + _join_phrases([_short_phrase(n, sv, v, _strength(abs(sv), max_abs)) for n, sv, v in down[:3]]))

    summary = f"The statistical model predicted {final_pred:.1f} for this activity, with the rating " + ", and ".join(parts) + ". However, several other factors played an important role in the model's prediction as well."
    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    # ---- model ----
    with open(MODEL_PATH, 'rb') as f:
        rf_model = pickle.load(f)
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_names = meta['feature_names']   # 57

    # ---- saved predictions ----
    preds_df   = pd.read_csv(PREDS_PATH).set_index('activity_id')
    val_ids    = preds_df.index[preds_df['split'] == 'val']
    rf_preds   = pd.read_csv(RF_PREDS_PATH).set_index('activity_id')['pred_rf']

    # ---- LLM forecasts (verbose mode only) ----
    if not CONCISE:
        llm_preds  = load_llm_predictions()

    # ---- reconstruct features ----
    data       = reconstruct_features()

    # pool: val activities present in features + RF preds
    pool       = val_ids.intersection(data.index).intersection(rf_preds.index)

    if len(pool) == 0:
        return

    # ---- pick activity ----
    if len(sys.argv) > 1:
        chosen = sys.argv[1]
        if chosen not in pool:
            return
    else:
        chosen = random.choice(list(pool))

    # ---- feature vector + median imputation ----
    X_one  = data.loc[[chosen], feature_names].astype(float)
    X_pool = data.loc[pool, feature_names].astype(float)
    medians = X_pool.median()
    X_one_imp = X_one.fillna(medians)

    # ---- SHAP ----
    explainer  = shap.TreeExplainer(rf_model, feature_perturbation="tree_path_dependent")
    base_val   = float(np.asarray(explainer.expected_value).ravel()[0])

    sv         = explainer.shap_values(X_one_imp.values)          # (1, 57)
    shap_vals  = np.asarray(sv).ravel()

    if not CONCISE:
        iv         = explainer.shap_interaction_values(X_one_imp.values)  # (1, 57, 57)
        inter_mat  = np.asarray(iv).reshape(len(feature_names), len(feature_names))

    # ---- prediction numbers ----
    rf_pred    = float(rf_preds.loc[chosen])
    final_pred = float(preds_df.loc[chosen, 'y_pred'])

    # ===========================================================
    # CONCISE OUTPUT
    # ===========================================================
    if CONCISE:
        print(generate_explanation_for_webapp(rf_model, X_one_imp, feature_names, final_pred))

    # ===========================================================
    # VERBOSE OUTPUT  (original, unchanged)
    # ===========================================================
    else:
        llm_pred   = float(llm_preds[chosen]) if chosen in llm_preds.index else float('nan')
        correction = final_pred - rf_pred

        p      = len(feature_names)
        order  = np.argsort(-np.abs(shap_vals))          # main effects, strongest first
        max_abs = np.abs(shap_vals[order[0]])

        # top-3 off-diagonal interaction pairs
        pairs = []
        for i in range(p):
            for j in range(i + 1, p):
                pairs.append((np.abs(inter_mat[i, j]), i, j, inter_mat[i, j]))
        pairs.sort(reverse=True)

        W = 70
        print("=" * W)
        print(f"PREDICTION EXPLANATION  –  {chosen}")
        print("=" * W)

        # ---- summary box ----
        print(f"\n  Final prediction  (RF + LLM correction):  {final_pred:.2f}   (0–5 scale)")
        print(f"  Raw RF prediction:                         {rf_pred:.2f}")
        if not np.isnan(llm_pred):
            print(f"  LLM ensemble forecast:                     {llm_pred:.2f}")
        print(f"  Residual correction applied:               {correction:+.3f}")

        # ---- main effects ----
        print(f"\n{'-' * W}")
        print(f"WHY THE RF PREDICTED {rf_pred:.2f}")
        print(f"{'-' * W}")
        print(f"  The RF's baseline (expected value) is {base_val:.2f}.")
        print(f"  Feature contributions shifted it to {rf_pred:.2f}:\n")

        for rank in range(min(7, p)):
            idx  = int(order[rank])
            sv_  = float(shap_vals[idx])
            val_ = float(X_one_imp.iloc[0, idx])
            name = feature_names[idx]
            desc = _desc(name)

            # direction
            if sv_ > 0.005:
                arrow = "pushed UP"
            elif sv_ < -0.005:
                arrow = "pushed DOWN"
            else:
                arrow = "negligible"

            # strength relative to strongest driver
            frac = abs(sv_) / max_abs if max_abs > 0 else 0
            if   frac > 0.9: strength = "strongly by"
            elif frac > 0.6: strength = "moderately by"
            elif frac > 0.3: strength = "somewhat by"
            else:            strength = "weakly by"

            print(f"  {rank+1}. {desc}")
            print(f"     value = {_fmt(val_)},  SHAP = {sv_:+.3f}  →  {strength} {arrow}")

        # ---- interactions ----
        print(f"\n{'-' * W}")
        print("TOP 3 FEATURE INTERACTIONS")
        print(f"{'-' * W}")
        for rank, (abs_score, i, j, signed) in enumerate(pairs[:3]):
            fi, fj = feature_names[i], feature_names[j]
            vi     = float(X_one_imp.iloc[0, i])
            vj     = float(X_one_imp.iloc[0, j])

            print(f"\n  {rank+1}. {_desc(fi)}  ×  {_desc(fj)}")
            print(f"     values: {_fmt(vi)}  and  {_fmt(vj)}")
            print(f"     interaction SHAP: {signed:+.4f}  (|strength| = {abs_score:.4f})")
            if   signed >  0.003:
                print(f"     → Together they nudged the prediction UP.")
            elif signed < -0.003:
                print(f"     → Together they nudged the prediction DOWN.")
            else:
                print(f"     → Their joint effect was negligible.")

        # ---- LLM + residual correction ----
        print(f"\n{'-' * W}")
        print("HOW THE LLM FORECAST CHANGED THE PREDICTION")
        print(f"{'-' * W}")
        if np.isnan(llm_pred):
            print("  No LLM forecast was available for this activity.")
            print("  The final prediction equals the raw RF prediction.")
        else:
            n_variants = sum(1 for p_ in LLM_VARIANT_PATHS if p_.exists())
            print(f"  The LLM ensemble (mean of {n_variants} forecast variants)")
            print(f"  predicted {llm_pred:.2f} for this activity.")
            print(f"  The RF predicted {rf_pred:.2f}.")
            gap = llm_pred - rf_pred
            print(f"  LLM − RF gap: {gap:+.2f}")
            print()
            print(f"  A Ridge meta-learner, trained on [LLM prediction, RF prediction]")
            print(f"  to predict the RF's residual error, applied a correction of {correction:+.3f},")
            print(f"  yielding a final prediction of {final_pred:.2f}.")
            print()
            if abs(correction) < 0.03:
                print("  → The LLM and RF largely agreed; the correction was minimal.")
            elif correction > 0 and llm_pred > rf_pred:
                print("  → The LLM was more optimistic than the RF. The meta-learner")
                print("    shifted the prediction upward, partially toward the LLM's view.")
            elif correction < 0 and llm_pred < rf_pred:
                print("  → The LLM was more pessimistic than the RF. The meta-learner")
                print("    shifted the prediction downward, partially toward the LLM's view.")
            elif correction > 0:
                print(f"  → The meta-learner shifted the prediction up by {correction:+.3f}.")
            else:
                print(f"  → The meta-learner shifted the prediction down by {correction:+.3f}.")

        print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()
