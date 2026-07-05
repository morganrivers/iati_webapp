---
title: Glossary
tags: [glossary, reference]
---

# Glossary

Back to [[Home]] · related: [[Forecasting Model]], [[Extraction Pipeline]].

## Domain

- **IATI** — International Aid Transparency Initiative; the open standard and dataset the model is trained on.
- **Activity** — a single aid project (the unit being forecast).
- **Reporting organization** — the org that publishes the activity to IATI; drives the baseline in [[Forecasting Model]]. Covered: UK FCDO, Asian Development Bank, World Bank, BMZ.
- **Success rating** — 0–5 evaluation score the model predicts; anchored to a 6-option World-Bank-style scale in the [[Narrative Forecast (RAG)]].
- **Ex-post / retrospective forecast** — evaluation-time summary of a past activity, used as few-shot examples.
- **CPIA** — Country Policy and Institutional Assessment score (a location feature).
- **WGI** — Worldwide Governance Indicators; source of `governance_composite` and `wgi_any_missing`.

## Model features

LLM grades (from [[Extraction Pipeline]] Phase 4):

- **finance** — qualitative finance/budget soundness grade.
- **integratedness** — how integrated the activity is with local systems/partners.
- **implementer_performance** — track record / capacity of the implementer.
- **targets** — quality and clarity of the activity's targets.
- **context** — favourability of the operating context.
- **risks** — risk profile grade.
- **complexity** — activity complexity grade.

Metadata / derived:

- **activity_scope**, **finance_is_loan**, **planned_duration**, **planned_expenditure** (raw USD), **log_planned_expenditure**, **expenditure_per_year_log**, **expenditure_x_complexity**.

Location:

- **gdp_percap** (log), **cpia_score**, **governance_composite**, **wgi_any_missing**, region one-hots **region_AFE/AFW/EAP/ECA/LAC/MENA/SAS**.

Embeddings / distances (see [[Narrative Forecast (RAG)]] and `targets_embedder.py`):

- **umap3_x/y/z** — 3D UMAP coordinates of the targets embedding.
- **country_distance**, **sector_distance** — embedding distances to reference sets.
- **sector_cluster_*** — expenditure proportion per sector cluster (names sourced from `feature_names.json`).

Bookkeeping:

- **rep_org_0/1/2** — one-hot reporting-org dummies selecting the baseline.
- ***_missing** flags and **feature_completeness_ratio** — imputation / completeness indicators; missing values fall back to `train_medians` (see [[Data and Artifacts]]).

## Pipeline / infra terms

- **BM25** — lexical retrieval used by default for KNN neighbor selection (no embeddings file).
- **RAG synthesis** — retrieving PDF passages to ground the narrative forecast.
- **KNN neighbors** — the `k=3` most similar past activities used as few-shot context.
- **train_medians** — per-feature median fallback for any missing model input.
