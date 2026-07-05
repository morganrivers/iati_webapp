---
title: Data and Artifacts
tags: [data, artifacts, sqlite]
---

# Data and Artifacts

Back to [[Home]] · related: [[Forecasting Model]], [[Narrative Forecast (RAG)]].

The app reads from `./data` by default; set `DATA_DIR` to override (used to point at a Render persistent disk in production). Loaded once at boot by `webapp/model_loader.py`.

## Model artifacts (`data/rating_model_outputs/`)

Loaded by `load_model_and_data()`:

| File | Contents |
| --- | --- |
| `model.pkl` | Random Forest regressor (rating delta). |
| `extra_model.pkl` | ExtraTrees regressor (rating delta). |
| `per_org_baseline.json` | Modal historical rating per reporting org; must contain `__overall__`. |
| `start_year_correction.json` | Ridge drift fit; must contain `slope` and `intercept`. |
| `train_medians.json` | Median value per feature, used as imputation fallback. |
| `feature_names.json` | Ordered feature list; also the single source for sector-cluster names. |
| `train_features.csv` | Training feature rows, used for UI histograms/context. |

Asserts in `model_loader.py` guard the required keys. See how these combine in [[Forecasting Model]].

## Runtime data files (`data/`)

| Path | Purpose |
| --- | --- |
| `webapp.db` | SQLite store: activity info, mock (retrospective) forecasts, embeddings. |
| `outcome_tags/tag_models.pkl` | Outcome-tag classifiers (optional; app boots without it). |
| `postactivity_summaries.jsonl` | Ex-post summaries used by the narrative forecast. |
| `trained_umap_models_trainval.pkl` | UMAP models for target embeddings. |
| `best_model_predictions.csv` | Reference predictions for the RAG prompt. |
| `outputs_summaries.jsonl` | Activity summaries, BM25 corpus source. |
| `merged_overall_ratings.jsonl` | Ground-truth ratings. |
| `outputs_retrospective_forecast.jsonl` | Mock forecasts for few-shot examples. |

## SQLite schema (`webapp.db`)

Built by `webapp/scripts/build_webapp_db.py`:

- `activity_info` — all CSV columns plus `chatgpt_description` and `risks_summary`.
- `mock_forecasts` — `activity_id`, `content` (retrospective forecast text).
- `embeddings` — activity embedding BLOBs (used when `USE_VECTOR_KNN=True`).

`modules/sqlite_loaders.py` provides `load_activity_info_sqlite`, `make_lazy_mock_forecasts` (lazy per-key lookup), and `load_bm25_corpus_sqlite`. These replace the research code's CSV/JSONL loaders at runtime; see [[Narrative Forecast (RAG)]].

## App-generated data

- `webapp/extracted_pdf_data/{activity_id}/` — per-PDF extraction cache from the [[Extraction Pipeline]], plus `uploaded.pdf` and `app_state.json`.
- `webapp/llm_forecasts/` — narrative forecast outputs, RAG indexes, and PDF text.

## Large-file handling

Large artifacts (`tag_models.pkl`, and `webapp.db` if it exceeds GitHub's file limit) are delivered via a Render persistent disk rather than committed to git. Point `DATA_DIR` at the mount.
