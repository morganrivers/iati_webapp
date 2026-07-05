---
title: Development
tags: [dev, setup, deployment, testing]
---

# Development

Back to [[Home]] · related: [[Data and Artifacts]], [[Architecture]].

## Local setup

```bash
pip install -r requirements.txt
cp .env.template .env          # then fill in API keys and APP_PASSWORD
python webapp/scripts/build_webapp_db.py
streamlit run webapp/app.py
```

By default the app reads from `./data`. Set `DATA_DIR` to override. See [[Data and Artifacts]].

## Configuration (environment variables)

Read from `.env` (see `.env.template`):

| Variable | Purpose |
| --- | --- |
| `APP_PASSWORD` | Password gate for the app. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini: feature extraction and embeddings. |
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | Narrative forecasts. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Notifications (optional). |
| `DATA_DIR` | Runtime data directory (defaults to `./data`). |

## Tests

Under `tests/`:

- `test_suite.py` — core logic, including the [[Forecasting Model]] formula via the shared `predict_rating()`.
- `test_suite_ui_e2e.py` — end-to-end UI checks.
- `conftest.py`, `streamlit_mock.py` — fixtures and a Streamlit mock so UI modules import without a running server.

Run with:

```bash
micromamba run -n py311 pytest tests/
```

## Deployment

`render.yaml` defines the Render service. Large artifacts (`tag_models.pkl`, and `webapp.db` when it exceeds GitHub's file limit) live on a Render persistent disk; `DATA_DIR` points at the mount. `.streamlit/config.toml` holds Streamlit server config.

## Regenerating model artifacts

See [[Forecasting Model#Regenerating artifacts]]. The artifacts are exported from the thesis forecasting repo, not produced by this app.

## Diagram source

`docs/build_webapp_drawio.py` regenerates `webapp_flow_diagram.*` (drawio, json, svg, png) embedded in [[Architecture]].

## Observability notes

`webapp/debug_utils._print_ram()` is sprinkled through boot and the pipelines to trace memory (relevant on Render's memory limits). `logging_config.py` sets up logging; `app.py` installs a matplotlib import tracer to catch accidental heavy imports.
