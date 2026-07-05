---
title: Home
tags: [moc, index]
---

# IATI Activity Success Forecasting — Documentation

Map of content for this project. Open this folder as an Obsidian vault; every note is cross-linked.

A Streamlit app that forecasts the likely evaluation success rating (0–5) of an international development activity. Upload a project PDF or enter details by hand; the app extracts structured features with an LLM, predicts a rating with a tree-ensemble model, and optionally writes a narrative forecast.

## Start here

- [[Architecture]] — how the pieces fit together, top-level data flow.
- [[Extraction Pipeline]] — the 5-phase LLM pipeline that turns a PDF into features.
- [[Forecasting Model]] — the per-org baseline + RF/ExtraTrees ensemble + start-year correction.
- [[Narrative Forecast (RAG)]] — the multi-stage LLM forecast with retrieval.
- [[Data and Artifacts]] — runtime data, model pickles, and the SQLite store.
- [[UI Pages]] — tour of the Streamlit pages the user sees.
- [[Development]] — local setup, configuration, tests, deployment.
- [[Glossary]] — IATI terms and model feature names.

## One-paragraph summary

The user uploads a project document. [[Extraction Pipeline|Extraction]] categorizes pages, summarizes the activity, and extracts finance, sector, and qualitative features with Gemini. Those features feed the [[Forecasting Model]], which produces a 0–5 rating. Optionally the [[Narrative Forecast (RAG)]] retrieves evidence from the PDF and similar past activities to write a staged narrative. All UI is Streamlit; see [[UI Pages]].

## Diagram

See the full data-flow diagram in [[Architecture#Data flow diagram]].
