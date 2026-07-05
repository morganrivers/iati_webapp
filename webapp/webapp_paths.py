"""Single source of truth for webapp filesystem paths.

Two logical roots the webapp reads and writes:

  DATA_DIR           Model artifacts and reference data: rating_model_outputs/,
                     the reference CSV/JSONL files, webapp.db, outcome_tags/.
  EXTRACTED_PDF_DIR  Per-project extracted-PDF folders written at runtime.

Both honor the ``DATA_DIR`` environment variable, which is set on Render to the
persistent disk that holds the artifacts and the runtime project folders. Without
it, paths resolve from the repo layout via ``__file__`` so they are identical
whether the app is launched from the repo root (``streamlit run webapp/app.py``)
or from inside ``webapp/`` (as the test suite does).
"""
import os
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent

assert WEBAPP_DIR.name == "webapp", f"webapp_paths.py must live under webapp/, found {WEBAPP_DIR}"

_ENV_DATA_DIR = os.environ.get("DATA_DIR")

DATA_DIR = Path(_ENV_DATA_DIR).resolve() if _ENV_DATA_DIR else REPO_ROOT / "data"

# On Render the persistent disk (DATA_DIR) also holds the runtime project folders.
# Locally they live beside the webapp code, not in the repo data/ dir.
EXTRACTED_PDF_DIR = (
    DATA_DIR / "extracted_pdf_data" if _ENV_DATA_DIR else WEBAPP_DIR / "extracted_pdf_data"
)

MODEL_DIR = DATA_DIR / "rating_model_outputs"
TRAIN_MEDIANS_PATH = MODEL_DIR / "train_medians.json"
