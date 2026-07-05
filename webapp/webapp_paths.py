"""Single source of truth for webapp filesystem paths.

Two logical roots the webapp reads and writes:

  DATA_DIR      Model artifacts and reference data: rating_model_outputs/,
                the reference CSV/JSONL files, webapp.db, outcome_tags/.
  PROJECTS_DIR  Per-project folders written at runtime (PDF extractions plus
                user edits and app state). Named ``projects/`` at the repo root.

Both honor the ``DATA_DIR`` environment variable, which is set on Render to the
persistent disk that holds the artifacts and the runtime project folders. Without
it, paths resolve from the repo layout via ``__file__`` so they are identical
whether the app is launched from the repo root (``streamlit run webapp/app.py``)
or from inside ``webapp/`` (as the test suite does).
"""
import os
import sys
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent

assert WEBAPP_DIR.name == "webapp", f"webapp_paths.py must live under webapp/, found {WEBAPP_DIR}"

_ENV_DATA_DIR = os.environ.get("DATA_DIR")

DATA_DIR = Path(_ENV_DATA_DIR).resolve() if _ENV_DATA_DIR else REPO_ROOT / "data"

# On Render the persistent disk (DATA_DIR) also holds the runtime project folders.
# Locally they live at the repo root, not in the repo data/ dir.
PROJECTS_DIR = (
    DATA_DIR / "projects" if _ENV_DATA_DIR else REPO_ROOT / "projects"
)

MODEL_DIR = DATA_DIR / "rating_model_outputs"
TRAIN_MEDIANS_PATH = MODEL_DIR / "train_medians.json"

_SRC_UTILS_DIR = REPO_ROOT / "src" / "utils"
_SRC_EXTRACT_DIR = REPO_ROOT / "src" / "extract_structured_database"
_SRC_FORECAST_DIR = REPO_ROOT / "src" / "forecast_outcomes"


def ensure_src_paths() -> None:
    """Idempotently add all src/ subtrees and webapp dirs to sys.path."""
    for p in [
        str(WEBAPP_DIR),
        str(WEBAPP_DIR / "modules"),
        str(_SRC_UTILS_DIR),
        str(_SRC_EXTRACT_DIR),
        str(_SRC_FORECAST_DIR),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)
