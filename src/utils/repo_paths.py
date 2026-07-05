"""
Single source of truth for repo-relative filesystem paths.

Scripts historically hardcoded cwd-relative paths like "../../data/foo.csv",
which only resolve correctly when the script is run from its own directory.
Import DATA_DIR from here instead so paths resolve from the repo location
(via __file__), independent of the current working directory.

DATA_DIR defaults to <repo>/data and can be redirected with the environment
variable FORECASTING_DATA_DIR (set it before import) so tests can point the
pipeline at a fixture directory.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_override = os.environ.get("FORECASTING_DATA_DIR")
DATA_DIR = Path(_override).resolve() if _override else REPO_ROOT / "data"
