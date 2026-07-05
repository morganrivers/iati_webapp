"""
Canonical train/validation/test split configuration.

Single source of truth for all scripts in this project.

  train: activities with start_date <= LATEST_TRAIN_POINT
  val:   LATEST_TRAIN_POINT < start_date <= LATEST_VALIDATION_POINT
  test:  LATEST_VALIDATION_POINT < start_date < TOO_LATE_CUTOFF
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

LATEST_TRAIN_POINT = "2013-02-06"
LATEST_VALIDATION_POINT = "2016-06-06"
TOO_LATE_CUTOFF = "2020-01-01"



