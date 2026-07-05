import sys

import csv
import re

from datetime import datetime
from pathlib import Path
import asyncio
import pprint
UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from repo_paths import DATA_DIR
from extracting_and_grading_helper_functions import consolidate_rows_by_activity, loop_over_rows_to_call_model

def get_prompts_summary(rows_summary):
    prompts = {}
    for row in rows_summary:
        # pprint.pprint("row")
        # pprint.pprint(row)
        aid = row.get("activity_id")
        title = row.get("title") or row.get("activity_title") or ""
        # cf  = obj.get("cached_file")
        # ps  = obj.get("page_start")
        # prompts[aid] = f"aid: {aid}"#" + cachedf {cf} + ps {ps}"
        if aid[:4] == "DE-1":
            prompts[aid] = f"""You are extracting information from the uploaded page(s) of project information documents for the following activity. Provide all information contained in the pages about the activity that would be known at the beginning of the activity only, in paragraph-only format, including information useful for forecasting overall activity success (but only if it could have been known at the beginning of the activity). Do not in any way reveal the actual outcome of the activity. Include only information found in the pages. Respond only in English.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
        else:
            prompts[aid] = f"""You are extracting information from the uploaded page(s) of project information documents for the following activity. Provide all information contained in the pages about the activity in paragraph-only format, including information useful for forecasting overall activity success. Include only information found in the pages. Respond only in English.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
    return prompts

