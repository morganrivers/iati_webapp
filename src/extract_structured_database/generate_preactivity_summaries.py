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
from get_all_pages_within_category import load_and_filter_rows

from extracting_and_grading_helper_functions import make_executor, consolidate_rows_by_activity,loop_over_rows_to_call_model, add_fallback_rows_for_missing_activities_strict_baseline

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

if __name__ == "__main__":
    _program_start = datetime.now()  # <-- add this
    execpool = make_executor()

    output_jsonl = str(DATA_DIR / "outputs_summaries.jsonl")

    # The rows_summary below seems to be quite good for identifying pages which summarize key objectives and activities. Sometimes, the next page would be useful as it gets cut off.
    rows_summary = load_and_filter_rows(
        section="Baseline",
        subcategory_a=["condensed_summary","broad_objectives"],
        subcategory_b=["condensed_summary","broad_objectives"],
        top_k_per_activity=5,  # adjust as needed; set to None to disable
        min_score=7,
        get_at_least_top_k=True
    )

    print("loading ratings")
    from helpers_for_ratings_and_final_activity_features import load_ratings  
    print("done loading ratings")

    rated = load_ratings(str(DATA_DIR / "merged_overall_ratings.jsonl"))
    R = set(rated.index)

    Bsel = {r["activity_id"] for r in consolidate_rows_by_activity(rows_summary)}
    print(f"[DEBUG] selected activities (before fallback): {len(Bsel)}")
    print(f"[DEBUG] rated ∩ selected: {len(R & Bsel)}")
    print(f"[DEBUG] rated missing at selection: {len(R - Bsel)}")

    AID = "44000-P157571"
    hits = [r for r in rows_summary if r.get("activity_id") == AID]
    print("[DEBUG] after load_and_filter_rows hits:", len(hits))
    for r in hits[:10]:
        print("[DEBUG] row:", {k: r.get(k) for k in [
            "activity_id","section","subcategory_a","subcategory_b","score",
            "title","doc_type","cached_file","page_start","page_end"
        ]})
    rows_summary = add_fallback_rows_for_missing_activities_strict_baseline(rows_summary, DATA_DIR, max_pages=10)
    hits_after = [r for r in rows_summary if r.get("activity_id") == AID]
    print("[DEBUG] after fallback hits:", len(hits_after))
    for r in hits_after[:10]:
        print("[DEBUG] fallback row:", {k: r.get(k) for k in [
            "activity_id","section","subcategory_a","subcategory_b","score",
            "title","doc_type","cached_file","page_start","page_end"
        ]})

    Bafter = {r["activity_id"] for r in consolidate_rows_by_activity(rows_summary)}
    print(f"[DEBUG] activities after fallback: {len(Bafter)}")
    print(f"[DEBUG] fallback added activities: {len(Bafter - Bsel)}")
    print("[DEBUG] is AID missing?", AID in (R - Bsel))
    print(f"[DEBUG] rated ∩ after_fallback: {len(R & Bafter)}")
    print(f"[DEBUG] rated still missing after fallback: {len(R - Bafter)}")

    chunked_by_activity_id = consolidate_rows_by_activity(rows_summary)
    prompts_summary = get_prompts_summary(chunked_by_activity_id)
    # response_schema = get_response_schema_summary()
    # asyncio.run(loop_over_rows_to_call_model(output_jsonl,chunked_by_activity_id,prompts_summary,response_schema))
    try:
        asyncio.run(loop_over_rows_to_call_model(output_jsonl, chunked_by_activity_id,
                                                  prompts_summary, None, execpool))
    finally:
        # CRUCIAL: don’t wait for stuck HTTP retries
        execpool.shutdown(wait=False, cancel_futures=True)
    print(f"\n(END PROGRAM): took {(datetime.now() - _program_start).total_seconds():.2f}s\n\n\n\n")  # <-- add this

