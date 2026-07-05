import json
import pprint
import sys
import asyncio
from datetime import datetime
from typing import Optional, Set, Dict, Any, Tuple
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from repo_paths import DATA_DIR
from get_all_pages_within_category import load_and_filter_rows
from extracting_and_grading_helper_functions import consolidate_rows_by_activity, loop_over_rows_to_call_model, load_summaries_from_chatgpt_into_bundles, add_fallback_rows_for_missing_activities_strict_baseline
from view_categorization_scores import view_rows_as_pdf
import csv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor


"""
NOTE:
load_summaries_from_chatgpt_into_bundles
uses: 
"../../data/outputs_summaries.jsonl"
"""
CHATGPT_SUMMARIES_JSONL = str(DATA_DIR / "outputs_summaries.jsonl")
ACTIVITY_INFO_CSV = DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"

@lru_cache(None)
def _load_activity_info() -> Dict[str, Dict[str, str]]:
    """Map activity_id -> row from the constructed CSV."""
    out: Dict[str, Dict[str, str]] = {}
    with ACTIVITY_INFO_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            aid = (r.get("activity_id") or "").strip()
            if aid:
                out[aid] = r
    return out

def get_implementer_performance_prompt(rows_summary,info):
    prompts = {}
    for row in rows_summary:
        aid = row.get("activity_id")

        # Prefer enriched data from CSV; fall back to what's already in the row
        meta = info.get(aid, {})
        title = row.get("title") or ""
        orgs = meta.get("participating_orgs") or ""
        locations = meta.get("country_location") or ""
        gdp_percap = meta.get("gdp_percap") or ""
        implementing_org_type = meta.get("implementing_org_type") or ""
        chatgpt_description = row.get("chatgpt_description") or ""
        if chatgpt_description is None or chatgpt_description == "":
            continue    

        if aid[:4] == "DE-1":
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, describe what is said relevant to assessing the likely performance of the organization implementing the activity, but only information that could have been known at the beginning of the activity. Do not in any way reveal the actual outcome of the activity. Include information which could inform the implementing organization's abilities and skin-in-the-game of key actors. Where present in the activity, include details about who would be implementing the activity, whether the partner organization is truly invested in the activity, and whether other relevant stakeholders would be aiding or deteriorating activity outcomes. If there is no additional information, respond only with: "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
        else:
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, describe what is said relevant to assessing the likely performance of the organization implementing the activity, including information which could inform the implementing organization's abilities and skin-in-the-game of key actors. Where present in the activity, include details about who would be implementing the activity, whether the partner organization is truly invested in the activity, and whether other relevant stakeholders would be aiding or deteriorating activity outcomes. If there is no additional information, respond only with: "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}"""
        if locations != "" and locations is not None:
            prompts[aid] += f"\nACTIVITY LOCATION(S): {locations}"
        if gdp_percap != "" and gdp_percap is not None:
            prompts[aid] += f"\nLOCATION GDP PER CAPITA, USD: {int(float(gdp_percap))}"
        if orgs != "" and orgs is not None:
            prompts[aid] += f"\nPARTICIPATING ORGANIZATIONS: {orgs}"
        if implementing_org_type != "" and implementing_org_type is not None:
            prompts[aid] += f"\nIMPLEMENTING ORGANIZATION CATEGORY: {implementing_org_type}"
        if chatgpt_description != "" and (chatgpt_description != "NO RESPONSE"):
            prompts[aid] += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"

    print("\n\nprompt")
    print(prompts[aid])
        # quit()
    return prompts

def get_target_outcomes_prompt(rows_summary,info):
    prompts = {}

    for row in rows_summary:
        # pprint.pprint("row")
        # pprint.pprint(row)
        aid = row.get("activity_id")
        title = row.get("title")
        chatgpt_description = row.get("chatgpt_description")
        if chatgpt_description is None or chatgpt_description == "":
            continue    
        meta = info.get(aid, {})

        planned_start_date = meta.get("original_planned_start_date") if meta.get("original_planned_start_date") != "" else  meta.get("actual_start_date")
        planned_end_date = meta.get("original_planned_close_date")

        # cf  = obj.get("cached_file")
        # ps  = obj.get("page_start")
        # prompts[aid] = f"aid: {aid}"#" + cachedf {cf} + ps {ps}"
        if aid[:4] == "DE-1":
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, detail the targeted outcomes of the activity, but only information that could have been known at the beginning of the activity. Do not in any way reveal the actual outcome of the activity. Include in your response what targeted outcomes would be necessary to acheive success as defined at the beginning of the activity. If no targets or intended outcomes are mentioned in the pages, respond only with: "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
        else:
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, detail the targeted outcomes of the activity. Include in your response what would be needed to consider the activity an overall success. Do not omit important details. If no targets or intended outcomes are mentioned in the pages, respond only with: "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}""" # Do not omit important details. => is it really necessary to say?
        if planned_start_date != "" and planned_start_date is not None:
            prompts[aid] += f"\nPLANNED START DATE: {planned_start_date}"
        if planned_end_date != "" and planned_end_date is not None:
            prompts[aid] += f"\nPLANNED END DATE: {planned_end_date}"
        if chatgpt_description != "" and (chatgpt_description != "NO RESPONSE"):
            prompts[aid] += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"
    # print("\n\nprompt")
    # print(prompts[aid])

    return prompts

def get_risks_prompt(rows_summary,info):
    prompts = {}

    for row in rows_summary:
        # pprint.pprint("row")
        # pprint.pprint(row)
        aid = row.get("activity_id")
        title = row.get("title")
        chatgpt_description = row.get("chatgpt_description")
        if chatgpt_description is None or chatgpt_description == "":
            continue    
        meta = info.get(aid, {})

        planned_start_date = meta.get("original_planned_start_date") if meta.get("original_planned_start_date") != "" else  meta.get("actual_start_date")
        planned_end_date = meta.get("original_planned_close_date")

        # cf  = obj.get("cached_file")
        # ps  = obj.get("page_start")
        # prompts[aid] = f"aid: {aid}"#" + cachedf {cf} + ps {ps}"
        if aid[:4] == "DE-1":
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, detail all relevant risks to the activity as would have been known at the beginning of the activity. Only provide information present in the uploaded pages. If there is no additional information on the risks, respond only with "NO RESPONSE". Do not in any way reveal the actual outcome of the activity. Include only information found in the pages. Respond only in English.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
        else:
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, detail all relevant risks to the activity not achieving overall success. Only provide information present in the uploaded pages. If there is no additional information on the risks, respond only with "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}"""
        if planned_start_date != "" and planned_start_date is not None:
            prompts[aid] += f"\nPLANNED START DATE: {planned_start_date}"
        if planned_end_date != "" and planned_end_date is not None:
            prompts[aid] += f"\nPLANNED END DATE: {planned_end_date}"
        if chatgpt_description != "" and (chatgpt_description != "NO RESPONSE"):
            prompts[aid] += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"
    print("\n\nprompt")
    print(prompts[aid])

    return prompts

def get_possibilities_prompt(rows_summary,info):
    prompts = {}

    for row in rows_summary:
        # pprint.pprint("row")
        # pprint.pprint(row)
        aid = row.get("activity_id")
        title = row.get("title")
        chatgpt_description = row.get("chatgpt_description")
        if chatgpt_description is None or chatgpt_description == "":
            continue    
        meta = info.get(aid, {})

        planned_start_date = meta.get("original_planned_start_date") if meta.get("original_planned_start_date") != "" else  meta.get("actual_start_date")
        planned_end_date = meta.get("original_planned_close_date")
        # cf  = obj.get("cached_file")
        # ps  = obj.get("page_start")
        # prompts[aid] = f"aid: {aid}"#" + cachedf {cf} + ps {ps}"
        if aid[:4] == "DE-1":
            prompts[aid] = f"""You are extracting information from the uploaded page(s) of project information documents for the following activity. Respond with positive indicators for the activity as they would have been known at the beginning of the activity. Include reasons that the activity could be overall successful, or could overcome inherent risks, from the perspective of the beginning of the activity. Do not in any way reveal the actual outcome of the activity. Include only information found in the pages. If there is no additional information on positive indicators, respond only with "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
        else:
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, respond with positive indicators for the activity as they are described in the attached pages. Include reasons that the activity could be overall successful, or could overcome inherent risks. Only provide information present in the uploaded pages. If there is no additional information on positive indicators, respond only with "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}"""
        if planned_start_date != "" and planned_start_date is not None:
            prompts[aid] += f"\nPLANNED START DATE: {planned_start_date}"
        if planned_end_date != "" and planned_end_date is not None:
            prompts[aid] += f"\nPLANNED END DATE: {planned_end_date}"
        if chatgpt_description != "" and (chatgpt_description != "NO RESPONSE"):
            prompts[aid] += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"

    print("\n\nprompt")
    print(prompts[aid])

    return prompts

def get_context_prompt(rows_summary,info):
    prompts = {}

    for row in rows_summary:
        # pprint.pprint("row")
        # pprint.pprint(row)
        chatgpt_description = row.get("chatgpt_description")
        if chatgpt_description is None or chatgpt_description == "":
            continue    
        aid = row.get("activity_id")
        title = row.get("title")
        meta = info.get(aid, {})
        planned_start_date = meta.get("original_planned_start_date") if meta.get("original_planned_start_date") != "" else  meta.get("actual_start_date")
        planned_end_date = meta.get("original_planned_close_date")
        orgs = meta.get("participating_orgs") or ""

        locations = meta.get("country_location") or ""

        gdp_percap = meta.get("gdp_percap") or ""

        # cf  = obj.get("cached_file")
        # ps  = obj.get("page_start")
        # prompts[aid] = f"aid: {aid}"#" + cachedf {cf} + ps {ps}"
        if aid[:4] == "DE-1":
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, provide details on the external context of the activity, as relates to assessing its probability of overall success from the perspective of the beginning of the activity. Do not reveal the actual outcome of the activity. Do not omit important details. If there is no additional information on the external context, respond only with "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
        else:
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, provide details on the external context of the activity, as relates to assessing its probability of overall success. Do not omit important details. If there is no additional information on the external context, respond only with "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}"""
        if planned_start_date != "" and planned_start_date is not None:
            prompts[aid] += f"\nPLANNED START DATE: {planned_start_date}"
        if planned_end_date != "" and planned_end_date is not None:
            prompts[aid] += f"\nPLANNED END DATE: {planned_end_date}"
        if locations != "" and locations is not None:
            prompts[aid] += f"\nACTIVITY LOCATION(S): {locations}"
        if gdp_percap != "" and gdp_percap is not None:
            prompts[aid] += f"\nLOCATION GDP PER CAPITA, USD: {int(float(gdp_percap))}"
        if chatgpt_description != "" and (chatgpt_description != "NO RESPONSE"):
            prompts[aid] += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"
    print("\n\nprompt")
    print(prompts[aid])
    return prompts

def get_finance_prompt(rows_summary, info):
    prompts = {}

    for row in rows_summary:
        # pprint.pprint("row")
        # pprint.pprint(row)
        aid = row.get("activity_id")
        title = row.get("title")
        chatgpt_description = row.get("chatgpt_description")
        if chatgpt_description is None or chatgpt_description == "":
            continue    
        meta = info.get(aid, {})
        planned_start_date = meta.get("original_planned_start_date") if meta.get("original_planned_start_date") != "" else  meta.get("actual_start_date")
        planned_end_date = meta.get("original_planned_close_date")
        orgs = meta.get("participating_orgs") or ""

        locations = meta.get("country_location") or ""

        gdp_percap = meta.get("gdp_percap") or ""
        activity_scope = meta.get("activity_scope") or ""

        # cf  = obj.get("cached_file")
        # ps  = obj.get("page_start")
        # prompts[aid] = f"aid: {aid}"#" + cachedf {cf} + ps {ps}"
        if aid[:4] == "DE-1":
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, detail the specific sub-activities and how much is budgeted for them, as would have been known from the beginning of the activity. Include any details indicating what the cost of the activity would be as planned at the beginning. Include any details indicating how well financed the activity would likely be, or how financing may affect activity success, but do not reveal how the activity actually went with regards to financing. If there is no additional information regarding activity finances respond only with "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
        else:
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents for the following activity, detail the specific sub-activities and how much is budgeted for them. Include any details indicating what the cost of the activity would be, noting any aspects that appear likely to be under-funded. Include any details indicating how well financed the activity would likely be, or how financing may affect activity success. If there is no additional information regarding activity finances respond only with "NO RESPONSE". Respond only in English.
ACTIVITY TITLE: {title}"""
        if planned_start_date != "" and planned_start_date is not None:
            prompts[aid] += f"\nPLANNED START DATE: {planned_start_date}"
        if planned_end_date != "" and planned_end_date is not None:
            prompts[aid] += f"\nPLANNED END DATE: {planned_end_date}"
        if locations != "" and locations is not None:
            prompts[aid] += f"\nACTIVITY LOCATION(S): {locations}"
        if activity_scope != "" and activity_scope is not None:
            prompts[aid] += f"\nACTIVITY SCOPE: {activity_scope}"
        if orgs != "" and orgs is not None:
            prompts[aid] += f"\nPARTICIPATING ORGANIZATIONS: {orgs}"
        if chatgpt_description != "" and (chatgpt_description != "NO RESPONSE"):
            prompts[aid] += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"

    print("\n\nprompt")
    print(prompts[aid])

    return prompts

def get_misc_prompt(rows_summary,info):
    prompts = {}

    for row in rows_summary:
        # pprint.pprint("row")
        # pprint.pprint(row)
        aid = row.get("activity_id")
        title = row.get("title")
        chatgpt_description = row.get("chatgpt_description")
        if chatgpt_description is None or chatgpt_description == "":
            continue    
        meta = info.get(aid, {})
        planned_start_date = meta.get("original_planned_start_date") if meta.get("original_planned_start_date") != "" else  meta.get("actual_start_date")
        planned_end_date = meta.get("original_planned_close_date")
        orgs = meta.get("participating_orgs") or ""

        locations = meta.get("country_location") or ""

        gdp_percap = meta.get("gdp_percap") or ""
        activity_scope = meta.get("activity_scope") or ""

        # cf  = obj.get("cached_file")
        # ps  = obj.get("page_start")
        # prompts[aid] = f"aid: {aid}"#" + cachedf {cf} + ps {ps}"
        if aid[:4] == "DE-1":
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents and the provided details for the following activity, using only information that could have been known at the beginning of the activity:
- Detail information concerning the technical complexity and other concerning complexities of the activity, otherwise describe how the activity may in fact not require much complexity in implementation. Do not reveal activity outcomes.
- Consider whether the activity could be considered a larger program, and if not, if it is integrated directly and cohesively into a larger program, or whether it is rather independent or isolated from a larger program. Also detail whether the program itself could be considered a larger program.
- If disbursements are expected, respond with the total expected (non-loan, non-credit) disbursements for the future activity implementation, including the monetary unit. Do not reveal the true amount of disbursements, only what was initially planned if available.
- If loans or credit are expected, respond with the total size of loans and credit expected for the future activity implementation, including the monetary unit. Do not reveal the true amount of loans or credit, only what was initially planned if available.
ACTIVITY TITLE: {title}""" #" + cachedf {cf} + ps {ps}"
        else:
            prompts[aid] = f"""Using the uploaded page(s) of activity information documents and the provided details for the following activity:
- Detail information concerning the technical complexity and other concerning complexities of the activity, otherwise describe how the activity may in fact not require much complexity in implementation. 
- Consider whether the activity could be considered a larger program, and if not, if it is integrated directly and cohesively into a larger program, or whether it is rather independent or isolated from a larger program. Also detail whether the program itself could be considered a larger program.
- If disbursements are expected, respond with the total expected (non-loan, non-credit) disbursements for the future activity implementation, including the monetary unit.
- If loans or credit are expected, respond with the total size of loans and credit expected for the future activity implementation, including the monetary unit.
Report all amounts with their original currency units; do not convert. Respond only in English.
ACTIVITY TITLE: {title}""" # TODO: maybe cache the first part to confuse the model less for all these questions?
        if planned_start_date != "" and planned_start_date is not None:
            prompts[aid] += f"\nPLANNED START DATE: {planned_start_date}"
        if planned_end_date != "" and planned_end_date is not None:
            prompts[aid] += f"\nPLANNED END DATE: {planned_end_date}"
        if locations != "" and locations is not None:
            prompts[aid] += f"\nACTIVITY LOCATION(S): {locations}"
        if activity_scope != "" and activity_scope is not None:
            prompts[aid] += f"\nACTIVITY SCOPE: {activity_scope}"
        if orgs != "" and orgs is not None:
            prompts[aid] += f"\nPARTICIPATING ORGANIZATIONS: {orgs}"
        if chatgpt_description != "" and (chatgpt_description != "NO RESPONSE"):
            prompts[aid] += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"


    print("\n\nprompt")
    print(prompts[aid])

    return prompts

def get_description_schema():
    return {
        "type": "string",
        "maxLength": 500,
    }


def get_few_grades_schema():
    return {
        "type": "object",
        "properties": {
            "complexity_details": {"type": "string", "maxLength": 300},
            "how_integrated_description": {"type": "string", "maxLength": 300},
            "disbursement_total": {"type": "number"},
            "disbursement_units": {"type": "string","maxLength":25},
            "loan_total": {"type": "number"},
            "loan_units": {"type": "string","maxLength":25},
        },
        "required": ["complexity_details", "how_integrated_description"],
        "propertyOrdering": [
            "complexity_details",
            "how_integrated_description",
            "disbursement_total",
            "disbursement_units",
            "loan_total",
            "loan_units",
        ]
    }







if __name__ == "__main__":
    execpool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="genai")
    _program_start = datetime.now()  # <-- add this
    try:
        #IMPORTANT NOTE: if cgange this, need to also change in A_grade_baseline_features!!!#

        info = _load_activity_info()


        print("loading ratings...")
        from helpers_for_ratings_and_final_activity_features import load_ratings  
        print("done loading ratings...")

        rated = load_ratings(str(DATA_DIR / "merged_overall_ratings.jsonl"))

        output_jsonl = str(DATA_DIR / "outputs_implementer_performance.jsonl")

        # Overall, I was happy, the majority seem relevant for assessing involvment or skill levels/experience
        rows_performance = load_and_filter_rows(
            section="Baseline",
            subcategory_a=["partner_identity_or_skill","whether_skin_in_the_game", "other_stakeholder_engagement","who_implements",],
            subcategory_b=["partner_identity_or_skill","whether_skin_in_the_game", "other_stakeholder_engagement","who_implements",],
            # subcategory_a=["who_implements"],
            # subcategory_b=["who_implements"],
            top_k_per_activity=5,  # adjust as needed; set to None to disable
            min_score=7,
            get_surrounding_if_not_enough=True,
            get_at_least_top_k=True
        )
        rows_performance = add_fallback_rows_for_missing_activities_strict_baseline(rows_performance, DATA_DIR, max_pages=10, rated_ids=set(rated.index))

        # Activity coverage
        #   Scope:                           section='baseline'
        #   Matched unique activities:       471
        #   Denominator (unique activities): 626
        #   Fraction:                        471/626 = 0.752 (75.2%)
        #   Avg unique docs per activity:    1.223  (within provided rows)
        #   Avg pages per document:          2.009 (within provided rows)
        chunked_by_activity_id = consolidate_rows_by_activity(rows_performance)
        chunked_by_activity_id = load_summaries_from_chatgpt_into_bundles(
            chunked_by_activity_id,
            CHATGPT_SUMMARIES_JSONL,
        )
        prompts_performance = get_implementer_performance_prompt(chunked_by_activity_id,info)
        response_schema = get_description_schema()
        pprint.pprint("chunked_by_activity_id")
        pprint.pprint(chunked_by_activity_id)
        # quit()
        asyncio.run(loop_over_rows_to_call_model(output_jsonl, chunked_by_activity_id, prompts_performance, response_schema, execpool=execpool))
        print(f"\n(END PROGRAM): took {(datetime.now() - _program_start).total_seconds():.2f}s\n\n\n\n")  # <-- add this
        # view_rows_as_pdf(rows_partner_abilities)
        # quit()






        output_jsonl = str(DATA_DIR / "outputs_targets.jsonl")

        # generally looks good, unsurprisingly
        rows_projections = load_and_filter_rows(
            section="Baseline",
            subcategory_a=["possible_outcomes","quantitative_targets","qualitative_targets","detailed_implementation_plans","broad_objectives","possible_outcomes"],
            subcategory_b=["possible_outcomes","quantitative_targets","qualitative_targets","detailed_implementation_plans","broad_objectives","possible_outcomes"],
            top_k_per_activity=5,  # adjust as needed; set to None to disable
            min_score=7,
            get_surrounding_if_not_enough=True,
            get_at_least_top_k=True
        )
        rows_projections = add_fallback_rows_for_missing_activities_strict_baseline(rows_projections, DATA_DIR, max_pages=10, rated_ids=set(rated.index))

        # print_activity_coverage(rows_projections)
        # quit()
        chunked_by_activity_id = consolidate_rows_by_activity(rows_projections)
        chunked_by_activity_id = load_summaries_from_chatgpt_into_bundles(
            chunked_by_activity_id,
            CHATGPT_SUMMARIES_JSONL,
        )
        prompts_targets = get_target_outcomes_prompt(chunked_by_activity_id,info)
        response_schema = get_description_schema()
        asyncio.run(loop_over_rows_to_call_model(output_jsonl, chunked_by_activity_id, prompts_targets, response_schema, execpool=execpool))
        # print(f"\n(END PROGRAM): took {(datetime.now() - _program_start).total_seconds():.2f}s\n\n\n\n")  # <-- add this
        # view_rows_as_pdf(rows_projections)
        # quit()
        # Scope:                           section='baseline'
        # Matched unique activities:       439
        # Denominator (unique activities): 626
        # Fraction:                        439/626 = 0.701 (70.1%)
        # Avg unique docs per activity:    1.146  (within provided rows)
        # Avg pages per document:          3.288 (within provided rows)


        # # quit()


        output_jsonl = str(DATA_DIR / "outputs_risks.jsonl")


        rows_risks = load_and_filter_rows(
            section="Baseline",
            subcategory_a=["risks_as_word_or_numeric", "risks_or_dangers_generally","contextual_challenges"],
            subcategory_b=["risks_as_word_or_numeric", "risks_or_dangers_generally","contextual_challenges"],
            top_k_per_activity=5,  # adjust as needed; set to None to disable
            min_score=7,
            get_surrounding_if_not_enough=True,
            get_at_least_top_k=True
        )
        rows_risks = add_fallback_rows_for_missing_activities_strict_baseline(rows_risks, DATA_DIR, max_pages=10, rated_ids=set(rated.index))
        print("\n\nrows_risks")
        chunked_by_activity_id = consolidate_rows_by_activity(rows_risks)
        chunked_by_activity_id = load_summaries_from_chatgpt_into_bundles(
            chunked_by_activity_id,
            CHATGPT_SUMMARIES_JSONL,
        )
        prompts_risks = get_risks_prompt(chunked_by_activity_id)
        response_schema = get_description_schema()
        asyncio.run(loop_over_rows_to_call_model(output_jsonl, chunked_by_activity_id, prompts_risks, response_schema, execpool=execpool))

        # rows_risks
        # Activity coverage
        #   Scope:                           section='baseline'
        #   Matched unique activities:       589
        #   Denominator (unique activities): 626
        #   Fraction:                        589/626 = 0.941 (94.1%)
        #   Avg unique docs per activity:    1.630  (within provided rows)
        #   Avg pages per document:          2.478 (within provided rows)







        output_jsonl = str(DATA_DIR / "outputs_possibilities.jsonl")


        rows_possibilities = load_and_filter_rows(
            section="Baseline",
            subcategory_a=["positive_indicators", "plans_to_address_key_risks"],
            subcategory_b=["positive_indicators", "plans_to_address_key_risks"],
            min_score=7,
            top_k_per_activity=5,  # adjust as needed; set to None to disable
            get_surrounding_if_not_enough=True,
            get_at_least_top_k=True
        )
        rows_possibilities = add_fallback_rows_for_missing_activities_strict_baseline(rows_possibilities, DATA_DIR, max_pages=10, rated_ids=set(rated.index))
        print("\n\nrows_possibilities")
        chunked_by_activity_id = consolidate_rows_by_activity(rows_possibilities)
        chunked_by_activity_id = load_summaries_from_chatgpt_into_bundles(
            chunked_by_activity_id,
            CHATGPT_SUMMARIES_JSONL,
        )
        prompts_possibilities = get_possibilities_prompt(chunked_by_activity_id)
        response_schema = get_description_schema()
        asyncio.run(loop_over_rows_to_call_model(output_jsonl, chunked_by_activity_id, prompts_possibilities, response_schema, execpool=execpool))


        # these looked good, all looked at seemed like key risks
        # view_rows_as_pdf(rows_possibilities)
        # rows_possibilities
        # Activity coverage
        #   Scope:                           section='baseline'
        #   Matched unique activities:       427
        #   Denominator (unique activities): 626
        #   Fraction:                        427/626 = 0.682 (68.2%)
        #   Avg unique docs per activity:    1.274  (within provided rows)
        #   Avg pages per document:          1.781 (within provided rows)





        output_jsonl = str(DATA_DIR / "outputs_context.jsonl")

        rows_context = load_and_filter_rows(
            section="Baseline",
            subcategory_a="implementation_context_country",
            subcategory_b="implementation_context_country",
            min_score=7,
            top_k_per_activity=5,  # adjust as needed; set to None to disable
            get_surrounding_if_not_enough=True,
            get_at_least_top_k=True
        )
        rows_context = add_fallback_rows_for_missing_activities_strict_baseline(rows_context, DATA_DIR, max_pages=10, rated_ids=set(rated.index))

        # print("\n\nrows_country_context")
        chunked_by_activity_id = consolidate_rows_by_activity(rows_context)
        chunked_by_activity_id = load_summaries_from_chatgpt_into_bundles(
            chunked_by_activity_id,
            CHATGPT_SUMMARIES_JSONL,
        )
        prompts_context = get_context_prompt(chunked_by_activity_id)
        response_schema = get_description_schema()
        asyncio.run(loop_over_rows_to_call_model(output_jsonl, chunked_by_activity_id, prompts_context, response_schema, execpool=execpool))


        # quit()
        # rows_country_context
        # Activity coverage
        #   Scope:                           section='baseline'
        #   Matched unique activities:       598
        #   Denominator (unique activities): 626
        #   Fraction:                        598/626 = 0.955 (95.5%)
        #   Avg unique docs per activity:    1.599  (within provided rows)
        #   Avg pages per document:          2.537 (within provided rows)








        output_jsonl = str(DATA_DIR / "outputs_finance.jsonl")

        rows_financial = load_and_filter_rows(
            section="Baseline",
            subcategory_a=["financing_details", "budget_and_legal"],
            subcategory_b=["financing_details", "budget_and_legal"],
            min_score=7,
            top_k_per_activity=5,  # adjust as needed; set to None to disable
            get_surrounding_if_not_enough=True,
            get_at_least_top_k=True
        )
        rows_financial = add_fallback_rows_for_missing_activities_strict_baseline(rows_financial, DATA_DIR, max_pages=10, rated_ids=set(rated.index))

        # view_rows_as_pdf(rows_financial)
        # print_activity_coverage(rows_financial)
        # quit()

        # print("\n\nrows_country_context")
        chunked_by_activity_id = consolidate_rows_by_activity(rows_financial)
        chunked_by_activity_id = load_summaries_from_chatgpt_into_bundles(
            chunked_by_activity_id,
            CHATGPT_SUMMARIES_JSONL,
        )
        prompts_financial = get_finance_prompt(chunked_by_activity_id)
        response_schema = get_description_schema()
        asyncio.run(loop_over_rows_to_call_model(output_jsonl, chunked_by_activity_id, prompts_financial, response_schema, execpool=execpool))




        output_jsonl = str(DATA_DIR / "outputs_misc.jsonl")

        rows_misc = load_and_filter_rows(
            section="Baseline",
            subcategory_a=[
                    "condensed_summary",
                    "sub_activities_outlined",
                    "detailed_implementation_plans",
                    "broad_objectives",
                    "possible_outcomes",
                    "quantitative_targets",
                    "qualitative_targets",
                    "risks_as_word_or_numeric",
                    "risks_or_dangers_generally",
                    "plans_to_address_key_risks",
                    "positive_indicators",
                    "progress_reports",
                    "similar_cases_outcomes",
                    "implementation_context_country",
                    "contextual_challenges",
                    "financing_details",
                    "budget_and_legal",
                    "who_implements",
                    "whether_part_of_larger_program",
                    "partner_identity_or_skill",
                    "whether_skin_in_the_game",
                    "other_stakeholder_engagement",
                ],
            subcategory_b=[
                    "condensed_summary",
                    "sub_activities_outlined",
                    "detailed_implementation_plans",
                    "broad_objectives",
                    "possible_outcomes",
                    "quantitative_targets",
                    "qualitative_targets",
                    "risks_as_word_or_numeric",
                    "risks_or_dangers_generally",
                    "plans_to_address_key_risks",
                    "positive_indicators",
                    "progress_reports",
                    "similar_cases_outcomes",
                    "implementation_context_country",
                    "contextual_challenges",
                    "financing_details",
                    "budget_and_legal",
                    "who_implements",
                    "whether_part_of_larger_program",
                    "partner_identity_or_skill",
                    "whether_skin_in_the_game",
                    "other_stakeholder_engagement",
                ],
            min_score=9,
            top_k_per_activity=5,  # adjust as needed; set to None to disable
            get_surrounding_if_not_enough=True,
            get_at_least_top_k=True
        )
        rows_misc = add_fallback_rows_for_missing_activities_strict_baseline(rows_misc, DATA_DIR, max_pages=10, rated_ids=set(rated.index))

        # print_activity_coverage(rows_misc)
        # view_rows_as_pdf(rows_misc)
        # print("\n\nrows_country_context")
        chunked_by_activity_id = consolidate_rows_by_activity(rows_misc)
        chunked_by_activity_id = load_summaries_from_chatgpt_into_bundles(
            chunked_by_activity_id,
            CHATGPT_SUMMARIES_JSONL,
        )
        prompts_misc = get_misc_prompt(chunked_by_activity_id)
        response_schema = get_few_grades_schema()
        asyncio.run(loop_over_rows_to_call_model(output_jsonl, chunked_by_activity_id, prompts_misc, response_schema, execpool=execpool))

    finally:
        execpool.shutdown(wait=False, cancel_futures=True)
