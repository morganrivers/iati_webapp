"""
Feature Grading Module for Webapp
Adapted from src/forecast_outcomes/A_grade_baseline_features_gpt3p5.py
Grades extracted features using LLM prompts
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache

# Add necessary paths

import logging

logger = logging.getLogger(__name__)

UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from extracting_and_grading_helper_functions import loop_over_rows_to_call_model, AIRPLANE_MODE
from dummy_response_text_generator import DUMMY_GRADE_RESPONSE

ACTIVITY_INFO_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "info_for_activity_forecasting.csv"

@lru_cache(None)
def _load_activity_info() -> Dict[str, Dict[str, str]]:
    """Map activity_id -> row from the constructed CSV."""
    out: Dict[str, Dict[str, str]] = {}
    if not ACTIVITY_INFO_CSV.exists():
        return out
    with ACTIVITY_INFO_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            aid = (r.get("activity_id") or "").strip()
            if aid:
                out[aid] = r
    return out


def get_implementer_performance_prompt(
    activity_id: str,
    title: str,
    chatgpt_description: str,
    performance_summary: str,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """Generate prompt for implementer performance grading."""
    if not performance_summary or performance_summary == "NO RESPONSE":
        return None
    if not chatgpt_description or chatgpt_description == "NO RESPONSE":
        return None

    # Get metadata
    locations = metadata.get("country_location", "")
    gdp_percap = metadata.get("gdp_percap", "")
    orgs = metadata.get("participating_orgs", "")
    implementing_org_type = metadata.get("implementing_org_type", "")

    prompt = f"""Provide a single grade between 0 (extremely bad) and 100 (extremely good) for the likely performance of the organization(s) in implementing the activity. The grade should reflect whether the partner organization is truly invested in the project, and potentially how other relevant stakeholders would be aiding or deteriorating project outcomes. Where relevant consider how the people implementing the activity in the organization may perform the activity well, or perform badly, as well as baseline rates of performance in this context. If a grade cannot be assigned, respond only with: "GRADE: NO RESPONSE". Otherwise respond only with "GRADE: " and then the grade.
ACTIVITY TITLE: {title}"""

    if locations:
        prompt += f"\nACTIVITY LOCATION(S): {locations}"
    if gdp_percap:
        try:
            prompt += f"\nLOCATION GDP PER CAPITA, USD: {int(float(gdp_percap))}"
        except (ValueError, TypeError):
            pass
    if orgs:
        prompt += f"\nPARTICIPATING ORGANIZATIONS: {orgs}"
    if implementing_org_type:
        prompt += f"\nIMPLEMENTING ORGANIZATION CATEGORY: {implementing_org_type}"
    if performance_summary and performance_summary != "NO RESPONSE":
        prompt += f"\nPERFORMANCE SUMMARY: {performance_summary}"

    return prompt


def get_finance_prompt(
    activity_id: str,
    title: str,
    chatgpt_description: str,
    finance_summary: str,
    metadata: Dict[str, Any],
    total_loans: str = "",
    units_loans: str = "",
    total_disbursement: str = "",
    units_disbursement: str = ""
) -> Optional[str]:
    """Generate prompt for finance quality grading."""
    if not chatgpt_description or chatgpt_description == "NO RESPONSE":
        return None

    # Get metadata
    locations = metadata.get("country_location", "")
    orgs = metadata.get("participating_orgs", "")
    activity_scope = metadata.get("activity_scope", "")
    planned_start_date = metadata.get("original_planned_start_date") or metadata.get("actual_start_date")
    planned_end_date = metadata.get("original_planned_close_date")

    prompt = f"""Provide a numeric grade (0-100) for how well financed the activity is compared to the scope of its objectives, where 100 is very well financed with excellent resources for overall success given its size and scope, and 0 is a very challenging financial situation for the activity to succeed. If a grade cannot be assigned, respond only with: "GRADE: NO RESPONSE". Otherwise respond only with "GRADE: " and then the numeric grade (0-100).
ACTIVITY TITLE: {title}"""

    if planned_start_date:
        prompt += f"\nPLANNED START DATE: {planned_start_date}"
    if planned_end_date:
        prompt += f"\nPLANNED END DATE: {planned_end_date}"
    if locations:
        prompt += f"\nACTIVITY LOCATION(S): {locations}"
    if activity_scope:
        prompt += f"\nACTIVITY SCOPE: {activity_scope}"
    if orgs:
        prompt += f"\nPARTICIPATING ORGANIZATIONS: {orgs}"
    if chatgpt_description and chatgpt_description != "NO RESPONSE":
        prompt += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"
    if total_disbursement and total_disbursement != "NO RESPONSE":
        prompt += f"\nPLANNED TOTAL DISBURSEMENT: {total_disbursement} {units_disbursement}"
    if total_loans and total_loans != "NO RESPONSE":
        prompt += f"\nPLANNED TOTAL LOANS AND CREDIT: {total_loans} {units_loans}"
    if finance_summary and finance_summary != "NO RESPONSE":
        prompt += f"\nACTIVITY FINANCING: {finance_summary}"

    return prompt


def get_integratedness_prompt(
    activity_id: str,
    title: str,
    chatgpt_description: str,
    how_integrated_description: str,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """Generate prompt for integratedness grading (activity size/cohesiveness)."""
    if not chatgpt_description or chatgpt_description == "NO RESPONSE":
        return None

    # Get metadata
    locations = metadata.get("country_location", "")
    orgs = metadata.get("participating_orgs", "")
    activity_scope = metadata.get("activity_scope", "")
    planned_start_date = metadata.get("original_planned_start_date") or metadata.get("actual_start_date")
    planned_end_date = metadata.get("original_planned_close_date")

    prompt = f"""Provide a grade for the cohesiveness between this activity and the ecosystem of similar implemented activities, where 100 is a very cohesive, large, well-integrated program, and 0 is a very independent, one-off small program. If a grade cannot be assigned, respond only with: "GRADE: NO RESPONSE". Otherwise respond only with "GRADE: " and then the grade.
ACTIVITY TITLE: {title}"""

    if planned_start_date:
        prompt += f"\nPLANNED START DATE: {planned_start_date}"
    if planned_end_date:
        prompt += f"\nPLANNED END DATE: {planned_end_date}"
    if locations:
        prompt += f"\nACTIVITY LOCATION(S): {locations}"
    if activity_scope:
        prompt += f"\nACTIVITY SCOPE: {activity_scope}"
    if orgs:
        prompt += f"\nPARTICIPATING ORGANIZATIONS: {orgs}"
    if chatgpt_description and chatgpt_description != "NO RESPONSE":
        prompt += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"

    return prompt


def get_target_outcomes_prompt(
    activity_id: str,
    title: str,
    chatgpt_description: str,
    target_outcomes_summary: str,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """Generate prompt for target quality grading."""
    if not chatgpt_description or chatgpt_description == "NO RESPONSE":
        return None
    if not target_outcomes_summary or target_outcomes_summary == "NO RESPONSE":
        return None

    planned_start_date = metadata.get("original_planned_start_date") or metadata.get("actual_start_date")
    planned_end_date = metadata.get("original_planned_close_date")

    prompt = f"""Provide a single grade for the ease of achieving success in the activity, with 100 being extremely easily achieved and 0 being nearly impossible. This grade should reflect the overall degree of challenge faced, taking into account both reasons the targets may relatively easily achieved, and also reasons targets may be intrinsically challenging.  If a grade cannot be assigned, respond only with: "GRADE: NO RESPONSE". Otherwise respond only with "GRADE: " and then the grade.
ACTIVITY TITLE: {title}"""

    if planned_start_date:
        prompt += f"\nPLANNED START DATE: {planned_start_date}"
    if planned_end_date:
        prompt += f"\nPLANNED END DATE: {planned_end_date}"
    if chatgpt_description and chatgpt_description != "NO RESPONSE":
        prompt += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"
    if target_outcomes_summary and target_outcomes_summary != "NO RESPONSE":
        prompt += f"\nTARGETS SUMMARY: {target_outcomes_summary}"

    return prompt


def get_context_prompt(
    activity_id: str,
    title: str,
    chatgpt_description: str,
    context_summary: str,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """Generate prompt for context grading."""
    if not chatgpt_description or chatgpt_description == "NO RESPONSE":
        return None

    # Get metadata
    locations = metadata.get("country_location", "")
    gdp_percap = metadata.get("gdp_percap", "")
    orgs = metadata.get("participating_orgs", "")
    planned_start_date = metadata.get("original_planned_start_date") or metadata.get("actual_start_date")
    planned_end_date = metadata.get("original_planned_close_date")

    prompt = f"""Provide a grade for the influence of external factors, where 100 is a very good context conducive to activity success, and 0 is a very challenging context. Your grade should reflect how external factors may influence activity success, outside of the context of the organization and the specific implementers, and factors like the specific challenges of implementing activities in that country, economic and political conditions, base rates of success in such contexts, and the possibilities for external disruption. If a grade cannot be assigned, respond only with: "GRADE: NO RESPONSE". Otherwise respond only with "GRADE: " and then the grade.
ACTIVITY TITLE: {title}"""

    if planned_start_date:
        prompt += f"\nPLANNED START DATE: {planned_start_date}"
    if planned_end_date:
        prompt += f"\nPLANNED END DATE: {planned_end_date}"
    if locations:
        prompt += f"\nACTIVITY LOCATION(S): {locations}"
    if gdp_percap:
        try:
            prompt += f"\nLOCATION GDP PER CAPITA, USD: {int(float(gdp_percap))}"
        except (ValueError, TypeError):
            pass
    if chatgpt_description and chatgpt_description != "NO RESPONSE":
        prompt += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"
    if context_summary and context_summary != "NO RESPONSE":
        prompt += f"\nACTIVITY CONTEXT: {context_summary}"

    return prompt


def get_risks_prompt(
    activity_id: str,
    title: str,
    chatgpt_description: str,
    risks_summary: str,
    metadata: Dict[str, Any],
    possibilities_summary: str = ""
) -> Optional[str]:
    """Generate prompt for risks grading."""
    if not chatgpt_description or chatgpt_description == "NO RESPONSE":
        return None
    if not risks_summary or risks_summary == "NO RESPONSE":
        return None

    planned_start_date = metadata.get("original_planned_start_date") or metadata.get("actual_start_date")
    planned_end_date = metadata.get("original_planned_close_date")

    prompt = f"""Provide a single grade reflecting the overall level of risk for the project to not be successful, with 100 being very little risk (and thus, likely to be very successful), and 0 being extremely high risk. If a grade cannot be assigned, respond only with: "GRADE: NO RESPONSE". Otherwise respond only with "GRADE: " and then the grade.
ACTIVITY TITLE: {title}"""

    if planned_start_date:
        prompt += f"\nPLANNED START DATE: {planned_start_date}"
    if planned_end_date:
        prompt += f"\nPLANNED END DATE: {planned_end_date}"
    if chatgpt_description and chatgpt_description != "NO RESPONSE":
        prompt += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"
    if risks_summary and risks_summary != "NO RESPONSE":
        prompt += f"\nRISKS SUMMARY: {risks_summary}"
    if possibilities_summary and possibilities_summary != "NO RESPONSE":
        prompt += f"\nPOSSIBILITIES SUMMARY: {possibilities_summary}"

    return prompt


def get_complexity_prompt(
    activity_id: str,
    title: str,
    chatgpt_description: str,
    complexity_summary: str,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """Generate prompt for complexity grading."""
    if not chatgpt_description or chatgpt_description == "NO RESPONSE":
        return None
    if not complexity_summary or complexity_summary == "NO RESPONSE":
        return None

    # Get metadata
    locations = metadata.get("country_location", "")
    orgs = metadata.get("participating_orgs", "")
    activity_scope = metadata.get("activity_scope", "")
    planned_start_date = metadata.get("original_planned_start_date") or metadata.get("actual_start_date")
    planned_end_date = metadata.get("original_planned_close_date")

    prompt = f"""Provide a grade for the technical complexity and other concerning complexities of the activity in terms of how it may impact overall success, where 100 is very simple to implement and 0 is very complex. If a grade cannot be assigned, respond only with: "GRADE: NO RESPONSE". Otherwise respond only with "GRADE: " and then the grade.
ACTIVITY TITLE: {title}"""

    if planned_start_date:
        prompt += f"\nPLANNED START DATE: {planned_start_date}"
    if planned_end_date:
        prompt += f"\nPLANNED END DATE: {planned_end_date}"
    if locations:
        prompt += f"\nACTIVITY LOCATION(S): {locations}"
    if activity_scope:
        prompt += f"\nACTIVITY SCOPE: {activity_scope}"
    if orgs:
        prompt += f"\nPARTICIPATING ORGANIZATIONS: {orgs}"
    if chatgpt_description and chatgpt_description != "NO RESPONSE":
        prompt += f"\nACTIVITY DESCRIPTION: {chatgpt_description}"
    if complexity_summary and complexity_summary != "NO RESPONSE":
        prompt += f"\nACTIVITY COMPLEXITY: {complexity_summary}"

    return prompt


def parse_grade_response(response_text: str) -> Optional[float]:
    """
    Extract numeric grade from LLM response.
    Expected format: "GRADE: 75" or just "75"
    """
    if not response_text:
        return None

    # Remove "GRADE:" prefix if present
    text = response_text.strip()
    if text.upper().startswith("GRADE:"):
        text = text[6:].strip()

    # Handle "NO RESPONSE"
    if "NO RESPONSE" in text.upper():
        return None

    # Try to extract first number
    import re
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        try:
            grade = float(numbers[0])
            # Clamp to 0-100
            return max(0.0, min(100.0, grade))
        except ValueError:
            pass

    return None


async def grade_features_with_llm(
    activity_id: str,
    title: str,
    chatgpt_description: str,
    features: Dict[str, str],
    metadata: Dict[str, Any],
    model: str = "gemini-2.5-flash",
    log_callback = None
) -> Dict[str, float]:
    """
    Grade extracted features using LLM prompts.

    Args:
        activity_id: IATI activity ID
        title: Activity title
        chatgpt_description: LLM-generated activity description
        features: Dict of extracted feature summaries from feature_extractor
        metadata: Activity metadata (location, dates, orgs, etc.)
        model: LLM model to use
        log_callback: Optional callback for logging

    Returns:
        Dict mapping feature names to grades (0-100)
    """
    grades = {}

    # Map of feature names to prompt generators
    # Note: feature_extractor returns keys like 'implementer_performance', 'targets', etc.
    # with the actual summary text as values
    prompt_generators = {
        "implementer_performance": lambda: get_implementer_performance_prompt(
            activity_id, title, chatgpt_description,
            features.get("implementer_performance", ""),  # The summary text itself
            metadata
        ),
        "finance": lambda: get_finance_prompt(
            activity_id, title, chatgpt_description,
            features.get("finance", ""),  # The summary text itself
            metadata,
            features.get("total_loans", ""),
            features.get("units_loans", ""),
            features.get("total_disbursement", ""),
            features.get("units_disbursement", "")
        ),
        "integratedness": lambda: get_integratedness_prompt(
            activity_id, title, chatgpt_description,
            features.get("integratedness", ""),  # The summary text itself
            metadata
        ),
        "targets": lambda: get_target_outcomes_prompt(
            activity_id, title, chatgpt_description,
            features.get("targets", ""),  # The summary text itself
            metadata
        ),
        "context": lambda: get_context_prompt(
            activity_id, title, chatgpt_description,
            features.get("context", ""),  # The summary text itself
            metadata
        ),
        "risks": lambda: get_risks_prompt(
            activity_id, title, chatgpt_description,
            features.get("risks", ""),  # The summary text itself
            metadata,
            features.get("possibilities", "")  # If this exists
        ),
        "complexity": lambda: get_complexity_prompt(
            activity_id, title, chatgpt_description,
            features.get("complexity", ""),  # The summary text itself
            metadata
        ),
    }

    # Generate prompts and call LLM
    for feature_name, prompt_gen in prompt_generators.items():
        try:
            prompt = prompt_gen()
            if prompt is None:
                if log_callback:
                    log_callback(f"⚠️ Skipping {feature_name}: insufficient data")
                continue

            if log_callback:
                log_callback(f"🔍 Grading {feature_name}...")

            # Create a simple bundle for loop_over_rows_to_call_model
            bundle = [{
                "activity_id": activity_id,
                "prompt": prompt
            }]
            prompts_dict = {activity_id: prompt}

            # Generate response
            if AIRPLANE_MODE:
                response_text = DUMMY_GRADE_RESPONSE
            else:
                from google import genai
                import os

                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in environment")

                client = genai.Client(api_key=api_key)

                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=prompt
                )
                response_text = response.text if hasattr(response, 'text') else str(response)

            # Parse grade
            grade = parse_grade_response(response_text)
            if grade is not None:
                grades[feature_name] = grade
                if log_callback:
                    log_callback(f"✅ {feature_name}: {grade:.1f}")
            else:
                if log_callback:
                    log_callback(f"⚠️ {feature_name}: could not parse grade from response")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.exception(f"ERROR grading {feature_name}:")
            if log_callback:
                log_callback(f"❌ Error grading {feature_name}: {str(e)}")
                log_callback(f"Traceback: {error_trace}")
            continue

    return grades
