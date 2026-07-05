"""
Dummy LLM responses for AIRPLANE_MODE testing.

Returns pre-crafted responses that match the expected format for each prompt type,
so the full webapp pipeline can run end-to-end without any real API calls.
"""
import json
import re


# ── Grade response (used by feature_grader.py direct calls) ──────────────────
DUMMY_GRADE_RESPONSE = "GRADE: 65"

# ── Plain-text feature responses ──────────────────────────────────────────────
DUMMY_ACTIVITY_SUMMARY = (
    "This dummy activity supports environmental sustainability in developing countries "
    "through targeted investments in renewable energy, water management, and forest "
    "conservation. The project partners with local government and international "
    "implementing agencies to build capacity and deliver measurable outcomes.\n\n"
    "Key objectives: installing renewable energy systems for 50,000 households, "
    "improving rural water access, and protecting 10,000 hectares of forest. "
    "Total financing of approximately USD 45 million over five years (2020–2025)."
)

DUMMY_IMPLEMENTER_PERFORMANCE = (
    "The implementing organisation has a solid track record in comparable environmental "
    "projects across the region. Established government relationships, experienced "
    "technical staff, and capable local NGO partners suggest good delivery capacity. "
    "Governance structures are clear with appropriate oversight mechanisms."
)

DUMMY_TARGETS_TEXT = (
    "Activity targets are clearly defined with measurable indicators: 500 renewable "
    "energy systems installed, clean water access for 10,000 households, and "
    "reforestation of 5,000 hectares. Targets are ambitious but achievable given "
    "available resources and the implementation timeline."
)

DUMMY_RISKS_TEXT = (
    "Main risks include political instability in target regions, procurement delays "
    "for specialised equipment, and potential community resistance to land-use changes. "
    "These are moderate and manageable through the proposed mitigation measures: "
    "community engagement programmes and contingency budgets."
)

DUMMY_CONTEXT_TEXT = (
    "The activity operates in a challenging but improving governance environment. "
    "GDP per capita in target countries is low but growing. The political context "
    "is relatively stable with strong government commitment to environmental "
    "sustainability. High climate vulnerability increases urgency and relevance."
)

DUMMY_FINANCE_TEXT = (
    "The activity is moderately well-financed at USD 45 million total. The budget "
    "is adequate for the proposed scope with limited contingency. Cost estimates "
    "appear reasonable against regional comparators. Disbursement is structured "
    "across five tranches tied to milestone achievement."
)

DUMMY_KNN_SUMMARY = (
    "Based on similar activities, this project resembles several moderately successful "
    "environmental programmes. Closest neighbours achieved Moderately Satisfactory to "
    "Satisfactory ratings, driven by strong government commitment and experienced "
    "implementing organisations. Key risk factors in comparable activities were "
    "procurement delays and community engagement challenges."
)

DUMMY_RAG_SYNTHESIS = (
    "Evidence from the activity documents suggests reasonable prospects for success. "
    "Baseline documents indicate strong government ownership, a credible implementation "
    "plan, and adequate financing relative to scope. Some procurement risks are noted. "
    "Environmental monitoring frameworks appear robust and the implementing organisation "
    "has relevant regional experience."
)

DUMMY_FORECAST_STAGE = (
    "Based on available evidence and comparable historical activities, I estimate the "
    "probability of a Satisfactory or better rating at approximately 0.65.\n\n"
    "Positive indicators: strong government commitment, experienced implementing "
    "organisation, and adequate financing. Key uncertainties: procurement timeline "
    "and community engagement in remote areas.\n\n"
    "Probability estimate: 0.65 (Moderately Satisfactory to Satisfactory most likely)."
)

DUMMY_PHRASEGEN = (
    "PHRASE 1: Community engagement and local governance capacity building\n"
    "PHRASE 2: Renewable energy infrastructure installation and grid connection\n"
    "PHRASE 3: Forest conservation land tenure and community benefit sharing\n"
    "PHRASE 4: Water access improvement and sanitation facility construction\n"
    "PHRASE 5: Monitoring and evaluation framework for environmental outcomes"
)


# ── Schema-type detection ─────────────────────────────────────────────────────

def _detect_schema_type(response_schema) -> str:
    if response_schema is None:
        return "plain_text"
    props = response_schema.get("properties", {})
    if "pages" in props:
        return "page_categorization"
    if "title" in props and "country_locations" in props:
        return "metadata"
    if "complexity_details" in props or "how_integrated_description" in props:
        return "misc_features"
    if "total_allocation" in props or "quantitative_outcome_allocations" in props:
        return "finance_breakdown"
    return "generic_json"


def _detect_prompt_type(prompt) -> str:
    """Infer intent from prompt_type field or prompt text content."""
    if isinstance(prompt, dict):
        pt = prompt.get("prompt_type", "")
        if pt:
            if "phrasegen" in pt or "rag_queries" in pt or "generate_rag" in pt:
                return "phrasegen"
            if "knn_summary" in pt or "summarize_knn" in pt:
                return "knn_summary"
            if "rag_synthesis" in pt:
                return "rag_synthesis"
            # stage tags end like _s1, _s2, _s3
            if re.search(r'_s[123]$', pt):
                return "forecast_stage"
        prompt_text = prompt.get("prompt", "")
    else:
        prompt_text = str(prompt or "")

    lower = prompt_text.lower()
    if "search phrase" in lower or (("generate" in lower or "provide") and "phrase" in lower):
        return "phrasegen"
    if "summarize" in lower and ("similar" in lower or "nearest" in lower):
        return "knn_summary"
    if "rag" in lower or "synthesis" in lower:
        return "rag_synthesis"
    if "probability" in lower and ("satisfactory" in lower or "rating" in lower):
        return "forecast_stage"
    if "implementer" in lower:
        return "implementer_performance"
    if "target" in lower and "outcome" in lower:
        return "targets"
    if "risk" in lower and "grade" in lower:
        return "risks"
    if "context" in lower and "external" in lower:
        return "context"
    if "financ" in lower and "grade" in lower:
        return "finance_text"
    if "summar" in lower or "description" in lower:
        return "summary"
    return "generic_text"


# ── Structured dummy builders ─────────────────────────────────────────────────

def _make_dummy_page_categorization(activity_id: str) -> str:
    """One dummy page entry per page in the batch (parsed from the batch activity_id)."""
    n_pages = 1
    m = re.search(r'_batch_(\d+)_(\d+)$', str(activity_id))
    if m:
        n_pages = max(1, int(m.group(2)) - int(m.group(1)))

    pages = [
        {
            "category": "content",
            "subcategory_A": "broad_objectives",
            "subcategory_B": "condensed_summary",
            "informativeness": 6,
        }
        for _ in range(n_pages)
    ]
    return json.dumps({"pages": pages, "scratchpad": "Dummy categorisation for testing."})


def _make_dummy_metadata() -> str:
    return json.dumps({
        "title": "Dummy Environmental Aid Activity (AIRPLANE_MODE)",
        "participating_orgs": ["Dummy Implementing Agency", "Local Government Partner"],
        "country_locations": [{"iso2_code": "KE", "percentage": 100}],
        "planned_end_date": "2025-12-31",
    })


def _make_dummy_misc_features() -> str:
    return json.dumps({
        "complexity_details": (
            "The activity has moderate technical complexity with standard implementation "
            "challenges for the region."
        ),
        "how_integrated_description": (
            "This is a large integrated programme closely connected to other regional initiatives."
        ),
        "disbursement_total": "45",
        "disbursement_units": "USD millions",
        "loan_total": "",
        "loan_units": "",
        "is_RCT": "No",
    })


def _make_dummy_finance_breakdown() -> str:
    return json.dumps({
        "total_allocation": {"amount": 45, "currency": "million USD"},
        "quantitative_outcome_allocations": [
            {
                "outcome": "reduced CO2 emissions",
                "grant_or_loan": "loan",
                "amount_allocated": 22.5,
                "currency": "million USD",
            },
            {
                "outcome": "increased managed forest land",
                "grant_or_loan": "grant",
                "amount_allocated": 13.5,
                "currency": "million USD",
            },
            {
                "outcome": "more people with drinking water services",
                "grant_or_loan": "grant",
                "amount_allocated": 9.0,
                "currency": "million USD",
            },
        ],
    })


# ── Public entry point ────────────────────────────────────────────────────────

def get_dummy_response_text(response_schema, prompt, activity_id: str = "") -> str:
    """
    Return a dummy response string appropriate for the given schema and prompt.

    For JSON-schema calls the response is valid JSON matching the expected structure.
    For plain-text calls the response matches the expected natural-language format.
    """
    schema_type = _detect_schema_type(response_schema)

    if schema_type == "page_categorization":
        return _make_dummy_page_categorization(activity_id)
    if schema_type == "metadata":
        return _make_dummy_metadata()
    if schema_type == "misc_features":
        return _make_dummy_misc_features()
    if schema_type == "finance_breakdown":
        return _make_dummy_finance_breakdown()
    if schema_type == "generic_json":
        return json.dumps({"result": "dummy", "status": "ok"})

    # Plain text — detect from prompt content
    prompt_type = _detect_prompt_type(prompt)
    return {
        "phrasegen":             DUMMY_PHRASEGEN,
        "knn_summary":           DUMMY_KNN_SUMMARY,
        "rag_synthesis":         DUMMY_RAG_SYNTHESIS,
        "forecast_stage":        DUMMY_FORECAST_STAGE,
        "implementer_performance": DUMMY_IMPLEMENTER_PERFORMANCE,
        "targets":               DUMMY_TARGETS_TEXT,
        "risks":                 DUMMY_RISKS_TEXT,
        "context":               DUMMY_CONTEXT_TEXT,
        "finance_text":          DUMMY_FINANCE_TEXT,
        "summary":               DUMMY_ACTIVITY_SUMMARY,
        "generic_text":          DUMMY_ACTIVITY_SUMMARY,
    }.get(prompt_type, DUMMY_ACTIVITY_SUMMARY)
