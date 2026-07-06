from location_features import ORG_NAME_TO_DUMMY

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "finance": "How well-financed the activity is (0–100). Considers adequacy of the budget relative to planned scope and duration.",
    "integratedness": "How well the activity is integrated within the broader aid ecosystem (0–100). Higher scores indicate strong coordination with complementary programmes, government systems, and local institutions.",
    "implementer_performance": "Expected quality of the implementing organisation based on evidence in the documents (0–100).",
    "targets": "Ease and clarity of the targeted outcomes (0–100). Higher scores indicate well-defined, realistic, and measurable targets.",
    "context": "Degree of contextual challenge faced by the activity (0–100). Higher scores indicate a more favourable operating environment.",
    "risks": "Overall risk level of the activity (0–100). Higher scores indicate lower risk.",
    "complexity": "Technical complexity of the activity (0–100). Higher scores indicate greater complexity.",
    "gdp_percap": "log(GDP per capita) of the country or countries where the activity takes place, weighted by the share of activity performed in each country.",
    "cpia_score": "Country Policy and Institutional Assessment score from the World Bank. Rates the quality of a country's policies and institutions on a scale of 1–6.",
    "governance_composite": "Mean of the five World Governance Indicators (WGI): control of corruption, government effectiveness, political stability, regulatory quality, and rule of law. Computed only when none of the five components are missing.",
    "region_AFE": "Region dummy: Eastern and Southern Africa.",
    "region_AFW": "Region dummy: Western and Central Africa.",
    "region_EAP": "Region dummy: East Asia and Pacific.",
    "region_ECA": "Region dummy: Europe and Central Asia.",
    "region_LAC": "Region dummy: Latin America and the Caribbean.",
    "region_MENA": "Region dummy: Middle East and North Africa.",
    "region_SAS": "Region dummy: South Asia.",
    "activity_scope": "Geographic scope of the activity on a 1–7 scale: 1 = local/sub-national, 7 = global.",
    "finance_is_loan": "Binary flag (1/0) indicating whether the primary financing instrument is a loan rather than a grant.",
    "planned_duration": "Planned length of the activity in days, derived from the planned start and end dates.",
    "planned_expenditure": "Planned total disbursement in USD.",
    "log_planned_expenditure": "Natural log of planned total disbursement. Reduces the influence of extreme values; used directly in the model alongside the raw figure.",
    "expenditure_x_complexity": "Planned expenditure (raw USD) multiplied by the complexity grade. Captures interaction between scale and technical difficulty.",
    "expenditure_per_year_log": "log(planned expenditure / planned duration in years). Annualised spending rate.",
    "umap3_x": "First UMAP dimension of target-embedding space. Low values: forestry, agriculture, water management, natural resource objectives. High values: energy-sector and financing-oriented objectives.",
    "umap3_y": "Second UMAP dimension of target-embedding space. Low values: energy-focused activities. High values: biodiversity, wastewater, and conservation objectives.",
    "umap3_z": "Third UMAP dimension of target-embedding space. Primarily reflects an urban–rural axis. Low values: rural, wildlife, and UNDP-led activities. High values: urban sanitation and wastewater objectives.",
    "sector_distance": "Distance in UMAP embedding space between the activity and the centroid of its sector cluster. Measures how atypical the activity's objectives are relative to similar activities.",
    "country_distance": "Distance in UMAP embedding space between the activity and the centroid of activities in the same country. Measures how contextually unusual the activity's objectives are for that country.",
    "llm_features_missing_count": "Number of the seven LLM-graded features (finance, integratedness, etc.) that are absent.",
    "llm_features_present_ratio": "Fraction of the seven LLM-graded features that are present (1 - missing_count / 7).",
    "governance_missing_count": "Number of the five WGI governance indicators that are missing for the activity's country.",
    "feature_completeness_ratio": "Overall fraction of model features that have non-missing values for this activity.",
    "cpia_missing": "1 if the CPIA score is unavailable, 0 otherwise.",
    "sector_clusters_missing": "1 if no budget breakdown could be identified for sector cluster assignment, 0 otherwise.",
    "gdp_percap_missing": "1 if GDP per capita data is unavailable for the activity's country, 0 otherwise.",
    "planned_expenditure_missing": "1 if planned total disbursement is not reported, 0 otherwise.",
    "planned_duration_missing": "1 if planned start or end date is missing (duration cannot be computed), 0 otherwise.",
    "wgi_any_missing": "1 if any of the five WGI governance indicators are missing, 0 otherwise.",
    "umap_missing": "1 if UMAP coordinates could not be computed (e.g. targets text was absent), 0 otherwise.",
}

SECTOR_CLUSTER_DESCRIPTIONS: dict[str, str] = {
    "increased_food_production": "Interventions targeting increases in agricultural output and food security.",
    "Project_Management": "Budget allocated to project management, administration, and oversight.",
    "more_people_with_drinking_water_services": "Investments in drinking water access and supply infrastructure.",
    "reduced_CO2_emissions": "Activities targeting greenhouse gas / CO2 emission reductions.",
    "Improved_transport_infrastructure": "Road, rail, port, or other transport infrastructure improvements.",
    "Contingencies": "Contingency reserves built into the budget.",
    "Institutional_capacity_building": "Strengthening the capacity of government or partner institutions.",
    "Financing_charges": "General financing charges and fees.",
    "Urban_flood_protection": "Infrastructure and measures for urban flood risk reduction.",
    "Climate_adaptation": "Activities building resilience to climate change impacts.",
    "Capacity_Building_and_Technical_Assistance": "Capacity development and technical assistance for institutions.",
    "increased_managed_forest_land": "Increasing the area of sustainably managed or protected forest land.",
    "Green_growth_initiatives": "Programmes promoting environmentally sustainable economic growth.",
    "Road_safety_improvements": "Measures reducing road traffic injuries and improving road safety.",
    "reduced_PM2.5_air_pollution": "Activities targeting reductions in fine-particulate (PM2.5) air pollution.",
    "Land_acquisition_and_resettlement": "Costs of land acquisition and resettlement of affected populations.",
}

_ORG_DUMMY_TO_NAME: dict[str, str] = {
    col: org
    for org, dummies in ORG_NAME_TO_DUMMY.items()
    for col, val in dummies.items()
    if val == 1
}


def get_feature_description(feature_name: str) -> str:
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]
    if feature_name.startswith("sector_cluster_"):
        cluster = feature_name[len("sector_cluster_"):]
        return SECTOR_CLUSTER_DESCRIPTIONS.get(
            cluster,
            f"Budget share allocated to sector cluster: {cluster.replace('_', ' ')}."
        )
    if feature_name in _ORG_DUMMY_TO_NAME:
        return f"Reporting organisation: {_ORG_DUMMY_TO_NAME[feature_name]}."
    if feature_name.startswith("rep_org_"):
        return f"Reporting organisation dummy variable ({feature_name})."
    return feature_name.replace("_", " ").title()
