import streamlit as st
import plotly.graph_objects as go

from tag_model_loader import load_tag_models


_TAG_META = {
    "tag_funds_cancelled_or_unutilized":                  {"label": "Funds Cancelled or Unutilized",             "definition": "Project funds were cancelled, not disbursed, or significantly underutilized."},
    "tag_funds_reallocated":                               {"label": "Funds Reallocated",                         "definition": "Project funds were reallocated across components or activities."},
    "tag_high_disbursement":                               {"label": "High Disbursement",                         "definition": "Project achieved high or full disbursement of funds."},
    "tag_external_factors_affected_outcomes":              {"label": "External Factors Affected Outcomes",        "definition": "External factors such as political, economic, or environmental conditions negatively affected project implementation or outcomes."},
    "tag_activities_not_completed":                        {"label": "Activities Not Completed",                  "definition": "Some planned activities, outputs, or components were not fully completed or were dropped."},
    "tag_design_or_appraisal_shortcomings":                {"label": "Design or Appraisal Shortcomings",          "definition": "Deficiencies were present in the original project design, appraisal, or objective clarity."},
    "tag_monitoring_and_evaluation_challenges":            {"label": "Monitoring and Evaluation Challenges",      "definition": "Challenges with data collection, M&E systems, or reporting were encountered."},
    "tag_improved_financial_performance":                  {"label": "Improved Financial Performance",            "definition": "Improved revenues, profitability, financial management, or financial capacity of beneficiary institutions were significantly achieved."},
    "tag_project_restructured":                            {"label": "Project Restructured",                      "definition": "Project was formally restructured, including changes to scope, components, or objectives."},
    "tag_implementation_delays":                           {"label": "Implementation Delays",                     "definition": "Project experienced significant implementation delays."},
    "tag_closing_date_extended":                           {"label": "Closing Date Extended",                     "definition": "Project closing date was extended beyond the original plan."},
    "tag_targets_revised":                                 {"label": "Targets Revised",                           "definition": "Project targets, indicators, or results framework were revised during implementation."},
    "tag_policy_regulatory_reforms_success_success":       {"label": "Policy and Regulatory Reforms",             "definition": "Policy, regulatory, or legal frameworks were significantly and successfully developed, strengthened, or operationalized."},
    "tag_targets_met_or_exceeded_success":                 {"label": "Targets Met or Exceeded",                   "definition": "Project targets were met as planned, or exceeded."},
    "tag_over_budget_success":                             {"label": "Over Budget",                               "definition": "Project had to exceed the original budget."},
    "tag_capacity_building_delivered_success":             {"label": "Capacity Building Delivered",               "definition": "Capacity building, training, or institutional strengthening activities were significantly delivered."},
    "tag_infrastructure_completed_success":                {"label": "Infrastructure Completed",                  "definition": "Physical infrastructure was constructed, rehabilitated, or delivered as planned."},
    "tag_high_beneficiary_satisfaction_or_reach_success":  {"label": "High Beneficiary Satisfaction or Reach",   "definition": "High beneficiary satisfaction, broad beneficiary reach, or high uptake was achieved."},
    "tag_gender_equitable_outcomes_success":               {"label": "Gender-Equitable Outcomes",                 "definition": "Outcomes affecting women and girls were equitable or positive."},
    "tag_improved_service_delivery_success":               {"label": "Improved Service Delivery",                 "definition": "Improved access to, quality of, or efficiency in service delivery was significantly achieved."},
    "tag_improved_livelihoods_success":                    {"label": "Improved Livelihoods",                      "definition": "Improved livelihoods, incomes, or agricultural outcomes for beneficiaries were significantly achieved."},
    "tag_energy_sector_improvements_success":              {"label": "Energy Sector Improvements",                "definition": "Improved electricity supply, reliability, access, or generation capacity was delivered at or above targets."},
    "tag_private_sector_engagement_success":               {"label": "Private Sector Engagement",                 "definition": "Success in attracting private sector investment or improving the business environment was achieved at or above targets."},
}

# 4 sections: only model-predicted tags appear in charts; const_base shown at bottom
_TAG_SECTIONS = [
    {
        "title": "Financial Management Outcomes",
        "tags": [
            "tag_funds_cancelled_or_unutilized",
            "tag_funds_reallocated",
            "tag_high_disbursement",
            "tag_improved_financial_performance",
            "tag_over_budget_success",
        ],
        "positive": False,  # higher prob = potential issue (amber/red)
    },
    {
        "title": "Implementation Changes and Adjustments",
        "tags": [
            "tag_closing_date_extended",
            "tag_project_restructured",
            "tag_targets_revised",
            "tag_activities_not_completed",
            "tag_implementation_delays",
        ],
        "positive": False,
    },
    {
        "title": "Infrastructure and Sector Delivery",
        "tags": [
            "tag_infrastructure_completed_success",
            "tag_energy_sector_improvements_success",
            "tag_capacity_building_delivered_success",
            "tag_improved_service_delivery_success",
            "tag_policy_regulatory_reforms_success_success",
        ],
        "positive": True,  # higher prob = good (green)
    },
    {
        "title": "Beneficiary Reach and Livelihood Outcomes",
        "tags": [
            "tag_high_beneficiary_satisfaction_or_reach_success",
            "tag_improved_livelihoods_success",
            "tag_targets_met_or_exceeded_success",
            "tag_gender_equitable_outcomes_success",
            "tag_private_sector_engagement_success",
        ],
        "positive": True,
    },
]


def _is_model_predicted(model_dict: dict) -> bool:
    """True if this tag has an actual trained model (not just a base_rate)."""
    return bool(model_dict) and ("rf" in model_dict or "ridge" in model_dict)


def _tag_section_chart(section: dict, preds: dict, tag_data: dict) -> None:
    """Render one horizontal bar chart for a section of tags."""
    models = tag_data.get("models", {}) if tag_data else {}
    base_rates = tag_data.get("tag_base_rates", {}) if tag_data else {}

    # Only include model-predicted tags from this section that have predictions
    tags_in_section = [
        t for t in section["tags"]
        if t in preds and _is_model_predicted(models.get(t, {}))
    ]
    if not tags_in_section:
        st.caption("*(No model-predicted tags in this section for this activity.)*")
        return

    # Sort by predicted probability descending
    tags_in_section = sorted(tags_in_section, key=lambda t: preds[t], reverse=True)

    from tag_predictor import get_model_type_label
    labels    = [_TAG_META.get(t, {}).get("label", t) for t in tags_in_section]
    probs     = [preds[t] for t in tags_in_section]
    defs      = [_TAG_META.get(t, {}).get("definition", "") for t in tags_in_section]
    mtypes    = [get_model_type_label(models.get(t, {})) for t in tags_in_section]
    db_rates  = [base_rates.get(t) for t in tags_in_section]

    positive = section["positive"]

    # Bar colours
    colors = []
    for p in probs:
        if positive:
            colors.append("mediumseagreen" if p >= 0.6 else "steelblue" if p >= 0.35 else "lightsteelblue")
        else:
            colors.append("tomato" if p >= 0.6 else "goldenrod" if p >= 0.35 else "steelblue")

    # Y-axis tick labels include model type in grey
    ytick_labels = [
        f"{lbl} <span style='color:#aaa;font-size:10px'>({mt})</span>"
        for lbl, mt in zip(labels, mtypes)
    ]

    height_per_bar = 80
    fig_height = len(tags_in_section) * height_per_bar + 55

    fig = go.Figure()

    # Main prediction bars
    fig.add_trace(go.Bar(
        x=probs,
        y=ytick_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.0%}" for p in probs],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>Predicted probability: %{x:.1%}<br>Database average: %{customdata[1]:.1%}<extra></extra>",
        customdata=[[lbl, dr] for lbl, dr in zip(labels, [r if r is not None else 0 for r in db_rates])],
        showlegend=False,
    ))

    # DB prevalence markers (vertical tick at each base rate)
    valid_db = [(i, r) for i, r in enumerate(db_rates) if r is not None]
    if valid_db:
        db_x = [r for _, r in valid_db]
        db_y = [ytick_labels[i] for i, _ in valid_db]
        fig.add_trace(go.Scatter(
            x=db_x,
            y=db_y,
            mode="markers",
            marker=dict(
                symbol="line-ns-open",
                size=28,
                color="crimson",
                line=dict(width=2),
            ),
            name="Database average",
            hovertemplate="Database average: %{x:.1%}<extra></extra>",
            showlegend=True,
        ))

    fig.update_layout(
        xaxis=dict(title="Probability", range=[0, 1.18], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        height=fig_height,
        margin=dict(l=10, r=70, t=10, b=40),
        bargap=0.55,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Descriptions in grey beneath the chart
    for lbl, defn, prob in zip(labels, defs, probs):
        st.markdown(
            f"<span style='color:#888;font-size:12px'>**{lbl}** ({prob:.0%}) — {defn}</span>",
            unsafe_allow_html=True,
        )


def render_tag_predictions() -> None:
    """Display outcome tag probability predictions from session_state.tag_predictions."""
    if "tag_predictions" not in st.session_state or not st.session_state.tag_predictions:
        return

    preds: dict = st.session_state.tag_predictions

    st.markdown("---")
    st.header("Outcome Tag Probabilities")
    st.markdown(
        "Predicted probability that each outcome tag applies to this activity. "
        "The red tick marks show the database prevalence for each tag. "
        "Only tags with trained models are shown in the charts; "
        "tags where no model beat the baseline are listed at the bottom."
    )

    try:
        _tag_data = load_tag_models()
    except Exception:
        _tag_data = None

    for section in _TAG_SECTIONS:
        st.subheader(section["title"])
        _tag_section_chart(section, preds, _tag_data)

    # ---- Database prevalence for const_base tags ----
    models = (_tag_data.get("models", {}) if _tag_data else {})
    base_rates = (_tag_data.get("tag_base_rates", {}) if _tag_data else {})

    const_base_tags = [
        t for t in _TAG_META
        if t in preds and not _is_model_predicted(models.get(t, {}))
    ]

    if const_base_tags:
        st.markdown("---")
        st.subheader("Database Prevalence (no activity-specific model)")
        st.markdown(
            "<span style='color:#888;font-size:13px'>"
            "For the following tags, no trained model outperformed simply predicting "
            "the training-set average. The percentage below is the prevalence of each tag "
            "in the training database (not a prediction specific to this activity)."
            "</span>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, tag in enumerate(const_base_tags):
            lbl  = _TAG_META.get(tag, {}).get("label", tag)
            defn = _TAG_META.get(tag, {}).get("definition", "")
            rate = base_rates.get(tag, preds.get(tag, 0.0))
            with cols[i % 3]:
                st.metric(label=lbl, value=f"{rate:.0%}")
                st.markdown(
                    f"<span style='color:#aaa;font-size:11px'>{defn}</span>",
                    unsafe_allow_html=True,
                )
