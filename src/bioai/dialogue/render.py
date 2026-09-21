"""Deterministic rendering of an assessment to markdown.

This is not a fallback of last resort - it is the reference output. When the
language model is available it paraphrases this same content; when it is not,
this is what the user sees, and it is required to be complete and readable on its
own. Keeping a renderer that never calls out is also what makes the output
testable.
"""
from __future__ import annotations

from enum import Enum

from ..reasoning.causal import metric_label
from ..schemas import ClarifyingQuestion, SiteAssessment, SiteProfile

_HORIZON_TEXT = {
    "short": "short term (effect measurable within ~2 years)",
    "medium": "medium term (2-5 years)",
    "long": "long term (5+ years)",
}

_CONFIDENCE_TEXT = {
    "very_high": "very high",
    "high": "high",
    "moderate": "moderate",
    "low": "low",
}

_INDEX_TEXT = {
    "soil_health": "Soil health",
    "water_security": "Water security",
    "habitat_integrity": "Habitat integrity",
    "human_pressure": "Human pressure (higher = less pressure)",
    "biodiversity_state": "Biodiversity state",
}


# Units are appended on display so a bare "0.3" is never ambiguous.
_UNITS: dict[str, str] = {
    "soil_organic_carbon_pct": "%",
    "soil_moisture_pct": "% vol",
    "soil_bulk_density_g_cm3": "g/cm3",
    "soil_nitrogen_pct": "%",
    "salinity_ec_ds_m": "dS/m",
    "slope_pct": "%",
    "annual_rainfall_mm": "mm/yr",
    "mean_annual_temp_c": "C",
    "dry_season_months": "months",
    "water_table_depth_m": "m",
    "area_ha": "ha",
    "tree_cover_pct": "% canopy",
    "native_vegetation_pct": "% of landscape",
    "mean_patch_size_ha": "ha",
    "hedgerow_density_m_per_ha": "m/ha",
    "distance_to_natural_habitat_m": "m",
    "species_richness_observed": "species",
    "bird_species_count": "species",
    "earthworm_count_per_m2": "per m2",
    "fertilizer_n_kg_ha_yr": "kg N/ha/yr",
    "pesticide_applications_per_season": "per season",
    "grazing_intensity_lsu_ha": "LSU/ha",
    "deforestation_last_5yr_pct": "% in 5 yr",
}


def _bar(score: float, width: int = 10) -> str:
    filled = max(0, min(width, round(score * width)))
    return "#" * filled + "." * (width - filled)


def _format_value(key: str, value: object) -> str:
    # model_dump() keeps Enum members in python mode, and their repr
    # ("ClimateZone.ARID") is not what a reader should see.
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    unit = _UNITS.get(key)
    return f"{value} {unit}" if unit else str(value)


def render_site_summary(profile: SiteProfile) -> str:
    known = profile.known()
    if not known:
        return "_No site data captured yet._"
    lines: list[str] = []
    for key, value in known.items():
        if key in ("provenance", "notes"):
            continue
        if key == "geo":
            parts = [f"{k}={v}" for k, v in value.items() if v is not None]
            if parts:
                lines.append(f"- **location**: {', '.join(parts)}")
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(v) for v in value)
        else:
            rendered = _format_value(key, value)
        source = profile.provenance.get(key)
        tag = ""
        if source == "inferred":
            tag = "  _(inferred)_"
        elif source == "regional_default":
            tag = "  _(regional default)_"
        lines.append(f"- **{metric_label(key)}**: {rendered}{tag}")
    return "\n".join(lines)


def render_clarify(
    assessment: SiteAssessment, questions: list[ClarifyingQuestion]
) -> str:
    out: list[str] = ["## I need a few more numbers before I can advise responsibly", ""]
    known = assessment.profile.known()
    if known:
        out.append("**What I have so far**")
        out.append("")
        out.append(render_site_summary(assessment.profile))
        out.append("")

    if assessment.diagnoses:
        out.append("**What this already suggests**")
        out.append("")
        for diagnosis in assessment.diagnoses[:2]:
            out.append(f"- {diagnosis.label} - {diagnosis.explanation.strip()}")
        out.append("")

    out.append(
        f"Site data is {assessment.data_completeness * 100:.0f}% complete against the "
        f"diagnostic field set. The questions below are ranked by how much each answer "
        f"would change the recommendation - I simulated the plausible answers and measured "
        f"how far the advice moved."
    )
    out.append("")
    for i, question in enumerate(questions, 1):
        out.append(f"**{i}. {question.question}**")
        out.append(f"   - _Units/options_: {question.unit_or_options}")
        out.append(f"   - _Why it changes the answer_: {question.why_it_matters}")
        out.append(f"   - _Expected information gain_: {question.expected_information_gain:.2f}")
        out.append("")
    out.append(
        "A partial answer is fine - even one of these materially sharpens the assessment, "
        "and I will say which conclusions remain provisional."
    )
    return "\n".join(out)


def render_assessment(assessment: SiteAssessment, include_trace: bool = False) -> str:
    out: list[str] = []

    # --- header / state ---
    out.append("## Site assessment")
    out.append("")
    out.append(render_site_summary(assessment.profile))
    out.append("")

    if assessment.derived_indices:
        out.append("**Composite condition indices** (0 = critical, 1 = good)")
        out.append("")
        for name, score in sorted(assessment.derived_indices.items(), key=lambda kv: kv[1]):
            label = _INDEX_TEXT.get(name, name.replace("_", " ").capitalize())
            out.append(f"- `{_bar(score)}` **{score:.2f}** {label}")
        out.append("")

    out.append(
        f"_Data completeness: {assessment.data_completeness * 100:.0f}% of the diagnostic "
        f"field set._"
        + (
            "  **This assessment is provisional** - the confidence level on every "
            "recommendation below is capped accordingly."
            if assessment.data_completeness < 0.35
            else ""
        )
    )
    out.append("")

    # --- diagnosis ---
    out.append("## Diagnosis")
    out.append("")
    if not assessment.diagnoses:
        out.append("No diagnostic threshold was crossed by the data provided.")
    for diagnosis in assessment.diagnoses[:6]:
        out.append(f"### {diagnosis.label}")
        out.append(f"_Severity {diagnosis.severity:.2f}_"
                   + (f" | _limiting factor: {metric_label(diagnosis.limiting_factor)}_"
                      if diagnosis.limiting_factor else ""))
        out.append("")
        out.append(diagnosis.explanation.strip())
        if diagnosis.interacting_metrics:
            out.append("")
            out.append(
                "**Interacts with**: "
                + ", ".join(metric_label(m) for m in diagnosis.interacting_metrics)
            )
        out.append("")

    # --- recommendations ---
    out.append("## Recommendations")
    out.append("")
    if not assessment.recommendations:
        out.append(
            "No recommendation can be supported by site-applicable evidence from the "
            "knowledge base on the data provided."
        )
    for i, rec in enumerate(assessment.recommendations, 1):
        novelty = " | **non-obvious**" if rec.novelty == "non_obvious" else ""
        out.append(f"### {i}. {rec.title}{novelty}")
        out.append("")
        out.append(f"**What to do** -- {rec.what_to_do.strip()}")
        out.append("")
        out.append(f"**Why it works** -- {rec.why_it_works.strip()}")
        out.append("")
        if rec.causal_pathway:
            out.append(f"**Causal pathway** -- `{' -> '.join(rec.causal_pathway)}`")
            out.append("")

        if rec.impacted_metrics:
            out.append("**Metrics affected**")
            out.append("")
            out.append("| Metric | Direction | Expected change | Time to effect | Basis |")
            out.append("| --- | --- | --- | --- | --- |")
            for impact in rec.impacted_metrics:
                low, high = impact.time_to_effect_years
                basis = "secondary (causal model)" if impact.is_secondary else "primary (cited study)"
                out.append(
                    f"| {metric_label(impact.metric)} | {impact.direction.value} "
                    f"| {impact.magnitude} | {low:g}-{high:g} yr | {basis} |"
                )
            out.append("")

        out.append(
            f"**Time horizon** -- {_HORIZON_TEXT[rec.time_horizon.value]}  |  "
            f"**Confidence** -- {_CONFIDENCE_TEXT[rec.confidence.label]} "
            f"({rec.confidence.score:.2f})  |  "
            f"**Cost** -- {rec.cost_intensity}  |  "
            f"**Reversibility** -- {rec.reversibility}"
        )
        out.append("")
        if rec.confidence.drivers:
            out.append("_Confidence supported by_: " + "; ".join(rec.confidence.drivers))
        if rec.confidence.limiters:
            out.append("")
            out.append("_Confidence limited by_: " + "; ".join(rec.confidence.limiters))
        out.append("")

        out.append("**Evidence**")
        out.append("")
        for item in rec.evidence:
            marker = f" ({item['evidence_type'].replace('_', ' ')})"
            out.append(f"- {item['citation']}{marker}")
            out.append(f"  - Finding: {item['claim']}")
            if item["matched_site_conditions"]:
                out.append(
                    "  - Applicability confirmed for this site: "
                    + "; ".join(item["matched_site_conditions"])
                )
            if item["unverified_conditions"]:
                out.append(
                    "  - Applicability not yet verifiable: "
                    + "; ".join(item["unverified_conditions"])
                )
        out.append("")

        if rec.preconditions:
            out.append("**Preconditions to confirm** -- " + "; ".join(rec.preconditions))
            out.append("")
        if rec.risks_and_tradeoffs:
            out.append("**Risks and trade-offs**")
            out.append("")
            for risk in rec.risks_and_tradeoffs:
                out.append(f"- {risk}")
            out.append("")
        if rec.synergies:
            out.append("**Reinforces** -- " + "; ".join(rec.synergies))
            out.append("")
        if rec.conflicts:
            out.append("**Conflicts with** -- " + "; ".join(rec.conflicts))
            out.append("")
        if rec.monitoring:
            out.append("**How to verify it is working**")
            out.append("")
            for item in rec.monitoring:
                out.append(f"- {item}")
            out.append("")
        if include_trace and rec.rationale_trace:
            out.append("<details><summary>Scoring trace</summary>")
            out.append("")
            for line in rec.rationale_trace:
                out.append(f"- {line}")
            out.append("")
            out.append("</details>")
            out.append("")

    # --- set-level interactions ---
    if assessment.interaction_notes:
        out.append("## Interactions, sequencing and exclusions")
        out.append("")
        for note in assessment.interaction_notes:
            out.append(f"- {note}")
        out.append("")

    # --- outstanding questions ---
    if assessment.clarifying_questions:
        out.append("## To sharpen this further")
        out.append("")
        for question in assessment.clarifying_questions:
            out.append(
                f"- **{question.question}** ({question.unit_or_options}) -- "
                f"{question.why_it_matters}"
            )
        out.append("")

    return "\n".join(out)
