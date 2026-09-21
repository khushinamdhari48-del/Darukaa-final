"""Candidate generation, multi-metric scoring and evidence attachment.

Pipeline for one assessment:

  diagnoses
     -> candidate interventions       (an intervention must address a fired diagnosis)
     -> hard contraindication filter  (site-violating candidates are dropped, with reason)
     -> evidence retrieval            (site-conditioned, per candidate)
     -> effect composition            (card effects = primary; causal graph = secondary)
     -> confidence calculus           (evidence strength x site match x data completeness)
     -> priority scoring              (severity x leverage x breadth x evidence - cost)
     -> synergy / conflict resolution across the selected set

Every recommendation carries a `rationale_trace`, which is the audit log of why
it scored where it did. Nothing in the output is asserted without a traceable
reason, and nothing is asserted at all if the only supporting evidence has a
predicate the site contradicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..knowledge.loader import Intervention, KnowledgeBase
from ..knowledge.retriever import EvidenceIndex
from ..schemas import (
    EVIDENCE_WEIGHT,
    Confidence,
    Diagnosis,
    Direction,
    MetricImpact,
    Recommendation,
    RetrievedEvidence,
    SiteProfile,
    TimeHorizon,
)
from .causal import forward_effects, metric_label, pathway_text
from .metrics import DerivedIndex, MetricAssessment

# Scoring weights. Kept explicit and in one place so the ranking is auditable
# and tunable rather than buried in the arithmetic.
W_DIAGNOSIS = 3.0     # how well it targets what is actually wrong
W_LEVERAGE = 2.0      # how degraded the metrics it moves currently are
W_BREADTH = 1.6       # multi-metric reach (the brief's core differentiator)
W_EVIDENCE = 1.5      # strength of the supporting literature
W_NOVELTY = 0.7       # reward non-obvious measures over generic ones
W_COST = 0.8          # penalty
W_SLOW = 0.4          # penalty for very long time-to-effect

COST_PENALTY = {"low": 0.0, "moderate": 0.4, "high": 1.0}
HORIZON_PENALTY = {"short": 0.0, "medium": 0.3, "long": 0.8}


@dataclass
class Candidate:
    intervention: Intervention
    matched: list[Diagnosis] = field(default_factory=list)
    evidence: list[RetrievedEvidence] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    unknown_preconditions: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


@dataclass
class RejectedCandidate:
    intervention_id: str
    title: str
    reasons: list[str]


def _check_conditions(
    intervention: Intervention, profile: SiteProfile
) -> tuple[list[str], list[str], list[str]]:
    """Evaluate `requires` and `contraindications` against the site.

    Returns (blocking, unknown_preconditions, satisfied).
    An unknown requirement does not block; it becomes a stated precondition and
    feeds the clarifying-question ranking, which is how the system distinguishes
    "does not apply" from "we do not yet know whether it applies".
    """
    blocking: list[str] = []
    unknown: list[str] = []
    satisfied: list[str] = []

    for condition in intervention.requires:
        verdict = condition.evaluate(profile.get_field(condition.field))
        if verdict is False:
            reason = f"requires {condition.describe()}"
            if condition.note:
                reason += f" - {condition.note}"
            blocking.append(reason)
        elif verdict is None:
            unknown.append(f"requires {condition.describe()} (not yet known for this site)")
        else:
            satisfied.append(condition.describe())

    for condition in intervention.contraindications:
        verdict = condition.evaluate(profile.get_field(condition.field))
        if verdict is True:
            reason = f"contraindicated where {condition.describe()}"
            if condition.note:
                reason += f" - {condition.note}"
            blocking.append(reason)

    return blocking, unknown, satisfied


def generate_candidates(
    kb: KnowledgeBase, profile: SiteProfile, diagnoses: list[Diagnosis]
) -> tuple[list[Candidate], list[RejectedCandidate]]:
    """An intervention becomes a candidate only if it addresses a fired diagnosis."""
    # Limiting-factor diagnoses are synthesised, so they match on the metric they
    # implicate rather than on a code in the catalogue.
    limiting_metrics = {d.limiting_factor for d in diagnoses if d.code.startswith("limiting::")}

    by_code: dict[str, list[Diagnosis]] = {}
    for d in diagnoses:
        by_code.setdefault(d.code, []).append(d)

    candidates: list[Candidate] = []
    rejected: list[RejectedCandidate] = []

    for intervention in kb.interventions:
        matched: list[Diagnosis] = []
        for code in intervention.addresses:
            matched.extend(by_code.get(code, []))
        # Also match an intervention that directly targets a metric the graph
        # identified as limiting.
        for d in diagnoses:
            if d.code.startswith("limiting::") and d.limiting_factor in intervention.targets:
                matched.append(d)
        if not matched:
            continue

        blocking, unknown, satisfied = _check_conditions(intervention, profile)
        if blocking:
            rejected.append(
                RejectedCandidate(
                    intervention_id=intervention.id,
                    title=intervention.title,
                    reasons=blocking,
                )
            )
            continue

        trace = [
            f"Matched {len(matched)} diagnosis/es: "
            + ", ".join(sorted({d.code for d in matched}))
        ]
        if satisfied:
            trace.append("Site satisfies: " + "; ".join(satisfied))
        if limiting_metrics & set(intervention.targets):
            trace.append(
                "Directly targets a graph-identified limiting factor: "
                + ", ".join(sorted(limiting_metrics & set(intervention.targets)))
            )
        candidates.append(
            Candidate(
                intervention=intervention,
                matched=matched,
                unknown_preconditions=unknown,
                trace=trace,
            )
        )

    return candidates, rejected


def attach_evidence(
    index: EvidenceIndex, candidate: Candidate, profile: SiteProfile, query_hint: str
) -> None:
    """Retrieve site-conditioned evidence for one candidate.

    The query is built from the intervention plus the diagnoses it addresses, so
    retrieval is steered by the reasoning rather than only by the user's wording.
    """
    diagnosis_terms = " ".join(
        sorted({d.label for d in candidate.matched})
    )
    query = f"{candidate.intervention.title}. {candidate.intervention.what_to_do} {diagnosis_terms} {query_hint}"
    results, _ = index.retrieve(
        query=query,
        profile=profile,
        top_k=4,
        intervention_filter={candidate.intervention.id},
    )
    candidate.evidence = results
    if results:
        candidate.trace.append(
            "Evidence retrieved: "
            + ", ".join(f"{r.card.id} (score {r.score:.2f})" for r in results)
        )


def _evidence_strength(candidate: Candidate) -> float:
    """Best-of, with a modest corroboration bonus for independent sources."""
    if not candidate.evidence:
        return 0.0
    weights = [
        EVIDENCE_WEIGHT.get(r.card.citation.type.value, 0.5) * max(r.condition_score, 0.2)
        for r in candidate.evidence
    ]
    best = max(weights)
    distinct_sources = len({r.card.citation.formatted() for r in candidate.evidence})
    corroboration = min(0.15, 0.05 * (distinct_sources - 1))
    return min(1.0, best + corroboration)


def _compose_impacts(
    kb: KnowledgeBase, candidate: Candidate, profile: SiteProfile
) -> tuple[list[MetricImpact], list[list[str]]]:
    """Primary impacts from evidence-card effects; secondary from the causal graph."""
    impacts: list[MetricImpact] = []
    pathways: list[list[str]] = []
    seen: set[str] = set()

    # --- primary: quantified effects from the cited cards, restricted to the
    # metrics this intervention actually targets ---
    for retrieved in candidate.evidence:
        for effect in retrieved.card.effects:
            if effect.metric in seen:
                continue
            if effect.metric not in candidate.intervention.targets and len(impacts) >= 3:
                continue
            magnitude = effect.absolute_change or (
                f"{effect.relative_change_pct[0]:g}-{effect.relative_change_pct[1]:g}% relative change"
                if effect.relative_change_pct
                else "direction established; magnitude site-dependent"
            )
            if effect.measurement_context:
                magnitude += f" ({effect.measurement_context})"
            impacts.append(
                MetricImpact(
                    metric=effect.metric,
                    direction=effect.direction,
                    magnitude=f"{magnitude} [{retrieved.card.citation.short()}]",
                    time_to_effect_years=effect.time_to_effect_years,
                    pathway=None,
                    is_secondary=False,
                )
            )
            seen.add(effect.metric)

    # --- secondary: propagate each targeted metric through the causal graph ---
    for target in candidate.intervention.targets:
        seed_sign = "negative" if target in _REDUCTION_TARGETS else "positive"
        for effect in forward_effects(kb, target, seed_sign=seed_sign, profile=profile):
            if effect.metric in seen or effect.strength < 0.2:
                continue
            if effect.sign == "context_dependent":
                continue
            chain = pathway_text(effect, seed_label=metric_label(target))
            impacts.append(
                MetricImpact(
                    metric=effect.metric,
                    direction=Direction.INCREASE if effect.sign == "positive" else Direction.DECREASE,
                    magnitude=(
                        f"secondary effect, propagation strength {effect.strength:.2f} "
                        f"via {len(effect.path)} causal step(s)"
                    ),
                    time_to_effect_years=(
                        round(max(1.0, effect.lag_years), 1),
                        round(max(2.0, effect.lag_years * 2), 1),
                    ),
                    pathway=" -> ".join(chain),
                    is_secondary=True,
                )
            )
            pathways.append(chain)
            seen.add(effect.metric)

    return impacts, pathways


# Metrics where the intervention's intent is to reduce, not increase, the value.
_REDUCTION_TARGETS = {
    "fertilizer_n_kg_ha_yr",
    "pesticide_applications_per_season",
    "salinity_ec_ds_m",
    "erosion_class",
    "soil_bulk_density_g_cm3",
    "mean_patch_size_ha",
    "distance_to_natural_habitat_m",
    "deforestation_last_5yr_pct",
    "water_table_depth_m",
}


def _leverage(
    candidate: Candidate, assessments: dict[str, MetricAssessment]
) -> tuple[float, list[str]]:
    """How degraded are the metrics this intervention moves?

    A measure that improves an already-good metric scores low leverage even when
    its literature support is excellent. This is what stops the engine from
    recommending cover crops to a 4% SOC soil.
    """
    observed = [
        assessments[m].score for m in candidate.intervention.targets if m in assessments
    ]
    if not observed:
        return 0.5, ["no targeted metric measured; leverage assumed neutral"]
    deficit = 1.0 - (sum(observed) / len(observed))
    notes = [
        f"{metric_label(m)} currently {assessments[m].verdict} (score {assessments[m].score:.2f})"
        for m in candidate.intervention.targets
        if m in assessments
    ]
    return round(deficit, 3), notes


def _confidence(
    candidate: Candidate,
    evidence_strength: float,
    completeness: float,
    impacts: list[MetricImpact],
) -> Confidence:
    drivers: list[str] = []
    limiters: list[str] = []

    if candidate.evidence:
        best = max(candidate.evidence, key=lambda r: r.score)
        drivers.append(
            f"strongest supporting source is a {best.card.citation.type.value.replace('_', ' ')} "
            f"({best.card.citation.short()})"
        )
        matched = [c for r in candidate.evidence for c in r.satisfied_conditions]
        if matched:
            drivers.append(
                "site conditions match the evidence's stated applicability: "
                + "; ".join(sorted(set(matched))[:3])
            )
        unknown = [c for r in candidate.evidence for c in r.unknown_conditions]
        if unknown:
            limiters.append(
                "applicability conditions not yet verifiable for this site: "
                + "; ".join(sorted(set(unknown))[:3])
            )
    else:
        limiters.append("no quantified evidence card matched this intervention")

    if len({r.card.citation.formatted() for r in candidate.evidence}) > 1:
        drivers.append("corroborated by more than one independent source")

    if completeness < 0.35:
        limiters.append(
            f"site data only {completeness * 100:.0f}% complete on the diagnostic field set"
        )
    else:
        drivers.append(f"site data {completeness * 100:.0f}% complete")

    if candidate.unknown_preconditions:
        limiters.append(
            f"{len(candidate.unknown_preconditions)} unverified precondition(s)"
        )

    secondary = sum(1 for i in impacts if i.is_secondary)
    if secondary:
        limiters.append(
            f"{secondary} of {len(impacts)} impacts are model-propagated secondary effects, "
            "not directly measured for this intervention"
        )

    # Confidence is deliberately capped by data completeness: strong literature
    # on a barely-described site does not justify a confident recommendation.
    completeness_cap = 0.45 + 0.5 * min(completeness / 0.6, 1.0)
    raw = evidence_strength * (0.6 + 0.4 * min(completeness / 0.5, 1.0))
    if candidate.unknown_preconditions:
        raw *= 0.9
    score = round(min(raw, completeness_cap), 3)

    if score >= 0.8:
        label = "very_high"
    elif score >= 0.62:
        label = "high"
    elif score >= 0.42:
        label = "moderate"
    else:
        label = "low"

    return Confidence(score=score, label=label, drivers=drivers, limiters=limiters)


_HORIZON_ORDER = {TimeHorizon.SHORT: 0, TimeHorizon.MEDIUM: 1, TimeHorizon.LONG: 2}


def _horizon(candidate: Candidate, impacts: list[MetricImpact]) -> TimeHorizon:
    """Time to the first *verifiable* effect.

    An effect cannot be measured before the intervention is established, so the
    literature's time-to-effect is floored by the establishment period. The
    catalogue's own horizon is then taken as a lower bound, so a slow-maturing
    measure is never advertised as short-term just because one cited study
    reported an early response on an already-established system.
    """
    establishment = candidate.intervention.establishment_years[0]
    if impacts:
        earliest = max(min(i.time_to_effect_years[0] for i in impacts), establishment)
    else:
        earliest = candidate.intervention.establishment_years[1]

    if earliest <= 2:
        derived = TimeHorizon.SHORT
    elif earliest <= 5:
        derived = TimeHorizon.MEDIUM
    else:
        derived = TimeHorizon.LONG

    catalogue = TimeHorizon(candidate.intervention.time_horizon)
    return derived if _HORIZON_ORDER[derived] >= _HORIZON_ORDER[catalogue] else catalogue


def build_recommendations(
    kb: KnowledgeBase,
    index: EvidenceIndex,
    profile: SiteProfile,
    diagnoses: list[Diagnosis],
    assessments: dict[str, MetricAssessment],
    indices: dict[str, DerivedIndex],
    completeness: float,
    query_hint: str = "",
    limit: int = 4,
) -> tuple[list[Recommendation], list[RejectedCandidate], list[str]]:
    candidates, rejected = generate_candidates(kb, profile, diagnoses)
    for candidate in candidates:
        attach_evidence(index, candidate, profile, query_hint)

    # Drop candidates whose only evidence is contradicted by the site, rather
    # than recommending something we cannot legitimately cite.
    unsupported = [c for c in candidates if not c.evidence]
    for candidate in unsupported:
        rejected.append(
            RejectedCandidate(
                intervention_id=candidate.intervention.id,
                title=candidate.intervention.title,
                reasons=["no site-applicable evidence card survived condition filtering"],
            )
        )
    candidates = [c for c in candidates if c.evidence]

    scored: list[tuple[float, Candidate, list[MetricImpact], list[list[str]], Confidence]] = []
    for candidate in candidates:
        impacts, pathways = _compose_impacts(kb, candidate, profile)
        evidence_strength = _evidence_strength(candidate)
        leverage, leverage_notes = _leverage(candidate, assessments)
        severity = max((d.severity for d in candidate.matched), default=0.3)
        distinct_codes = len({d.code for d in candidate.matched})
        breadth = min(1.0, len({i.metric for i in impacts}) / 5.0)
        novelty_bonus = 1.0 if candidate.intervention.novelty == "non_obvious" else 0.0

        score = (
            W_DIAGNOSIS * severity * min(1.0, 0.6 + 0.2 * distinct_codes)
            + W_LEVERAGE * leverage
            + W_BREADTH * breadth
            + W_EVIDENCE * evidence_strength
            + W_NOVELTY * novelty_bonus
            - W_COST * COST_PENALTY[candidate.intervention.cost_intensity]
            - W_SLOW * HORIZON_PENALTY[candidate.intervention.time_horizon]
        )

        confidence = _confidence(candidate, evidence_strength, completeness, impacts)
        candidate.trace.extend(
            [
                f"Diagnosis severity {severity:.2f} across {distinct_codes} distinct diagnosis code(s) -> {W_DIAGNOSIS * severity:.2f}",
                f"Leverage {leverage:.2f} ({'; '.join(leverage_notes[:3])}) -> {W_LEVERAGE * leverage:.2f}",
                f"Multi-metric breadth {len({i.metric for i in impacts})} metric(s) -> {W_BREADTH * breadth:.2f}",
                f"Evidence strength {evidence_strength:.2f} -> {W_EVIDENCE * evidence_strength:.2f}",
                f"Cost penalty ({candidate.intervention.cost_intensity}) -> -{W_COST * COST_PENALTY[candidate.intervention.cost_intensity]:.2f}",
                f"Horizon penalty ({candidate.intervention.time_horizon}) -> -{W_SLOW * HORIZON_PENALTY[candidate.intervention.time_horizon]:.2f}",
                f"Priority score {score:.2f}",
            ]
        )
        scored.append((score, candidate, impacts, pathways, confidence))

    scored.sort(key=lambda t: t[0], reverse=True)
    selected = scored[:limit]
    selected_ids = {c.intervention.id for _, c, _, _, _ in selected}

    recommendations: list[Recommendation] = []
    for score, candidate, impacts, pathways, confidence in selected:
        iv = candidate.intervention
        top_evidence = candidate.evidence[0]
        mechanism = top_evidence.card.mechanism.strip()
        caveats = [c for r in candidate.evidence for c in r.card.caveats]

        causal_pathway = _best_pathway(pathways) or [metric_label(m) for m in iv.targets[:3]]

        recommendations.append(
            Recommendation(
                id=iv.id,
                title=iv.title,
                what_to_do=iv.what_to_do,
                why_it_works=mechanism,
                causal_pathway=causal_pathway,
                impacted_metrics=impacts,
                time_horizon=_horizon(candidate, impacts),
                confidence=confidence,
                evidence=[
                    {
                        "card_id": r.card.id,
                        "claim": r.card.claim.strip(),
                        "citation": r.card.citation.formatted(),
                        "citation_short": r.card.citation.short(),
                        "evidence_type": r.card.citation.type.value,
                        "doi": r.card.citation.doi,
                        "url": r.card.citation.url,
                        "retrieval_score": r.score,
                        "matched_site_conditions": r.satisfied_conditions,
                        "unverified_conditions": r.unknown_conditions,
                    }
                    for r in candidate.evidence
                ],
                preconditions=candidate.unknown_preconditions,
                risks_and_tradeoffs=list(dict.fromkeys(caveats))[:5],
                synergies=[
                    f"{kb.interventions_by_id[s].title} (also selected)"
                    if s in selected_ids
                    else kb.interventions_by_id[s].title
                    for s in iv.synergies
                    if s in kb.interventions_by_id
                ][:4],
                conflicts=[
                    f"{kb.interventions_by_id[c].title}"
                    + (" - both selected; resolve before implementing" if c in selected_ids else "")
                    for c in iv.conflicts
                    if c in kb.interventions_by_id
                ],
                monitoring=list(iv.monitoring),
                cost_intensity=iv.cost_intensity,
                reversibility=iv.reversibility,
                priority_score=round(score, 3),
                novelty=iv.novelty,
                rationale_trace=candidate.trace,
            )
        )

    notes = _interaction_notes(kb, recommendations, selected_ids)
    return recommendations, rejected, notes


# Chains that terminate in one of these are the ones worth showing: they carry
# the argument all the way from the action to a biodiversity outcome.
_OUTCOME_ENDPOINTS = {
    "species richness",
    "pollinator activity",
    "bird species richness",
    "earthworm abundance",
}


def _best_pathway(pathways: list[list[str]]) -> list[str] | None:
    """Prefer the longest chain that ends at a biodiversity outcome; fall back to
    the longest chain of any kind."""
    if not pathways:
        return None
    terminal_outcome = [
        p for p in pathways if p and p[-1].lstrip("+-?").strip() in _OUTCOME_ENDPOINTS
    ]
    pool = terminal_outcome or pathways
    return max(pool, key=len)


def _interaction_notes(
    kb: KnowledgeBase, recommendations: list[Recommendation], selected_ids: set[str]
) -> list[str]:
    """Cross-recommendation interactions: reinforcing pairs, conflicting pairs and
    sequencing constraints. These are properties of the *set*, not of any single
    recommendation, so they are reported separately."""
    notes: list[str] = []
    ids = [r.id for r in recommendations]
    # Conflicts are declared symmetrically in the catalogue, so a pair would
    # otherwise be reported once from each side.
    reported_pairs: set[frozenset[str]] = set()

    for rec in recommendations:
        iv = kb.interventions_by_id[rec.id]
        for other in iv.synergies:
            if other in selected_ids and ids.index(other) > ids.index(rec.id):
                notes.append(
                    f"Reinforcing pair: '{iv.title}' and '{kb.interventions_by_id[other].title}' "
                    f"act on overlapping mechanisms, so implementing both yields more than the "
                    f"sum of the two applied separately."
                )
        for other in iv.conflicts:
            pair = frozenset({iv.id, other})
            if other in selected_ids and pair not in reported_pairs:
                reported_pairs.add(pair)
                notes.append(
                    f"Conflict to resolve: '{iv.title}' and "
                    f"'{kb.interventions_by_id[other].title}' compete for the same land, water or "
                    f"labour on this site. Choose one, or zone them spatially."
                )

    # Sequencing: gating constraints that must precede the rest.
    gating = {
        "liming_ph_correction": "pH correction gates legume-based nitrogen and carbon measures; sequence it first.",
        "gypsum_sodic_reclamation": "Sodic reclamation (with drainage) gates all planting measures; sequence it first.",
        "invasive_species_control": "Invasive control gates regeneration measures; sequence it first or succession will be arrested.",
        "wetland_restoration": "Hydrology restoration gates vegetation work; sequence it first.",
        "mangrove_restoration": "Tidal hydrology restoration gates planting; sequence it first.",
    }
    for rec_id, message in gating.items():
        if rec_id in selected_ids and len(selected_ids) > 1:
            notes.append(f"Sequencing: {message}")

    return list(dict.fromkeys(notes))
