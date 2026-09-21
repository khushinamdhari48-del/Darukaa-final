"""Propagation over the signed, weighted, lagged causal graph.

This module is the reason the system can answer "and what else changes?" rather
than only "what does the paper say about X". Evidence cards report primary
effects; the graph derives the secondary ones, accumulates the lag along the
path, and emits the chain of mechanisms as the stated causal pathway.

Three operations:

  forward_effects  - an intervention raises metric M; what else moves, by how
                     much, and after how long?
  upstream_causes  - metric T is degraded; which observed upstream metrics can
                     account for it, ranked by explanatory weight?
  pathway_text     - render a path as "cover crop -> +SOC -> +aggregate
                     stability -> +plant-available water -> +soil fauna".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..knowledge.loader import CausalEdge, KnowledgeBase
from ..schemas import SiteProfile

# Effects weaker than this after attenuation are not worth reporting.
MIN_REPORTABLE_STRENGTH = 0.12
MAX_DEPTH = 3


@dataclass
class PropagatedEffect:
    metric: str
    sign: str                     # positive | negative | context_dependent
    strength: float               # product of edge strengths along the path
    lag_years: float              # sum of edge lags along the path
    path: list[CausalEdge] = field(default_factory=list)
    depth: int = 1

    @property
    def is_secondary(self) -> bool:
        return self.depth > 1

    @property
    def horizon(self) -> str:
        if self.lag_years <= 2:
            return "short"
        if self.lag_years <= 5:
            return "medium"
        return "long"

    def mechanism_chain(self) -> list[str]:
        return [e.mechanism for e in self.path]

    def evidence_ids(self) -> list[str]:
        seen: list[str] = []
        for edge in self.path:
            for cid in edge.evidence:
                if cid not in seen:
                    seen.append(cid)
        return seen


@dataclass
class UpstreamCause:
    metric: str
    sign: str
    strength: float
    lag_years: float
    observed_score: float | None
    explanatory_weight: float
    path: list[CausalEdge] = field(default_factory=list)
    mechanism: str = ""


def _resolve_sign(edge: CausalEdge, current_value: float | None) -> str:
    """Non-monotonic edges only have a determinate sign once we know where the
    site sits relative to the optimum. Without a value, say so rather than
    guessing - this is exactly the case that drives a clarifying question."""
    if edge.sign != "non_monotonic":
        return edge.sign
    if current_value is None or edge.optimum is None:
        return "context_dependent"
    return "positive" if current_value < edge.optimum else "negative"


def _combine_sign(a: str, b: str) -> str:
    if "context_dependent" in (a, b):
        return "context_dependent"
    return "positive" if a == b else "negative"


def _attenuate(edge: CausalEdge, profile: SiteProfile | None) -> float:
    """Reduce an edge's effective strength where the site is already near
    saturation or below an activation threshold.

    Without this, the engine would promise the same soil-carbon gain to a 4% SOC
    peat soil as to a 0.3% depleted one, which is the single most common way
    naive recommendation systems become wrong.
    """
    strength = edge.strength
    if profile is None:
        return strength

    # Both `saturating_at` and `threshold` are properties of the SOURCE metric's
    # current level: they describe where on its dose-response curve the site
    # already sits, and therefore how much further movement is still available.
    source_value = profile.get_field(edge.source)
    try:
        source_numeric = float(source_value) if source_value is not None else None
    except (TypeError, ValueError):
        source_numeric = None
    if source_numeric is None:
        return strength

    if edge.saturating_at is not None and source_numeric >= edge.saturating_at:
        # Past the saturation point, additional movement in the source buys much
        # less in the target - adding habitat to an already habitat-rich
        # landscape, or carbon to an already carbon-rich soil.
        strength *= 0.35
    if edge.threshold is not None and source_numeric < edge.threshold:
        # Below the activation threshold the edge is largely inert: nitrogen only
        # suppresses richness once it has stopped being the limiting nutrient.
        strength *= 0.25
    return strength


def forward_effects(
    kb: KnowledgeBase,
    seed_metric: str,
    seed_sign: str = "positive",
    profile: SiteProfile | None = None,
    max_depth: int = MAX_DEPTH,
) -> list[PropagatedEffect]:
    """Breadth-first propagation from a metric the intervention moves directly.

    Cycles are common in this graph (soil carbon <-> moisture <-> richness), so
    each metric is kept once, at its strongest path, and revisits are pruned.
    """
    best: dict[str, PropagatedEffect] = {}
    frontier: list[PropagatedEffect] = [
        PropagatedEffect(metric=seed_metric, sign=seed_sign, strength=1.0, lag_years=0.0, depth=0)
    ]

    while frontier:
        current = frontier.pop(0)
        if current.depth >= max_depth:
            continue
        for edge in kb.edges_from.get(current.metric, []):
            source_value = profile.get_field(edge.source) if profile else None
            try:
                source_numeric = float(source_value) if source_value is not None else None
            except (TypeError, ValueError):
                source_numeric = None
            edge_sign = _resolve_sign(edge, source_numeric)
            strength = current.strength * _attenuate(edge, profile)
            if strength < MIN_REPORTABLE_STRENGTH:
                continue
            if edge.target == seed_metric:
                continue  # do not report the seed as its own downstream effect
            effect = PropagatedEffect(
                metric=edge.target,
                sign=_combine_sign(current.sign, edge_sign),
                strength=round(strength, 4),
                lag_years=current.lag_years + edge.lag_years,
                path=[*current.path, edge],
                depth=current.depth + 1,
            )
            existing = best.get(edge.target)
            if existing is None or effect.strength > existing.strength:
                best[edge.target] = effect
                frontier.append(effect)

    return sorted(best.values(), key=lambda e: e.strength, reverse=True)


def upstream_causes(
    kb: KnowledgeBase,
    target_metric: str,
    profile: SiteProfile,
    scores: dict[str, float],
    max_depth: int = 2,
) -> list[UpstreamCause]:
    """Backward search for what could explain a degraded target metric.

    `explanatory_weight` combines how strongly the upstream metric influences the
    target with how degraded that upstream metric actually is. An upstream metric
    with a strong edge but a healthy value explains nothing; one that is both
    influential and degraded is the limiting factor candidate.
    """
    results: dict[str, UpstreamCause] = {}
    frontier: list[tuple[str, float, float, list[CausalEdge]]] = [(target_metric, 1.0, 0.0, [])]
    visited: set[str] = {target_metric}

    depth = 0
    while frontier and depth < max_depth:
        next_frontier: list[tuple[str, float, float, list[CausalEdge]]] = []
        for metric, strength, lag, path in frontier:
            for edge in kb.edges_to.get(metric, []):
                if edge.source in visited:
                    continue
                source_value = profile.get_field(edge.source)
                try:
                    source_numeric = float(source_value) if source_value is not None else None
                except (TypeError, ValueError):
                    source_numeric = None
                edge_sign = _resolve_sign(edge, source_numeric)
                new_strength = strength * edge.strength
                if new_strength < MIN_REPORTABLE_STRENGTH:
                    continue
                observed = scores.get(edge.source)
                # A degraded upstream metric (low score) explains a degraded
                # target. Unobserved metrics get a neutral 0.5 so they remain
                # candidates worth asking about without being asserted.
                degradation = (1.0 - observed) if observed is not None else 0.5
                weight = new_strength * degradation
                new_path = [*path, edge]
                cause = UpstreamCause(
                    metric=edge.source,
                    sign=edge_sign,
                    strength=round(new_strength, 4),
                    lag_years=lag + edge.lag_years,
                    observed_score=observed,
                    explanatory_weight=round(weight, 4),
                    path=new_path,
                    mechanism=edge.mechanism,
                )
                existing = results.get(edge.source)
                if existing is None or cause.explanatory_weight > existing.explanatory_weight:
                    results[edge.source] = cause
                next_frontier.append((edge.source, new_strength, lag + edge.lag_years, new_path))
                visited.add(edge.source)
        frontier = next_frontier
        depth += 1

    return sorted(results.values(), key=lambda c: c.explanatory_weight, reverse=True)


def pathway_text(effect: PropagatedEffect, seed_label: str) -> list[str]:
    """Render a propagation path as an arrow chain of signed metric changes."""
    steps = [seed_label]
    for edge in effect.path:
        arrow = {"positive": "+", "negative": "-", "context_dependent": "?"}[
            _sign_of_edge_display(edge)
        ]
        steps.append(f"{arrow}{metric_label(edge.target)}")
    return steps


def _sign_of_edge_display(edge: CausalEdge) -> str:
    if edge.sign == "non_monotonic":
        return "context_dependent"
    return edge.sign


def _pretty(metric: str) -> str:
    return metric.replace("_pct", " %").replace("_", " ")


def metric_label(metric: str) -> str:
    """Human-readable metric name used across all output surfaces."""
    return _LABELS.get(metric, _pretty(metric))


_LABELS: dict[str, str] = {
    "soil_organic_carbon_pct": "soil organic carbon",
    "soil_ph": "soil pH",
    "soil_moisture_pct": "soil moisture",
    "soil_bulk_density_g_cm3": "bulk density (compaction)",
    "soil_nitrogen_pct": "soil total nitrogen",
    "salinity_ec_ds_m": "soil salinity (EC)",
    "erosion_class": "erosion severity",
    "annual_rainfall_mm": "annual rainfall",
    "soil_texture": "soil texture",
    "slope_pct": "slope",
    "water_table_depth_m": "water table depth",
    "tree_cover_pct": "tree cover",
    "native_vegetation_pct": "semi-natural vegetation cover",
    "mean_patch_size_ha": "mean field/patch size",
    "hedgerow_density_m_per_ha": "hedgerow density",
    "distance_to_natural_habitat_m": "isolation from natural habitat",
    "permanent_water_present": "permanent water availability",
    "species_richness_observed": "species richness",
    "bird_species_count": "bird species richness",
    "pollinator_activity": "pollinator activity",
    "earthworm_count_per_m2": "earthworm abundance",
    "invasive_species_present": "invasive species pressure",
    "fertilizer_n_kg_ha_yr": "nitrogen fertiliser loading",
    "pesticide_applications_per_season": "pesticide applications",
    "tillage": "tillage intensity",
    "grazing_intensity_lsu_ha": "grazing intensity",
    "residue_removal": "crop residue removal",
    "deforestation_last_5yr_pct": "recent deforestation",
    "irrigation": "irrigation method",
    "cropping_system": "cropping system",
    "mean_annual_temp_c": "mean annual temperature",
    "area_ha": "area",
    "climate_zone": "climate zone",
    "rainfall_pattern": "rainfall pattern",
    "dry_season_months": "dry season length",
    "land_use": "land use",
    "primary_crop": "primary crop",
    "keystone_or_threatened_species": "keystone / threatened species",
    "nearby_pollution_source": "nearby pollution source",
    "objectives": "stated objectives",
    "constraints": "stated constraints",
}
