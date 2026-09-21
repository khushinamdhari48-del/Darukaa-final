"""Diagnosis before prescription.

The engine never jumps from a symptom to an intervention. It first produces a
ranked set of `Diagnosis` objects, each of which names a limiting factor and the
interaction that makes it limiting. Interventions are then matched against
diagnoses, which is what prevents the single-variable answers the brief rules out.

Two mechanisms do the work:

  1. Rule set - explicit, auditable conditions over normalised metrics. Each rule
     states which metrics it implicates and, via the causal graph, which other
     metrics interact with them.
  2. Limiting-factor analysis - for each degraded outcome metric, the causal graph
     is traced upstream and the most influential *and* most degraded upstream
     metric is named as the limiting factor (Liebig's law of the minimum).

Diagnoses carry a severity in [0,1] that propagates into recommendation priority.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..knowledge.loader import KnowledgeBase
from ..schemas import Diagnosis, SiteProfile
from .causal import metric_label, upstream_causes
from .metrics import DerivedIndex, MetricAssessment

# Outcome metrics whose degradation triggers limiting-factor analysis.
OUTCOME_METRICS = (
    "species_richness_observed",
    "pollinator_activity",
    "soil_organic_carbon_pct",
    "soil_moisture_pct",
)


@dataclass
class DiagnosticContext:
    kb: KnowledgeBase
    profile: SiteProfile
    assessments: dict[str, MetricAssessment]
    indices: dict[str, DerivedIndex]

    def value(self, metric: str):
        return self.profile.get_field(metric)

    def score(self, metric: str) -> float | None:
        assessment = self.assessments.get(metric)
        return assessment.score if assessment else None

    def scores(self) -> dict[str, float]:
        return {m: a.score for m, a in self.assessments.items()}

    def verdict(self, metric: str) -> str | None:
        assessment = self.assessments.get(metric)
        return assessment.verdict if assessment else None

    def index(self, name: str) -> float | None:
        idx = self.indices.get(name)
        if idx is None or idx.coverage == 0:
            return None
        return idx.score

    def is_bad(self, metric: str) -> bool:
        return self.verdict(metric) in ("critical", "poor")

    def interacting(self, metric: str, limit: int = 3) -> list[str]:
        """Metrics linked to `metric` in either direction in the causal graph,
        restricted to those the engine has actually observed."""
        related: list[str] = []
        for edge in self.kb.edges_from.get(metric, []):
            related.append(edge.target)
        for edge in self.kb.edges_to.get(metric, []):
            related.append(edge.source)
        seen: list[str] = []
        for m in related:
            if m != metric and m not in seen and m in self.assessments:
                seen.append(m)
        return seen[:limit]


Rule = Callable[[DiagnosticContext], Diagnosis | None]
_RULES: list[Rule] = []


def rule(func: Rule) -> Rule:
    _RULES.append(func)
    return func


def _severity_from_score(score: float | None, floor: float = 0.3) -> float:
    if score is None:
        return floor
    return round(min(1.0, max(floor, 1.0 - score)), 3)


# ---------------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------------


@rule
def soc_depleted(ctx: DiagnosticContext) -> Diagnosis | None:
    soc = ctx.value("soil_organic_carbon_pct")
    if soc is None or soc >= 1.5:
        return None
    interacting = [
        m
        for m in ("soil_moisture_pct", "earthworm_count_per_m2", "soil_nitrogen_pct", "soil_bulk_density_g_cm3")
        if m in ctx.assessments
    ]
    threshold_note = (
        "below the ~1% threshold at which aggregate stability, water-holding capacity and "
        "soil faunal habitat decline together rather than proportionally"
        if soc < 1.0
        else "low enough that substantial sequestration headroom remains"
    )
    return Diagnosis(
        code="soc_depleted",
        label=f"Soil organic carbon depleted ({soc:.2f}%)",
        severity=_severity_from_score(ctx.score("soil_organic_carbon_pct")),
        limiting_factor="soil_organic_carbon_pct",
        explanation=(
            f"Soil organic carbon of {soc:.2f}% is {threshold_note}. Carbon is the hub metric "
            f"on this site: it simultaneously supplies the energy base for soil biota, binds the "
            f"aggregates that hold plant-available water, and buffers nutrient supply. Restoring it "
            f"therefore moves several metrics at once, and the marginal return per unit carbon is "
            f"highest precisely because the starting stock is low."
            + (
                f" Co-observed: {', '.join(metric_label(m) for m in interacting)}."
                if interacting
                else ""
            )
        ),
        metrics_implicated=["soil_organic_carbon_pct"],
        interacting_metrics=interacting or ctx.interacting("soil_organic_carbon_pct"),
    )


@rule
def acidity_constraint(ctx: DiagnosticContext) -> Diagnosis | None:
    ph = ctx.value("soil_ph")
    if ph is None or ph >= 5.5:
        return None
    return Diagnosis(
        code="acidity_constraint",
        label=f"Soil acidity constraint (pH {ph:.1f})",
        severity=_severity_from_score(ctx.score("soil_ph"), floor=0.45),
        limiting_factor="soil_ph",
        explanation=(
            f"At pH {ph:.1f}, aluminium solubility truncates rooting depth and phosphorus is "
            f"fixed, while rhizobial nodulation is impaired. This makes pH a gating constraint: "
            f"legume-based carbon and nitrogen interventions cannot deliver their stated effect "
            f"until it is relieved, so sequencing matters more than selection here."
        ),
        metrics_implicated=["soil_ph", "soil_nitrogen_pct"],
        interacting_metrics=ctx.interacting("soil_ph"),
    )


@rule
def alkalinity_constraint(ctx: DiagnosticContext) -> Diagnosis | None:
    ph = ctx.value("soil_ph")
    if ph is None or ph < 8.2:
        return None
    return Diagnosis(
        code="alkalinity_constraint",
        label=f"Strongly alkaline soil (pH {ph:.1f})",
        severity=_severity_from_score(ctx.score("soil_ph"), floor=0.4),
        limiting_factor="soil_ph",
        explanation=(
            f"At pH {ph:.1f}, phosphorus precipitates and iron, zinc and manganese become "
            f"unavailable. Strong alkalinity frequently co-occurs with sodicity, so the "
            f"structural and chemical constraints should be assessed together rather than "
            f"treating this as a nutrient problem alone."
        ),
        metrics_implicated=["soil_ph", "salinity_ec_ds_m"],
        interacting_metrics=ctx.interacting("soil_ph"),
    )


@rule
def salinity_constraint(ctx: DiagnosticContext) -> Diagnosis | None:
    ec = ctx.value("salinity_ec_ds_m")
    if ec is None or ec < 4:
        return None
    water_table = ctx.value("water_table_depth_m")
    drainage_note = ""
    if water_table is not None and water_table < 2:
        drainage_note = (
            f" The water table at {water_table:.1f} m is the binding constraint: amendment "
            f"without drainage will reverse through capillary rise, so drainage must precede it."
        )
    return Diagnosis(
        code="salinity_constraint",
        label=f"Salt-affected soil (EC {ec:.1f} dS/m)",
        severity=_severity_from_score(ctx.score("salinity_ec_ds_m"), floor=0.5),
        limiting_factor="salinity_ec_ds_m",
        explanation=(
            f"Electrical conductivity of {ec:.1f} dS/m imposes osmotic stress that excludes all "
            f"but salt-tolerant species, and where sodicity accompanies it, clay dispersion "
            f"destroys the structure that recruitment depends on.{drainage_note}"
        ),
        metrics_implicated=["salinity_ec_ds_m", "species_richness_observed"],
        interacting_metrics=ctx.interacting("salinity_ec_ds_m"),
    )


@rule
def compaction(ctx: DiagnosticContext) -> Diagnosis | None:
    bd = ctx.value("soil_bulk_density_g_cm3")
    if bd is None or bd < 1.45:
        return None
    grazing = ctx.value("grazing_intensity_lsu_ha")
    cause = ""
    if grazing is not None and grazing > 1.2:
        cause = f" Grazing at {grazing:.1f} LSU/ha is a plausible proximate cause."
    elif ctx.value("tillage") in ("conventional", "deep"):
        cause = " Machinery traffic under conventional tillage is a plausible proximate cause."
    return Diagnosis(
        code="compaction",
        label=f"Soil compaction (bulk density {bd:.2f} g/cm3)",
        severity=_severity_from_score(ctx.score("soil_bulk_density_g_cm3"), floor=0.4),
        limiting_factor="soil_bulk_density_g_cm3",
        explanation=(
            f"Bulk density of {bd:.2f} g/cm3 is approaching or past root-restricting. Compaction "
            f"collapses the macropores that conduct water, so the site loses infiltration and "
            f"gains runoff at unchanged rainfall, and burrowing fauna are excluded. Relieving "
            f"compaction is a precondition for amendment or planting to respond.{cause}"
        ),
        metrics_implicated=["soil_bulk_density_g_cm3", "soil_moisture_pct", "earthworm_count_per_m2"],
        interacting_metrics=ctx.interacting("soil_bulk_density_g_cm3"),
    )


@rule
def soil_biology_collapsed(ctx: DiagnosticContext) -> Diagnosis | None:
    worms = ctx.value("earthworm_count_per_m2")
    tillage = ctx.value("tillage")
    soc = ctx.value("soil_organic_carbon_pct")
    triggered = (worms is not None and worms < 50) or (
        tillage in ("conventional", "deep") and soc is not None and soc < 1.0
    )
    if not triggered:
        return None
    detail = f"Earthworm density of {worms:.0f}/m2 " if worms is not None else "Soil faunal habitat "
    return Diagnosis(
        code="soil_biology_collapsed",
        label="Soil biological function suppressed",
        severity=_severity_from_score(ctx.score("earthworm_count_per_m2"), floor=0.4),
        limiting_factor="earthworm_count_per_m2" if worms is not None else "tillage",
        explanation=(
            f"{detail}indicates suppressed soil biological function. This is a compound outcome, "
            f"not a single deficiency: annual disturbance removes the fauna, low carbon removes "
            f"their food base, and the loss of their burrows then reduces infiltration, which "
            f"further reduces the biological activity window. The loop has to be broken at the "
            f"disturbance and carbon-input ends simultaneously."
        ),
        metrics_implicated=["earthworm_count_per_m2", "soil_organic_carbon_pct", "tillage"],
        interacting_metrics=ctx.interacting("earthworm_count_per_m2"),
    )


@rule
def erosion_risk(ctx: DiagnosticContext) -> Diagnosis | None:
    erosion = ctx.value("erosion_class")
    slope = ctx.value("slope_pct")
    residue = ctx.value("residue_removal")
    triggered = (
        erosion in ("moderate", "severe")
        or (slope is not None and slope >= 8 and residue in ("full", "burned", "partial"))
        or (slope is not None and slope >= 15)
    )
    if not triggered:
        return None
    parts = []
    if erosion:
        parts.append(f"erosion class '{erosion}'")
    if slope is not None:
        parts.append(f"slope {slope:.0f}%")
    if residue:
        parts.append(f"residue removal '{residue}'")
    return Diagnosis(
        code="erosion_risk",
        label="Active or imminent soil erosion",
        severity=max(
            _severity_from_score(ctx.score("erosion_class"), floor=0.4),
            0.5 if (slope is not None and slope >= 15) else 0.4,
        ),
        limiting_factor="erosion_class",
        explanation=(
            f"Erosion indicators ({', '.join(parts)}) point to ongoing loss of the surface layer. "
            f"Erosion is self-reinforcing here because it preferentially removes the "
            f"carbon-enriched fine fraction, which weakens aggregates, which increases crusting "
            f"and runoff, which accelerates the next erosion event. Any carbon-building "
            f"intervention will leak until the hydrological pathway is interrupted."
        ),
        metrics_implicated=["erosion_class", "soil_organic_carbon_pct", "slope_pct"],
        interacting_metrics=ctx.interacting("erosion_class"),
    )


@rule
def bare_soil_exposure(ctx: DiagnosticContext) -> Diagnosis | None:
    residue = ctx.value("residue_removal")
    if residue not in ("full", "burned"):
        return None
    return Diagnosis(
        code="bare_soil_exposure",
        label=f"Soil surface left bare (residue: {residue})",
        severity=0.6 if residue == "burned" else 0.5,
        limiting_factor="residue_removal",
        explanation=(
            f"Residue is {residue}, so the dominant aboveground carbon input is removed while "
            f"decomposition of existing stocks continues, and the surface loses its barrier to "
            f"direct evaporation and raindrop-impact crusting. Three metrics move adversely from "
            f"one practice: carbon input, stored water, and infiltration capacity."
        ),
        metrics_implicated=["residue_removal", "soil_organic_carbon_pct", "soil_moisture_pct"],
        interacting_metrics=ctx.interacting("residue_removal"),
    )


@rule
def crusted_surface(ctx: DiagnosticContext) -> Diagnosis | None:
    rainfall = ctx.value("annual_rainfall_mm")
    soc = ctx.value("soil_organic_carbon_pct")
    if rainfall is None or rainfall > 800 or soc is None or soc >= 0.8:
        return None
    return Diagnosis(
        code="crusted_surface",
        label="Surface sealing likely under low-carbon dryland conditions",
        severity=0.55,
        limiting_factor="soil_organic_carbon_pct",
        explanation=(
            f"With {soc:.2f}% organic carbon at {rainfall:.0f} mm rainfall, aggregates are too "
            f"weak to survive raindrop impact, so a surface seal forms in the first storms of the "
            f"season. The consequence is that rainfall total overstates available water: the "
            f"limiting variable is the infiltrating fraction, not the amount that falls."
        ),
        metrics_implicated=["soil_organic_carbon_pct", "soil_moisture_pct", "annual_rainfall_mm"],
        interacting_metrics=ctx.interacting("soil_organic_carbon_pct"),
    )


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------


@rule
def water_partitioning_loss(ctx: DiagnosticContext) -> Diagnosis | None:
    rainfall = ctx.value("annual_rainfall_mm")
    zone = ctx.value("climate_zone")
    zone_name = zone.value if hasattr(zone, "value") else zone
    dryland = (rainfall is not None and rainfall < 900) or zone_name in (
        "arid",
        "semi_arid",
        "dry_subhumid",
    )
    if not dryland:
        return None
    irrigation = ctx.value("irrigation")
    if irrigation in ("drip", "sprinkler"):
        return None
    moisture = ctx.value("soil_moisture_pct")
    moisture_note = (
        f" Measured soil moisture of {moisture:.0f}% confirms the deficit is realised, not merely predicted."
        if moisture is not None and moisture < 15
        else ""
    )
    return Diagnosis(
        code="water_partitioning_loss",
        label="Rainfall is poorly partitioned into plant-available water",
        severity=0.65 if (rainfall is not None and rainfall < 600) else 0.5,
        limiting_factor="soil_moisture_pct",
        explanation=(
            f"In water-limited systems the binding constraint is usually not how much rain falls "
            f"but what fraction of it becomes transpiration rather than runoff, direct evaporation "
            f"or deep drainage. At {rainfall:.0f} mm, dry-spell exposure within the season governs "
            f"yield and faunal persistence more than the annual total does, which is why "
            f"infiltration and evaporation control outrank drought-tolerant variety choice "
            f"here.{moisture_note}"
        ),
        metrics_implicated=["soil_moisture_pct", "annual_rainfall_mm", "soil_organic_carbon_pct"],
        interacting_metrics=ctx.interacting("soil_moisture_pct"),
    )


@rule
def water_scarcity_for_fauna(ctx: DiagnosticContext) -> Diagnosis | None:
    if ctx.value("permanent_water_present") is not False:
        return None
    zone = ctx.value("climate_zone")
    zone_name = zone.value if hasattr(zone, "value") else zone
    if zone_name not in ("arid", "semi_arid", "dry_subhumid"):
        return None
    return Diagnosis(
        code="water_scarcity_for_fauna",
        label="No free water available to terrestrial fauna",
        severity=0.45,
        limiting_factor="permanent_water_present",
        explanation=(
            "With no permanent or seasonal surface water in a dryland setting, water availability "
            "acts on the fauna directly rather than only through productivity: it constrains "
            "which consumers can persist through the dry season regardless of how much vegetation "
            "is present. Vegetation cover can therefore look stable while the characteristic "
            "fauna is being lost."
        ),
        metrics_implicated=["permanent_water_present", "species_richness_observed"],
        interacting_metrics=ctx.interacting("permanent_water_present"),
    )


@rule
def waterlogging(ctx: DiagnosticContext) -> Diagnosis | None:
    moisture = ctx.value("soil_moisture_pct")
    water_table = ctx.value("water_table_depth_m")
    irrigation = ctx.value("irrigation")
    triggered = (moisture is not None and moisture > 60) or (
        water_table is not None and water_table < 1.5 and irrigation in ("flood", "furrow")
    )
    if not triggered:
        return None
    return Diagnosis(
        code="waterlogging",
        label="Waterlogging / shallow water table",
        severity=0.5,
        limiting_factor="water_table_depth_m",
        explanation=(
            "Saturated conditions suppress root and aerobic faunal function and, under irrigation, "
            "drive secondary salinisation by capillary rise. Drainage capacity is the variable to "
            "resolve before any amendment or planting programme."
        ),
        metrics_implicated=["water_table_depth_m", "soil_moisture_pct", "salinity_ec_ds_m"],
        interacting_metrics=ctx.interacting("water_table_depth_m"),
    )


# ---------------------------------------------------------------------------
# Land use and habitat structure
# ---------------------------------------------------------------------------


@rule
def monoculture_risk(ctx: DiagnosticContext) -> Diagnosis | None:
    system = ctx.value("cropping_system")
    system_name = system.value if hasattr(system, "value") else system
    if system_name != "monoculture":
        return None
    crop = ctx.value("primary_crop") or "the single crop"
    return Diagnosis(
        code="monoculture_risk",
        label=f"Monoculture cropping system ({crop})",
        severity=0.6,
        limiting_factor="cropping_system",
        explanation=(
            f"A monoculture of {crop} provides one rooting depth, one phenology and one "
            f"disturbance calendar. Floral and root-exudate resources are therefore available in "
            f"one narrow window, which caps the populations of pollinators, natural enemies and "
            f"soil biota that can persist between seasons, and removes the response diversity "
            f"that buffers production against climate extremes."
        ),
        metrics_implicated=["cropping_system", "species_richness_observed", "soil_organic_carbon_pct"],
        interacting_metrics=ctx.interacting("cropping_system"),
    )


@rule
def structural_simplification(ctx: DiagnosticContext) -> Diagnosis | None:
    tree = ctx.value("tree_cover_pct")
    native = ctx.value("native_vegetation_pct")
    hedge = ctx.value("hedgerow_density_m_per_ha")
    signals = [
        v for v in (
            tree if tree is not None and tree < 10 else None,
            native if native is not None and native < 10 else None,
            hedge if hedge is not None and hedge < 20 else None,
        ) if v is not None
    ]
    if not signals:
        return None
    parts = []
    if tree is not None:
        parts.append(f"tree cover {tree:.0f}%")
    if native is not None:
        parts.append(f"semi-natural cover {native:.0f}%")
    if hedge is not None:
        parts.append(f"hedgerow density {hedge:.0f} m/ha")
    return Diagnosis(
        code="structural_simplification",
        label="Vertical and perennial structure largely absent",
        severity=0.6,
        limiting_factor="native_vegetation_pct" if native is not None else "tree_cover_pct",
        explanation=(
            f"Structural indicators ({', '.join(parts)}) show a system with no perennial vertical "
            f"layer. This removes nesting substrate, overwintering refuge and cool microclimate "
            f"simultaneously, so the guilds that need any of the three are excluded outright "
            f"rather than merely reduced. Because the response to added semi-natural cover is "
            f"steepest at low cover, the marginal biodiversity return per hectare added is at its "
            f"maximum on a site in this state."
        ),
        metrics_implicated=["tree_cover_pct", "native_vegetation_pct", "species_richness_observed"],
        interacting_metrics=ctx.interacting("native_vegetation_pct"),
    )


@rule
def habitat_isolation(ctx: DiagnosticContext) -> Diagnosis | None:
    distance = ctx.value("distance_to_natural_habitat_m")
    if distance is None or distance < 500:
        return None
    severity = 0.75 if distance >= 1500 else 0.55
    return Diagnosis(
        code="habitat_isolation",
        label=f"Isolated from natural habitat ({distance:.0f} m)",
        severity=severity,
        limiting_factor="distance_to_natural_habitat_m",
        explanation=(
            f"At {distance:.0f} m from the nearest semi-natural habitat, the site sits beyond the "
            f"foraging radius of most wild pollinators and outside the recolonisation range of "
            f"many ground-dwelling taxa. This changes the correct order of action: in-field floral "
            f"provision alone cannot compensate for isolation, because the species that would use "
            f"it are not present to be attracted. Connectivity and on-site nesting habitat have "
            f"to come first."
        ),
        metrics_implicated=["distance_to_natural_habitat_m", "pollinator_activity", "species_richness_observed"],
        interacting_metrics=ctx.interacting("distance_to_natural_habitat_m"),
    )


@rule
def fragmentation(ctx: DiagnosticContext) -> Diagnosis | None:
    patch = ctx.value("mean_patch_size_ha")
    native = ctx.value("native_vegetation_pct")
    if patch is None or patch < 5:
        return None
    note = ""
    if native is not None and native < 20:
        note = (
            f" With semi-natural cover at {native:.0f}%, the landscape is below the habitat-amount "
            f"band where configuration stops mattering, so spatial arrangement carries independent "
            f"weight here."
        )
    return Diagnosis(
        code="fragmentation",
        label=f"Coarse-grained landscape (mean parcel {patch:.0f} ha)",
        severity=0.5 if patch < 20 else 0.65,
        limiting_factor="mean_patch_size_ha",
        explanation=(
            f"Mean parcel size of {patch:.0f} ha means field interiors lie beyond the influence of "
            f"margins, so any margin-based measure serves only a fraction of the area. Reducing "
            f"grain size raises edge density and lets mobile organisms reach complementary "
            f"resources within a daily foraging range, which delivers diversity gains without "
            f"requiring additional habitat area.{note}"
        ),
        metrics_implicated=["mean_patch_size_ha", "species_richness_observed"],
        interacting_metrics=ctx.interacting("mean_patch_size_ha"),
    )


@rule
def degraded_land(ctx: DiagnosticContext) -> Diagnosis | None:
    land_use = ctx.value("land_use")
    land_use_name = land_use.value if hasattr(land_use, "value") else land_use
    soil_health = ctx.index("soil_health")
    triggered = land_use_name == "degraded_barren" or (
        soil_health is not None and soil_health < 0.3
    )
    if not triggered:
        return None
    return Diagnosis(
        code="degraded_land",
        label="Land in a degraded state across multiple indicators",
        severity=0.7,
        limiting_factor="soil_organic_carbon_pct",
        explanation=(
            "Degradation indicators co-occur here rather than appearing singly, which is the "
            "characteristic pattern: carbon loss weakens aggregates, weak aggregates increase "
            "crusting and runoff, runoff removes more carbon-rich topsoil. Single-threat "
            "interventions get reversed by the remaining threats, so the sequencing of measures "
            "matters as much as their selection."
        ),
        metrics_implicated=["soil_organic_carbon_pct", "erosion_class", "native_vegetation_pct"],
        interacting_metrics=ctx.interacting("soil_organic_carbon_pct"),
    )


@rule
def invasive_dominance(ctx: DiagnosticContext) -> Diagnosis | None:
    if ctx.value("invasive_species_present") is not True:
        return None
    return Diagnosis(
        code="invasive_dominance",
        label="Invasive species present",
        severity=0.6,
        limiting_factor="invasive_species_present",
        explanation=(
            "Invasive dominance can arrest succession outright: dense exotic swards intercept "
            "light and moisture at the seedling layer and often raise fine-fuel loads and fire "
            "frequency, creating a self-reinforcing alternative state. Control is therefore a "
            "precondition for restoration measures on this site, not an alternative to them."
        ),
        metrics_implicated=["invasive_species_present", "species_richness_observed", "native_vegetation_pct"],
        interacting_metrics=ctx.interacting("invasive_species_present"),
    )


# ---------------------------------------------------------------------------
# Biodiversity function
# ---------------------------------------------------------------------------


@rule
def pollinator_deficit(ctx: DiagnosticContext) -> Diagnosis | None:
    activity = ctx.value("pollinator_activity")
    distance = ctx.value("distance_to_natural_habitat_m")
    pesticides = ctx.value("pesticide_applications_per_season")
    triggered = activity in ("absent", "low") or (
        distance is not None and distance > 500 and (pesticides or 0) >= 2
    )
    if not triggered:
        return None
    drivers = []
    if distance is not None and distance > 300:
        drivers.append(f"isolation from habitat ({distance:.0f} m)")
    if pesticides:
        drivers.append(f"{pesticides} pesticide applications per season")
    if ctx.value("native_vegetation_pct") is not None and ctx.value("native_vegetation_pct") < 10:
        drivers.append(f"semi-natural cover of {ctx.value('native_vegetation_pct'):.0f}%")
    return Diagnosis(
        code="pollinator_deficit",
        label="Pollinator function deficient",
        severity=0.6,
        limiting_factor="distance_to_natural_habitat_m" if distance else "pollinator_activity",
        explanation=(
            "Pollinator shortfall on this site is consistent with multiple co-acting drivers"
            + (f" ({'; '.join(drivers)})" if drivers else "")
            + ". These do not substitute for each other: floral provision addresses the forage "
            "limitation, nesting substrate addresses the reproduction limitation, and pesticide "
            "reduction addresses the mortality limitation. Fixing one while the others bind "
            "produces little measurable change, which is the usual reason flower-strip "
            "interventions disappoint."
        ),
        metrics_implicated=["pollinator_activity", "distance_to_natural_habitat_m", "pesticide_applications_per_season"],
        interacting_metrics=ctx.interacting("pollinator_activity"),
    )


@rule
def natural_enemy_deficit(ctx: DiagnosticContext) -> Diagnosis | None:
    pesticides = ctx.value("pesticide_applications_per_season")
    native = ctx.value("native_vegetation_pct")
    if pesticides is None or pesticides < 2:
        return None
    if native is not None and native >= 20:
        return None
    return Diagnosis(
        code="natural_enemy_deficit",
        label="Biological pest control potential degraded",
        severity=0.55,
        limiting_factor="pesticide_applications_per_season",
        explanation=(
            f"With {pesticides} applications per season and little semi-natural habitat to supply "
            f"overwintering refuge and alternative prey, natural enemy populations cannot persist "
            f"between crop cycles. Because enemies sit at a higher trophic level and recover more "
            f"slowly than their prey, each application degrades the control that would have "
            f"reduced the need for the next one, so the dependency is self-reinforcing and cannot "
            f"be exited by simply spraying less."
        ),
        metrics_implicated=["pesticide_applications_per_season", "species_richness_observed", "native_vegetation_pct"],
        interacting_metrics=ctx.interacting("pesticide_applications_per_season"),
    )


@rule
def pesticide_dependency(ctx: DiagnosticContext) -> Diagnosis | None:
    pesticides = ctx.value("pesticide_applications_per_season")
    if pesticides is None or pesticides < 3:
        return None
    return Diagnosis(
        code="pesticide_dependency",
        label=f"High pesticide intensity ({pesticides} applications/season)",
        severity=_severity_from_score(ctx.score("pesticide_applications_per_season"), floor=0.5),
        limiting_factor="pesticide_applications_per_season",
        explanation=(
            f"At {pesticides} applications per season, insecticide and fungicide pressure is the "
            f"single most consistently negative component of intensification for wild plants, "
            f"ground beetles and farmland birds. Exiting the dependency has to be staged: habitat "
            f"for natural enemies first, then thresholds and scouting, then reduced frequency. "
            f"Cutting applications before the enemy community exists invites an outbreak."
        ),
        metrics_implicated=["pesticide_applications_per_season", "pollinator_activity", "species_richness_observed"],
        interacting_metrics=ctx.interacting("pesticide_applications_per_season"),
    )


@rule
def nutrient_enrichment(ctx: DiagnosticContext) -> Diagnosis | None:
    nitrogen = ctx.value("fertilizer_n_kg_ha_yr")
    if nitrogen is None or nitrogen < 150:
        return None
    return Diagnosis(
        code="nutrient_enrichment",
        label=f"Nitrogen loading above crop uptake capacity ({nitrogen:.0f} kg N/ha/yr)",
        severity=_severity_from_score(ctx.score("fertilizer_n_kg_ha_yr"), floor=0.45),
        limiting_factor="fertilizer_n_kg_ha_yr",
        explanation=(
            f"At {nitrogen:.0f} kg N/ha/yr, a substantial fraction is lost to water and air, and "
            f"in receiving margins and water bodies enrichment shifts competition from nutrient "
            f"capture to light capture, so fast-growing nitrophiles displace low-stature "
            f"specialists. The intervention is synchronisation of supply with demand rather than "
            f"a blind rate cut, and it also slows the soil acidification that nitrification drives."
        ),
        metrics_implicated=["fertilizer_n_kg_ha_yr", "species_richness_observed", "soil_ph"],
        interacting_metrics=ctx.interacting("fertilizer_n_kg_ha_yr"),
    )


@rule
def nitrogen_limited(ctx: DiagnosticContext) -> Diagnosis | None:
    nitrogen = ctx.value("fertilizer_n_kg_ha_yr")
    soil_n = ctx.value("soil_nitrogen_pct")
    soc = ctx.value("soil_organic_carbon_pct")
    triggered = (
        (nitrogen is not None and nitrogen < 30)
        or (soil_n is not None and soil_n < 0.08)
        or (soc is not None and soc < 0.6 and nitrogen is None)
    )
    if not triggered:
        return None
    return Diagnosis(
        code="nitrogen_limited",
        label="Nitrogen supply limiting",
        severity=0.5,
        limiting_factor="soil_nitrogen_pct",
        explanation=(
            "Indicators point to nitrogen limitation rather than nitrogen excess. This inverts the "
            "usual biodiversity advice: here the appropriate move is to raise nitrogen supply "
            "biologically, through legumes and nitrogen-fixing woody species, which adds carbon "
            "and habitat structure at the same time, rather than to reduce inputs. Note also that "
            "carbon accumulation is stoichiometrically constrained by nitrogen availability, so "
            "soil-carbon targets cannot be met without resolving this first."
        ),
        metrics_implicated=["soil_nitrogen_pct", "soil_organic_carbon_pct", "fertilizer_n_kg_ha_yr"],
        interacting_metrics=ctx.interacting("soil_nitrogen_pct"),
    )


@rule
def overgrazing(ctx: DiagnosticContext) -> Diagnosis | None:
    intensity = ctx.value("grazing_intensity_lsu_ha")
    if intensity is None or intensity < 1.2:
        return None
    zone = ctx.value("climate_zone")
    zone_name = zone.value if hasattr(zone, "value") else zone
    arid_note = (
        " In an arid or semi-arid system the same stocking rate does more damage, because regrowth "
        "is water-limited and the recovery interval the sward needs is longer than the calendar "
        "rotation usually allows."
        if zone_name in ("arid", "semi_arid")
        else ""
    )
    return Diagnosis(
        code="overgrazing",
        label=f"Grazing pressure above sustainable range ({intensity:.1f} LSU/ha)",
        severity=_severity_from_score(ctx.score("grazing_intensity_lsu_ha"), floor=0.5),
        limiting_factor="grazing_intensity_lsu_ha",
        explanation=(
            f"At {intensity:.1f} LSU/ha, defoliation outpaces regrowth, so root carbon input falls "
            f"and hoof pressure compacts the surface, reducing infiltration. Note that the "
            f"operative variable is recovery time relative to regrowth rate and grazing "
            f"distribution, not the mean stocking rate alone, and that complete exclusion is not "
            f"the answer in systems with a long grazing history.{arid_note}"
        ),
        metrics_implicated=["grazing_intensity_lsu_ha", "soil_organic_carbon_pct", "soil_bulk_density_g_cm3"],
        interacting_metrics=ctx.interacting("grazing_intensity_lsu_ha"),
    )


@rule
def pollution_pathway(ctx: DiagnosticContext) -> Diagnosis | None:
    nitrogen = ctx.value("fertilizer_n_kg_ha_yr")
    pesticides = ctx.value("pesticide_applications_per_season")
    source = ctx.value("nearby_pollution_source")
    triggered = bool(source) or ((nitrogen or 0) >= 120 and (pesticides or 0) >= 2)
    if not triggered:
        return None
    detail = f" Reported nearby source: {source}." if source else ""
    return Diagnosis(
        code="pollution_pathway",
        label="Active pollution transport pathway to receiving habitat",
        severity=0.5,
        limiting_factor="fertilizer_n_kg_ha_yr",
        explanation=(
            "Nutrient and pesticide loads combined with an uninterrupted runoff pathway transfer "
            "the pressure off-site, where regulatory thresholds derived from single-species acute "
            "tests under-protect sensitive invertebrate families. Interrupting the pathway is "
            f"cheaper and faster than changing the loading, and does not require input reductions "
            f"to deliver its first effect.{detail}"
        ),
        metrics_implicated=["fertilizer_n_kg_ha_yr", "pesticide_applications_per_season", "species_richness_observed"],
        interacting_metrics=ctx.interacting("fertilizer_n_kg_ha_yr"),
    )


@rule
def deforestation_active(ctx: DiagnosticContext) -> Diagnosis | None:
    loss = ctx.value("deforestation_last_5yr_pct")
    if loss is None or loss < 1:
        return None
    return Diagnosis(
        code="deforestation_active",
        label=f"Recent forest loss ({loss:.0f}% in 5 years)",
        severity=min(1.0, 0.5 + loss / 40),
        limiting_factor="deforestation_last_5yr_pct",
        explanation=(
            f"Clearing of {loss:.0f}% in five years means the effective habitat loss exceeds the "
            f"cleared area, because edge creation alters microclimate hundreds of metres into the "
            f"remaining habitat and raises tree mortality there. Halting conversion outranks every "
            f"on-site enhancement measure: enhancement cannot outpace ongoing removal, and the "
            f"extinction debt already incurred will continue to be paid regardless."
        ),
        metrics_implicated=["deforestation_last_5yr_pct", "native_vegetation_pct", "species_richness_observed"],
        interacting_metrics=ctx.interacting("deforestation_last_5yr_pct"),
    )


@rule
def richness_below_baseline(ctx: DiagnosticContext) -> Diagnosis | None:
    observed = ctx.value("species_richness_observed")
    if observed is None:
        return None
    land_use = ctx.value("land_use")
    zone = ctx.value("climate_zone")
    baseline = ctx.kb.richness_baseline(
        land_use.value if hasattr(land_use, "value") else land_use,
        zone.value if hasattr(zone, "value") else zone,
    )
    if baseline is None or observed >= baseline.expected_min:
        return None
    shortfall = (baseline.expected_min - observed) / baseline.expected_min
    return Diagnosis(
        code="richness_below_baseline",
        label=f"Species richness below the regional baseline ({observed:g} vs {baseline.expected_min:g}-{baseline.expected_max:g})",
        severity=round(min(1.0, 0.4 + shortfall), 3),
        limiting_factor=None,
        explanation=(
            f"Observed richness of {observed:g} sits below the {baseline.expected_min:g}-"
            f"{baseline.expected_max:g} {baseline.unit} expected for {baseline.land_use} in this "
            f"zone ({baseline.source}). A shortfall against a land-use-matched baseline, rather "
            f"than against pristine reference, is the defensible signal that something on this "
            f"site is suppressing richness beyond what the land use itself implies."
        ),
        metrics_implicated=["species_richness_observed"],
        interacting_metrics=ctx.interacting("species_richness_observed"),
    )


# ---------------------------------------------------------------------------
# Limiting-factor analysis over the causal graph
# ---------------------------------------------------------------------------


def limiting_factor_diagnoses(ctx: DiagnosticContext) -> list[Diagnosis]:
    """For each degraded outcome metric, name the upstream limiting factor.

    This complements the rule set: rules encode known syndromes, while this
    traverses the graph and can surface a limiting factor no rule anticipated.
    """
    out: list[Diagnosis] = []
    scores = ctx.scores()
    for target in OUTCOME_METRICS:
        assessment = ctx.assessments.get(target)
        if assessment is None or assessment.score > 0.45:
            continue
        causes = upstream_causes(ctx.kb, target, ctx.profile, scores)
        observed = [c for c in causes if c.observed_score is not None][:2]
        if not observed:
            continue
        primary = observed[0]
        others = ", ".join(metric_label(c.metric) for c in observed[1:])
        out.append(
            Diagnosis(
                code=f"limiting::{target}::{primary.metric}",
                label=f"{metric_label(primary.metric).capitalize()} is limiting {metric_label(target)}",
                severity=round(min(1.0, primary.explanatory_weight * 1.6), 3),
                limiting_factor=primary.metric,
                explanation=(
                    f"Tracing {metric_label(target)} upstream through the causal graph, "
                    f"{metric_label(primary.metric)} carries the highest explanatory weight "
                    f"({primary.explanatory_weight:.2f}) because it is both influential "
                    f"(edge strength {primary.strength:.2f}) and itself degraded "
                    f"(condition score {primary.observed_score:.2f}). Mechanism: "
                    f"{primary.mechanism.strip()}"
                    + (f" Secondary contributor: {others}." if others else "")
                    + f" Expected lag before a change here shows up in {metric_label(target)}: "
                    f"about {primary.lag_years:.0f} year(s)."
                ),
                metrics_implicated=[target, primary.metric],
                interacting_metrics=[c.metric for c in observed[1:]] or ctx.interacting(target),
            )
        )
    return out


def diagnose(
    kb: KnowledgeBase,
    profile: SiteProfile,
    assessments: dict[str, MetricAssessment],
    indices: dict[str, DerivedIndex],
) -> list[Diagnosis]:
    """Run the full rule set plus limiting-factor analysis, ranked by severity."""
    ctx = DiagnosticContext(kb=kb, profile=profile, assessments=assessments, indices=indices)
    found: list[Diagnosis] = []
    for rule_fn in _RULES:
        try:
            result = rule_fn(ctx)
        except (TypeError, ValueError, AttributeError):
            # A malformed value must not take down the whole assessment.
            continue
        if result is not None:
            found.append(result)
    found.extend(limiting_factor_diagnoses(ctx))
    found.sort(key=lambda d: d.severity, reverse=True)
    return found
