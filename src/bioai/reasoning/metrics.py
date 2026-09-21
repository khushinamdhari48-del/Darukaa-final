"""Metric normalisation, derived indices and data-completeness accounting.

Raw site values are converted into a common 0-1 "condition score" using the
banded reference table in `datasets/metric_bands.csv`, so that a pH of 4.9 and a
soil organic carbon of 0.3% become commensurable. Derived indices then aggregate
those scores into the five composite dimensions the diagnostics work over.

Every index reports its own coverage - the fraction of its components that were
actually observed - because an index computed from one of six components must
not be presented with the same authority as one computed from all six.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..knowledge.loader import KnowledgeBase
from ..schemas import ClimateZone, SiteProfile

# Band verdict -> condition score. Deliberately non-linear: `critical` is scored
# very low so that a single critical metric dominates the composite, which is the
# behaviour implied by Liebig's law of the minimum.
VERDICT_SCORE: dict[str, float] = {
    "critical": 0.05,
    "poor": 0.30,
    "moderate": 0.55,
    "good": 0.82,
    "high": 0.90,
}

# Ordinal categoricals have no numeric band table, so they get explicit scales.
CATEGORICAL_SCORES: dict[str, dict[str, float]] = {
    "pollinator_activity": {"absent": 0.05, "low": 0.30, "moderate": 0.60, "high": 0.90},
    "erosion_class": {"none": 0.90, "slight": 0.65, "moderate": 0.35, "severe": 0.08},
    "tillage": {"none": 0.90, "reduced": 0.70, "conventional": 0.30, "deep": 0.10},
    "residue_removal": {"none": 0.90, "partial": 0.55, "full": 0.20, "burned": 0.05},
    "irrigation": {
        "drip": 0.90,
        "sprinkler": 0.70,
        "rainfed_supplemental": 0.60,
        "furrow": 0.40,
        "flood": 0.25,
        "none": 0.45,
    },
    "cropping_system": {
        "agroforestry": 0.90,
        "intercrop": 0.75,
        "rotation": 0.60,
        "fallow": 0.50,
        "monoculture": 0.20,
        "not_applicable": 0.50,
    },
    "soil_texture": {
        "loam": 0.85,
        "silt_loam": 0.80,
        "clay_loam": 0.75,
        "sandy_loam": 0.60,
        "clay": 0.55,
        "sand": 0.35,
    },
}

# Composite index definitions: (metric, weight). Weights are relative within an
# index and renormalised over whichever components are present.
INDEX_COMPONENTS: dict[str, list[tuple[str, float]]] = {
    "soil_health": [
        ("soil_organic_carbon_pct", 3.0),
        ("soil_ph", 2.0),
        ("soil_bulk_density_g_cm3", 1.5),
        ("salinity_ec_ds_m", 1.5),
        ("erosion_class", 1.5),
        ("earthworm_count_per_m2", 1.5),
        ("soil_texture", 0.5),
    ],
    "water_security": [
        ("annual_rainfall_mm", 2.5),
        ("soil_moisture_pct", 2.5),
        ("soil_organic_carbon_pct", 1.5),
        ("soil_bulk_density_g_cm3", 1.0),
        ("residue_removal", 1.0),
        ("irrigation", 1.0),
        ("slope_pct", 0.5),
    ],
    "habitat_integrity": [
        ("native_vegetation_pct", 3.0),
        ("distance_to_natural_habitat_m", 2.5),
        ("tree_cover_pct", 2.0),
        ("mean_patch_size_ha", 1.5),
        ("cropping_system", 1.0),
    ],
    "human_pressure": [
        ("pesticide_applications_per_season", 2.5),
        ("fertilizer_n_kg_ha_yr", 2.0),
        ("deforestation_last_5yr_pct", 2.5),
        ("grazing_intensity_lsu_ha", 1.5),
        ("tillage", 1.5),
        ("residue_removal", 1.0),
    ],
    "biodiversity_state": [
        ("pollinator_activity", 2.0),
        ("earthworm_count_per_m2", 1.5),
        ("native_vegetation_pct", 1.5),
        ("tree_cover_pct", 1.0),
    ],
}

# Fields that carry diagnostic weight. Used for the completeness fraction, which
# in turn gates how assertive the engine is allowed to be.
DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "soil_organic_carbon_pct",
    "soil_ph",
    "soil_moisture_pct",
    "soil_texture",
    "soil_bulk_density_g_cm3",
    "salinity_ec_ds_m",
    "erosion_class",
    "slope_pct",
    "annual_rainfall_mm",
    "rainfall_pattern",
    "dry_season_months",
    "mean_annual_temp_c",
    "climate_zone",
    "irrigation",
    "land_use",
    "cropping_system",
    "primary_crop",
    "tree_cover_pct",
    "native_vegetation_pct",
    "mean_patch_size_ha",
    "hedgerow_density_m_per_ha",
    "distance_to_natural_habitat_m",
    "permanent_water_present",
    "species_richness_observed",
    "pollinator_activity",
    "earthworm_count_per_m2",
    "invasive_species_present",
    "fertilizer_n_kg_ha_yr",
    "pesticide_applications_per_season",
    "tillage",
    "grazing_intensity_lsu_ha",
    "residue_removal",
    "deforestation_last_5yr_pct",
)


@dataclass
class MetricAssessment:
    metric: str
    value: object
    score: float
    verdict: str
    band: str | None
    interpretation: str
    source: str | None

    @property
    def is_limiting(self) -> bool:
        return self.verdict in ("critical", "poor")


@dataclass
class DerivedIndex:
    name: str
    score: float
    coverage: float
    components_present: list[str]
    components_missing: list[str]
    weakest: str | None


def assess_metric(kb: KnowledgeBase, metric: str, value: object) -> MetricAssessment | None:
    """Normalise a single raw value against the reference bands."""
    if value is None:
        return None

    if metric in CATEGORICAL_SCORES:
        key = value.value if hasattr(value, "value") else str(value)
        score = CATEGORICAL_SCORES[metric].get(key)
        if score is None:
            return None
        verdict = _verdict_from_score(score)
        return MetricAssessment(
            metric=metric,
            value=key,
            score=score,
            verdict=verdict,
            band=key,
            interpretation=f"{metric.replace('_', ' ')} = {key}",
            source="ordinal scale (see reasoning/metrics.py CATEGORICAL_SCORES)",
        )

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    band = kb.band_for(metric, numeric)
    if band is None:
        return None
    return MetricAssessment(
        metric=metric,
        value=numeric,
        score=VERDICT_SCORE[band.verdict],
        verdict=band.verdict,
        band=band.band,
        interpretation=band.interpretation,
        source=band.source,
    )


def _verdict_from_score(score: float) -> str:
    if score < 0.18:
        return "critical"
    if score < 0.45:
        return "poor"
    if score < 0.70:
        return "moderate"
    return "good"


def assess_all(kb: KnowledgeBase, profile: SiteProfile) -> dict[str, MetricAssessment]:
    out: dict[str, MetricAssessment] = {}
    for metric in DIAGNOSTIC_FIELDS:
        assessment = assess_metric(kb, metric, profile.get_field(metric))
        if assessment is not None:
            out[metric] = assessment
    return out


def derive_indices(
    assessments: dict[str, MetricAssessment]
) -> dict[str, DerivedIndex]:
    """Weighted aggregation with a minimum-sensitive correction.

    A plain weighted mean would let a good score on five components mask a
    critical score on the sixth, which is precisely the failure mode in
    environmental assessment. The composite is therefore pulled toward the
    weakest component: `0.7 * weighted_mean + 0.3 * minimum`.
    """
    out: dict[str, DerivedIndex] = {}
    for name, components in INDEX_COMPONENTS.items():
        present = [(m, w) for m, w in components if m in assessments]
        missing = [m for m, _ in components if m not in assessments]
        if not present:
            out[name] = DerivedIndex(name, 0.0, 0.0, [], missing, None)
            continue
        total_weight = sum(w for _, w in present)
        weighted_mean = sum(assessments[m].score * w for m, w in present) / total_weight
        minimum = min(assessments[m].score for m, _ in present)
        score = 0.7 * weighted_mean + 0.3 * minimum
        weakest = min(present, key=lambda mw: assessments[mw[0]].score)[0]
        coverage = sum(w for _, w in present) / sum(w for _, w in components)
        out[name] = DerivedIndex(
            name=name,
            score=round(score, 3),
            coverage=round(coverage, 3),
            components_present=[m for m, _ in present],
            components_missing=missing,
            weakest=weakest,
        )
    return out


def data_completeness(profile: SiteProfile) -> float:
    """Fraction of diagnostic fields that carry a value."""
    filled = sum(1 for f in DIAGNOSTIC_FIELDS if profile.get_field(f) is not None)
    return round(filled / len(DIAGNOSTIC_FIELDS), 3)


def infer_context(kb: KnowledgeBase, profile: SiteProfile) -> SiteProfile:
    """Fill a small number of derivable context fields, marked as inferred.

    Only relationships that are definitional or well-established are inferred -
    climate zone from rainfall, dry-season length from zone, cropping system from
    an agroforestry land use. Nothing that the engine would otherwise ask about
    is silently invented.
    """
    updates: dict[str, object] = {}
    provenance: dict[str, str] = {}

    if profile.climate_zone is None and profile.annual_rainfall_mm is not None:
        zone = kb.zone_for_rainfall(profile.annual_rainfall_mm)
        if zone is not None:
            updates["climate_zone"] = ClimateZone(zone.climate_zone)
            provenance["climate_zone"] = "inferred"

    zone_name = updates.get("climate_zone") or profile.climate_zone
    if profile.dry_season_months is None and zone_name is not None:
        name = zone_name.value if hasattr(zone_name, "value") else str(zone_name)
        match = next((z for z in kb.zones if z.climate_zone == name), None)
        if match is not None:
            updates["dry_season_months"] = match.typical_dry_season_months
            provenance["dry_season_months"] = "regional_default"

    if profile.cropping_system is None and profile.land_use is not None:
        if profile.land_use.value == "agroforestry":
            updates["cropping_system"] = "agroforestry"
            provenance["cropping_system"] = "inferred"
        elif profile.land_use.value in ("forest", "grassland", "wetland", "mangrove", "degraded_barren"):
            updates["cropping_system"] = "not_applicable"
            provenance["cropping_system"] = "inferred"

    if not updates:
        return profile
    patch = SiteProfile(**updates, provenance=provenance)  # type: ignore[arg-type]
    return profile.merge(patch)
