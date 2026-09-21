"""Reasoning invariants.

These are the tests that matter most: they assert the engine reasons across
several variables at once, respects site context, refuses to give advice it
cannot cite, and never contradicts itself between opposite sites.
"""
from __future__ import annotations

import pytest

from bioai.knowledge.loader import load_knowledge_base
from bioai.reasoning.causal import forward_effects, upstream_causes
from bioai.reasoning.engine import ReasoningEngine
from bioai.reasoning.metrics import assess_all, data_completeness, derive_indices, infer_context
from bioai.schemas import (
    ClimateZone,
    CroppingSystem,
    LandUse,
    RainfallPattern,
    SiteProfile,
)


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="module")
def engine():
    return ReasoningEngine()


# --- the brief's worked example ------------------------------------------------

SEMI_ARID_WHEAT = SiteProfile(
    soil_organic_carbon_pct=0.3,
    annual_rainfall_mm=420,
    rainfall_pattern=RainfallPattern.ERRATIC,
    land_use=LandUse.CROPLAND,
    cropping_system=CroppingSystem.MONOCULTURE,
    primary_crop="wheat",
    soil_ph=5.2,
    tillage="conventional",
    residue_removal="burned",
    pesticide_applications_per_season=4,
    distance_to_natural_habitat_m=1500,
    native_vegetation_pct=4,
    tree_cover_pct=1,
    irrigation="none",
)

# Deliberately the mirror image: high carbon, high rainfall, over-fertilised.
TEMPERATE_INTENSIVE = SiteProfile(
    soil_organic_carbon_pct=2.4,
    soil_ph=6.8,
    annual_rainfall_mm=850,
    land_use=LandUse.CROPLAND,
    cropping_system=CroppingSystem.ROTATION,
    primary_crop="wheat",
    fertilizer_n_kg_ha_yr=220,
    pesticide_applications_per_season=5,
    mean_patch_size_ha=40,
    hedgerow_density_m_per_ha=0,
    native_vegetation_pct=3,
    pollinator_activity="low",
    tillage="reduced",
    residue_removal="none",
)

OVERGRAZED_RANGELAND = SiteProfile(
    land_use=LandUse.GRASSLAND,
    annual_rainfall_mm=480,
    grazing_intensity_lsu_ha=2.2,
    soil_organic_carbon_pct=0.9,
    soil_bulk_density_g_cm3=1.55,
    erosion_class="severe",
    permanent_water_present=False,
    invasive_species_present=True,
)

SALINE_IRRIGATED = SiteProfile(
    land_use=LandUse.CROPLAND,
    primary_crop="cotton",
    irrigation="flood",
    salinity_ec_ds_m=9.0,
    soil_ph=8.4,
    water_table_depth_m=1.2,
    annual_rainfall_mm=300,
    soil_organic_carbon_pct=0.6,
)


# --- metric normalisation -----------------------------------------------------


def test_metric_banding_and_indices(kb):
    assessments = assess_all(kb, SEMI_ARID_WHEAT)
    assert assessments["soil_organic_carbon_pct"].verdict == "critical"
    assert assessments["soil_ph"].verdict == "poor"
    indices = derive_indices(assessments)
    assert 0 <= indices["soil_health"].score <= 0.4
    # Coverage must be reported so a thin index is not over-trusted
    assert 0 < indices["soil_health"].coverage <= 1.0
    assert indices["soil_health"].weakest == "soil_organic_carbon_pct"


def test_index_is_pulled_toward_its_weakest_component(kb):
    """Liebig's law of the minimum: one critical metric must not be averaged away."""
    good = SiteProfile(soil_organic_carbon_pct=3.0, soil_ph=6.5, soil_bulk_density_g_cm3=1.2,
                       salinity_ec_ds_m=0.5, erosion_class="none", earthworm_count_per_m2=200)
    one_critical = good.model_copy(update={"salinity_ec_ds_m": 30.0})
    score_good = derive_indices(assess_all(kb, good))["soil_health"].score
    score_bad = derive_indices(assess_all(kb, one_critical))["soil_health"].score
    assert score_good > 0.75
    assert score_bad < score_good - 0.15


def test_context_inference_is_marked_not_asserted(kb):
    enriched = infer_context(kb, SiteProfile(annual_rainfall_mm=420, land_use=LandUse.CROPLAND))
    assert enriched.climate_zone == ClimateZone.SEMI_ARID
    assert enriched.provenance["climate_zone"] == "inferred"
    assert enriched.provenance["dry_season_months"] == "regional_default"


def test_inference_never_overwrites_user_values(kb):
    stated = SiteProfile(
        annual_rainfall_mm=420,
        climate_zone=ClimateZone.MONTANE,
        provenance={"climate_zone": "user"},
    )
    enriched = infer_context(kb, stated)
    assert enriched.climate_zone == ClimateZone.MONTANE


def test_completeness_rises_with_data():
    assert data_completeness(SiteProfile()) == 0.0
    assert data_completeness(SEMI_ARID_WHEAT) > data_completeness(
        SiteProfile(soil_ph=6.0)
    )


# --- causal graph -------------------------------------------------------------


def test_forward_propagation_reaches_biodiversity_from_soil_carbon(kb):
    effects = forward_effects(kb, "soil_organic_carbon_pct", profile=SEMI_ARID_WHEAT)
    metrics = {e.metric for e in effects}
    assert "soil_moisture_pct" in metrics
    assert "species_richness_observed" in metrics
    # multi-step reasoning: at least one effect must be reached via >1 edge
    assert any(e.depth > 1 for e in effects)
    # lag accumulates along the path
    multi = [e for e in effects if e.depth > 1]
    assert all(e.lag_years >= min(x.lag_years for x in effects) for e in multi)


def test_propagation_attenuates_near_saturation(kb):
    """Adding habitat to an already habitat-rich landscape must yield a weaker
    predicted effect than adding it to a cleared one."""
    poor = SiteProfile(native_vegetation_pct=3)
    rich = SiteProfile(native_vegetation_pct=45)
    poor_effect = next(
        e for e in forward_effects(kb, "native_vegetation_pct", profile=poor)
        if e.metric == "species_richness_observed"
    )
    rich_effect = next(
        e for e in forward_effects(kb, "native_vegetation_pct", profile=rich)
        if e.metric == "species_richness_observed"
    )
    assert poor_effect.strength > rich_effect.strength


def test_non_monotonic_edge_flips_sign_with_site_value(kb):
    """Tree cover below the recharge optimum should help the water table;
    above it, the sign must reverse rather than staying positive."""
    low_cover = SiteProfile(tree_cover_pct=5.0)
    high_cover = SiteProfile(tree_cover_pct=45.0)
    low = next(
        e for e in forward_effects(kb, "tree_cover_pct", profile=low_cover)
        if e.metric == "water_table_depth_m"
    )
    high = next(
        e for e in forward_effects(kb, "tree_cover_pct", profile=high_cover)
        if e.metric == "water_table_depth_m"
    )
    assert low.sign != high.sign


def test_unknown_value_on_non_monotonic_edge_is_context_dependent(kb):
    effects = forward_effects(kb, "tree_cover_pct", profile=SiteProfile())
    water = [e for e in effects if e.metric == "water_table_depth_m"]
    assert water and water[0].sign == "context_dependent"


def test_upstream_tracing_ranks_degraded_influential_causes(kb):
    assessments = assess_all(kb, SEMI_ARID_WHEAT)
    scores = {m: a.score for m, a in assessments.items()}
    causes = upstream_causes(kb, "species_richness_observed", SEMI_ARID_WHEAT, scores)
    assert causes
    observed = [c for c in causes if c.observed_score is not None]
    assert observed, "no observed upstream cause identified"
    # explanatory weight must be sorted descending
    weights = [c.explanatory_weight for c in causes]
    assert weights == sorted(weights, reverse=True)
    # a degraded influential metric must outrank a healthy influential one
    top = observed[0]
    assert top.observed_score < 0.6


# --- diagnosis ----------------------------------------------------------------


def test_diagnoses_are_multi_metric(engine):
    assessment = engine.assess(SEMI_ARID_WHEAT)
    assert assessment.diagnoses
    multi = [d for d in assessment.diagnoses if len(d.metrics_implicated) >= 2]
    assert len(multi) >= 3, "diagnoses are not connecting multiple variables"
    # at least one diagnosis must name an interaction with another metric
    assert any(d.interacting_metrics for d in assessment.diagnoses)


def test_limiting_factor_analysis_fires(engine):
    """The graph-derived diagnoses must appear, not just the hand-written rules."""
    profile = SEMI_ARID_WHEAT.model_copy(update={"species_richness_observed": 5})
    assessment = engine.assess(profile)
    codes = [d.code for d in assessment.diagnoses]
    assert any(c.startswith("limiting::") for c in codes)


def test_opposite_sites_get_opposite_nitrogen_advice(engine):
    """The single clearest test that the engine reasons rather than pattern-matches:
    a nitrogen-deficient site and a nitrogen-saturated site must not receive the
    same prescription."""
    depleted = engine.assess(SEMI_ARID_WHEAT)
    intensive = engine.assess(TEMPERATE_INTENSIVE)

    depleted_codes = {d.code for d in depleted.diagnoses}
    intensive_codes = {d.code for d in intensive.diagnoses}
    assert "nitrogen_limited" in depleted_codes
    assert "nutrient_enrichment" in intensive_codes
    assert "nutrient_enrichment" not in depleted_codes
    assert "nitrogen_limited" not in intensive_codes

    depleted_ids = {r.id for r in depleted.recommendations}
    intensive_ids = {r.id for r in intensive.recommendations}
    assert "nitrogen_rate_optimization" not in depleted_ids
    assert depleted_ids != intensive_ids


def test_grassland_and_cropland_diverge(engine):
    grass = {r.id for r in engine.assess(OVERGRAZED_RANGELAND).recommendations}
    crop = {r.id for r in engine.assess(SEMI_ARID_WHEAT).recommendations}
    assert grass != crop
    # cropland-only measures must not appear on rangeland
    assert "no_till_residue_retention" not in grass


def test_saline_site_gets_drainage_first_reasoning(engine):
    assessment = engine.assess(SALINE_IRRIGATED)
    codes = {d.code for d in assessment.diagnoses}
    assert "salinity_constraint" in codes
    salinity = next(d for d in assessment.diagnoses if d.code == "salinity_constraint")
    # the shallow water table must be identified as the binding constraint
    assert "drainage" in salinity.explanation.lower()


# --- recommendations ----------------------------------------------------------


def test_every_recommendation_is_fully_specified(engine):
    for profile in (SEMI_ARID_WHEAT, TEMPERATE_INTENSIVE, OVERGRAZED_RANGELAND, SALINE_IRRIGATED):
        assessment = engine.assess(profile)
        assert assessment.recommendations, f"no recommendations for {profile.land_use}"
        for rec in assessment.recommendations:
            assert rec.what_to_do.strip()
            assert len(rec.why_it_works.strip()) > 80, "mechanism is too thin to be reasoning"
            assert rec.impacted_metrics, "no impacted metrics"
            assert rec.evidence, "no evidence attached"
            assert rec.time_horizon.value in ("short", "medium", "long")
            assert 0 <= rec.confidence.score <= 1
            assert rec.confidence.label in ("low", "moderate", "high", "very_high")
            assert rec.monitoring, "no monitoring plan"
            assert rec.rationale_trace, "no audit trail"


def test_every_recommendation_carries_a_real_citation(engine):
    assessment = engine.assess(SEMI_ARID_WHEAT)
    for rec in assessment.recommendations:
        for item in rec.evidence:
            assert item["citation"].strip()
            assert item["doi"] or item["url"], f"{rec.id} cites a source with no locator"
            assert item["evidence_type"]


def test_recommendations_touch_at_least_three_metrics(engine):
    """The brief's hard constraint: no single-variable answers."""
    assessment = engine.assess(SEMI_ARID_WHEAT)
    for rec in assessment.recommendations:
        metrics = {m.metric for m in rec.impacted_metrics}
        assert len(metrics) >= 3, f"{rec.id} moves only {metrics}"


def test_recommendations_mix_primary_and_propagated_effects(engine):
    assessment = engine.assess(SEMI_ARID_WHEAT)
    rec = assessment.recommendations[0]
    assert any(not m.is_secondary for m in rec.impacted_metrics), "no cited primary effect"
    assert any(m.is_secondary for m in rec.impacted_metrics), "no propagated secondary effect"
    # propagated effects must state the pathway they travelled
    for impact in rec.impacted_metrics:
        if impact.is_secondary:
            assert impact.pathway and "->" in impact.pathway


def test_contraindications_exclude_inapplicable_measures(engine):
    """Biochar must not be recommended on an alkaline soil: the liming mechanism
    it works through does not exist there."""
    alkaline = SiteProfile(
        land_use=LandUse.CROPLAND, soil_ph=8.0, soil_organic_carbon_pct=0.5,
        annual_rainfall_mm=600, primary_crop="wheat",
    )
    assessment = engine.assess(alkaline, max_recommendations=10)
    assert "biochar_application" not in {r.id for r in assessment.recommendations}


def test_liming_not_recommended_on_semi_natural_grassland(engine):
    """Acid grassland has acidophilous specialists; liming it would displace the
    conservation interest, so the contraindication must hold even though the pH
    rule fires."""
    acid_grass = SiteProfile(
        land_use=LandUse.GRASSLAND, soil_ph=4.8, annual_rainfall_mm=900,
        soil_organic_carbon_pct=2.0,
    )
    assessment = engine.assess(acid_grass, max_recommendations=10)
    assert "liming_ph_correction" not in {r.id for r in assessment.recommendations}


def test_dry_site_excludes_water_competing_tree_measures(engine):
    very_dry = SiteProfile(
        land_use=LandUse.CROPLAND, annual_rainfall_mm=220, soil_organic_carbon_pct=0.3,
        primary_crop="millet",
    )
    ids = {r.id for r in engine.assess(very_dry, max_recommendations=10).recommendations}
    assert "agroforestry_alley_cropping" not in ids
    assert "legume_cover_crops" not in ids


def test_exclusions_are_explained_to_the_user(engine):
    assessment = engine.assess(SEMI_ARID_WHEAT)
    notes = " ".join(assessment.interaction_notes)
    assert "Ruled out for this site" in notes


def test_leverage_suppresses_pointless_carbon_advice(engine):
    """A carbon-rich soil should rank carbon-building measures lower than a
    depleted one does, because the headroom is smaller."""
    depleted = engine.assess(
        SiteProfile(land_use=LandUse.CROPLAND, soil_organic_carbon_pct=0.3,
                    annual_rainfall_mm=700, primary_crop="maize",
                    cropping_system=CroppingSystem.MONOCULTURE),
        max_recommendations=10,
    )
    rich = engine.assess(
        SiteProfile(land_use=LandUse.CROPLAND, soil_organic_carbon_pct=3.8,
                    annual_rainfall_mm=700, primary_crop="maize",
                    cropping_system=CroppingSystem.MONOCULTURE),
        max_recommendations=10,
    )

    def score_of(assessment, rec_id):
        return next((r.priority_score for r in assessment.recommendations if r.id == rec_id), None)

    depleted_score = score_of(depleted, "legume_cover_crops")
    rich_score = score_of(rich, "legume_cover_crops")
    assert depleted_score is not None, "carbon-building measure missing on a depleted soil"
    # On a carbon-rich soil the stronger correct behaviour is to drop the measure
    # entirely, because no carbon-depletion diagnosis fires; ranking it lower is
    # the acceptable weaker outcome.
    assert rich_score is None or rich_score < depleted_score


def test_confidence_is_capped_by_data_completeness(engine):
    thin = engine.assess(
        SiteProfile(land_use=LandUse.CROPLAND, soil_organic_carbon_pct=0.3)
    )
    rich = engine.assess(SEMI_ARID_WHEAT)
    assert thin.recommendations and rich.recommendations
    assert max(r.confidence.score for r in thin.recommendations) <= max(
        r.confidence.score for r in rich.recommendations
    )
    for rec in thin.recommendations:
        assert rec.confidence.limiters, "low-data recommendation claims no limitations"


def test_secondary_effects_are_declared_as_modelled(engine):
    """Propagated effects must be labelled so a reader never mistakes a model
    output for a measured result."""
    assessment = engine.assess(SEMI_ARID_WHEAT)
    for rec in assessment.recommendations:
        if any(m.is_secondary for m in rec.impacted_metrics):
            limiters = " ".join(rec.confidence.limiters).lower()
            assert "secondary" in limiters or "propagated" in limiters


def test_non_obvious_recommendations_surface(engine):
    """The brief rejects shallow advice, so the ranking must be able to put a
    non-obvious measure on top."""
    assessment = engine.assess(SEMI_ARID_WHEAT)
    assert any(r.novelty == "non_obvious" for r in assessment.recommendations[:3])


def test_sequencing_and_conflicts_are_reported(engine):
    invaded = SiteProfile(
        land_use=LandUse.DEGRADED,
        invasive_species_present=True,
        annual_rainfall_mm=700,
        soil_organic_carbon_pct=0.5,
        distance_to_natural_habitat_m=800,
    )
    assessment = engine.assess(invaded, max_recommendations=6)
    ids = {r.id for r in assessment.recommendations}
    if "invasive_species_control" in ids and len(ids) > 1:
        assert any("Sequencing" in n for n in assessment.interaction_notes)


def test_conflicting_pairs_are_reported_once(engine):
    """Conflicts are declared on both sides of the catalogue, so a naive loop
    emits each pair twice."""
    profile = SiteProfile(
        land_use=LandUse.CROPLAND, soil_organic_carbon_pct=0.3, annual_rainfall_mm=420,
        primary_crop="millet", cropping_system=CroppingSystem.MONOCULTURE,
        residue_removal="burned", tillage="conventional",
    )
    notes = engine.assess(profile, max_recommendations=8).interaction_notes
    conflicts = [n for n in notes if n.startswith("Conflict to resolve")]
    pairs = [frozenset(n.split("'")[1::2]) for n in conflicts]
    assert len(pairs) == len(set(pairs)), f"duplicate conflict notes: {conflicts}"


def test_residue_burning_blocks_the_conservation_agriculture_package(engine):
    """No-till plus residue retention cannot be implemented where residues are
    burned or removed, and no-till alone carries a yield penalty. The engine must
    exclude the package and say why, rather than recommending a broken version."""
    burning = SiteProfile(
        land_use=LandUse.CROPLAND, soil_organic_carbon_pct=0.3, annual_rainfall_mm=420,
        primary_crop="wheat", residue_removal="burned",
    )
    retaining = burning.model_copy(update={"residue_removal": "none"})
    burning_ids = {r.id for r in engine.assess(burning, max_recommendations=10).recommendations}
    retaining_ids = {r.id for r in engine.assess(retaining, max_recommendations=10).recommendations}
    assert "no_till_residue_retention" not in burning_ids
    assert "no_till_residue_retention" in retaining_ids


def test_determinism(engine):
    """No LLM, no randomness: identical input must give identical output."""
    first = engine.assess(SEMI_ARID_WHEAT)
    second = engine.assess(SEMI_ARID_WHEAT)
    assert [r.id for r in first.recommendations] == [r.id for r in second.recommendations]
    assert [r.priority_score for r in first.recommendations] == [
        r.priority_score for r in second.recommendations
    ]
    assert [d.code for d in first.diagnoses] == [d.code for d in second.diagnoses]


def test_empty_profile_does_not_crash(engine):
    assessment = engine.assess(SiteProfile())
    assert assessment.data_completeness == 0.0
    assert assessment.clarifying_questions
    # With nothing known, nothing may be asserted.
    assert not assessment.recommendations


def test_geo_context_is_accepted(engine):
    from bioai.schemas import GeoContext

    profile = SEMI_ARID_WHEAT.model_copy(
        update={"geo": GeoContext(latitude=26.9, longitude=75.8, region_name="Rajasthan")}
    )
    assessment = engine.assess(profile)
    assert assessment.profile.geo.has_point
    assert assessment.recommendations


def test_out_of_range_geo_is_rejected():
    from pydantic import ValidationError

    from bioai.schemas import GeoContext

    with pytest.raises(ValidationError):
        GeoContext(latitude=200, longitude=0)
