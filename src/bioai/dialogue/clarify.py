"""Which question is worth asking?

Most conversational systems ask for whatever field is empty, in a fixed order.
This module instead estimates the *decision value* of each unknown field by
counterfactual simulation:

  for each unknown field:
      for each plausible probe value (low / high band, or worst / best category):
          re-run diagnosis and candidate generation with that value imputed
          measure how far the resulting recommendation set moves

A field whose plausible values all lead to the same advice is not worth a turn of
conversation, however empty it is. A field where the low and high cases lead to
opposite advice is the question to ask. This is a bounded approximation of
expected value of information, and it runs entirely on the deterministic
reasoning layer - no LLM call, no retrieval, so it is cheap enough to do live.

A structural prior (how central the field is in the causal graph, and how many
interventions gate on it) breaks ties and covers fields whose probe values do not
happen to flip a rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..knowledge.loader import KnowledgeBase
from ..reasoning.causal import metric_label
from ..reasoning.diagnostics import diagnose
from ..reasoning.metrics import CATEGORICAL_SCORES, DIAGNOSTIC_FIELDS, assess_all, derive_indices
from ..reasoning.recommend import generate_candidates
from ..schemas import ClarifyingQuestion, SiteProfile

# Candidate fields are capped so the simulation stays interactive.
MAX_FIELDS_TO_SIMULATE = 14

# How each field is phrased, and what unit or option list to offer back.
QUESTION_TEXT: dict[str, tuple[str, str]] = {
    "soil_organic_carbon_pct": (
        "What is the soil organic carbon in the topsoil?",
        "% by mass, 0-30 cm (a soil test report will have it; 'organic matter %' divided by 1.72 also works)",
    ),
    "soil_ph": ("What is the soil pH?", "pH units, 1:2.5 soil:water"),
    "soil_moisture_pct": ("Roughly what is the soil moisture in the growing season?", "volumetric %, or describe as dry / moist / wet"),
    "soil_texture": ("What is the soil texture?", "sand / sandy_loam / loam / silt_loam / clay_loam / clay"),
    "soil_bulk_density_g_cm3": ("Do you have a bulk density figure, or is the soil noticeably hard-setting?", "g/cm3"),
    "salinity_ec_ds_m": ("Is there any salinity, and if measured, what is the EC?", "dS/m"),
    "erosion_class": ("How visible is erosion - rills, gullies, exposed subsoil?", "none / slight / moderate / severe"),
    "slope_pct": ("What is the average slope of the land?", "% (or degrees, I will convert)"),
    "annual_rainfall_mm": ("What is the average annual rainfall?", "mm/year"),
    "rainfall_pattern": ("Is the rainfall one wet season, two, or erratic?", "unimodal / bimodal / erratic / aseasonal"),
    "dry_season_months": ("How many months of the year are effectively dry?", "months"),
    "mean_annual_temp_c": ("What is the mean annual temperature?", "degrees C"),
    "irrigation": ("Is the land irrigated, and by what method?", "none / rainfed_supplemental / flood / furrow / sprinkler / drip"),
    "land_use": ("What is the current land use?", "cropland / grassland / pasture / orchard / plantation / agroforestry / forest / degraded_barren / wetland / mangrove / peri_urban"),
    "cropping_system": ("Is it a single crop, a rotation, an intercrop, or does it include trees?", "monoculture / rotation / intercrop / agroforestry / fallow"),
    "primary_crop": ("What is the main crop or dominant vegetation?", "free text"),
    "tree_cover_pct": ("Roughly what percentage of the area has tree canopy over it?", "% canopy cover"),
    "native_vegetation_pct": (
        "About what share of the surrounding landscape is semi-natural vegetation - uncultivated scrub, woodland, grassland?",
        "% of landscape within roughly 1 km",
    ),
    "mean_patch_size_ha": ("What is the average size of an individual field or parcel?", "hectares"),
    "hedgerow_density_m_per_ha": ("Are there hedgerows or live boundaries, and roughly how much per hectare?", "metres per hectare"),
    "distance_to_natural_habitat_m": (
        "How far is the nearest patch of natural or semi-natural habitat?",
        "metres",
    ),
    "permanent_water_present": ("Is there any permanent or seasonal surface water on or beside the land?", "yes / no"),
    "species_richness_observed": ("Do you have any species count, from a survey or your own observation?", "number of species, and what was counted"),
    "pollinator_activity": ("How much bee and pollinator activity do you see during flowering?", "absent / low / moderate / high"),
    "earthworm_count_per_m2": ("If you dig a spade of moist soil, do you find earthworms, and roughly how many?", "count per m2, or count per spadeful x ~25"),
    "invasive_species_present": ("Are there aggressive invasive weeds or shrubs taking over?", "yes / no, and which species if known"),
    "fertilizer_n_kg_ha_yr": ("How much nitrogen fertiliser is applied per hectare per year?", "kg N/ha/yr"),
    "pesticide_applications_per_season": ("How many pesticide applications are made per season?", "number of applications"),
    "tillage": ("How is the soil cultivated?", "none / reduced / conventional / deep"),
    "grazing_intensity_lsu_ha": ("Is the land grazed, and at what stocking rate?", "livestock units per hectare"),
    "residue_removal": ("What happens to crop residues after harvest?", "left on field / partly removed / fully removed / burned"),
    "deforestation_last_5yr_pct": ("Has any tree cover been cleared in the last five years?", "% of area cleared"),
}

WHY_IT_MATTERS: dict[str, str] = {
    "soil_organic_carbon_pct": "Carbon is the hub variable: it sets water-holding capacity, nutrient supply and soil faunal habitat simultaneously, and the sequestration headroom depends entirely on the starting value.",
    "soil_ph": "pH gates whether legume-based nitrogen and carbon measures can work at all, so it changes the order of operations rather than just the choice.",
    "annual_rainfall_mm": "Rainfall decides whether added tree cover raises or lowers water availability, which flips the sign of the agroforestry recommendation.",
    "land_use": "Most evidence in the knowledge base is conditional on land use, so without it retrieval cannot filter out inapplicable studies.",
    "distance_to_natural_habitat_m": "Isolation determines whether in-field habitat measures have a source population to draw on, or whether connectivity has to come first.",
    "native_vegetation_pct": "The biodiversity response to added habitat is steeply non-linear, so the existing share decides how much each added hectare buys.",
    "pesticide_applications_per_season": "Pesticide pressure can cancel the effect of habitat measures entirely, so it determines whether they should be sequenced after input reduction.",
    "fertilizer_n_kg_ha_yr": "Nitrogen loading distinguishes two opposite prescriptions: add biological nitrogen, or reduce and re-time applications.",
    "tillage": "Tillage intensity governs both carbon loss and whether soil faunal and mycorrhizal measures can establish.",
    "grazing_intensity_lsu_ha": "Grazing intensity has a non-monotonic effect on richness, so without it the direction of the grazing recommendation is undetermined.",
    "residue_removal": "Residue handling is often the single largest carbon flux on a cropping site and is frequently the cheapest thing to change.",
    "tree_cover_pct": "Groundwater recharge peaks at intermediate tree cover, so the current value decides whether to add trees or stop adding them.",
    "salinity_ec_ds_m": "Salinity overrides other constraints and changes the species palette entirely.",
    "slope_pct": "Slope sets erosion risk and whether water-harvesting structures need engineered spillways.",
    "mean_patch_size_ha": "Parcel size determines whether margin-based measures reach the field interior at all.",
    "invasive_species_present": "Invasive dominance can arrest succession, making control a precondition rather than an option.",
    "permanent_water_present": "In drylands, free water limits which fauna can persist independently of how much vegetation there is.",
    "erosion_class": "Active erosion leaks any carbon gain, so it changes whether soil-building measures will hold.",
    "species_richness_observed": "An observed count lets the site be benchmarked against a land-use-matched baseline rather than assessed in the abstract.",
    "pollinator_activity": "Pollinator activity is the fastest-responding biodiversity indicator available without a survey budget.",
    "earthworm_count_per_m2": "Earthworm density is a free, reliable proxy for soil biological function and infiltration capacity.",
    "soil_moisture_pct": "Measured moisture distinguishes a real water deficit from one merely predicted by rainfall.",
    "soil_texture": "Texture sets the carbon-stabilisation ceiling and how much extra water each unit of carbon actually buys.",
    "cropping_system": "The cropping system determines whether diversification is the binding gap or already addressed.",
    "irrigation": "Irrigation method drives salinisation risk and determines whether water measures target capture or efficiency.",
    "deforestation_last_5yr_pct": "Active clearing outranks every enhancement measure, so it reorders the whole plan.",
    "dry_season_months": "Dry-season length, not annual total, determines which species and structures can persist.",
    "primary_crop": "The crop determines whether pollination is a yield pathway at all, and what the realistic rotation options are.",
}


# Fields that only make sense for certain land uses. Counterfactual simulation
# alone will happily rank "what is your stocking rate?" highly on a wheat field,
# because imputing a value does change the diagnosis set - but the question is
# absurd to the user and burns a turn. This gate is applied only when the land
# use is actually known.
FIELD_APPLICABILITY: dict[str, set[str]] = {
    "grazing_intensity_lsu_ha": {"grassland", "pasture", "agroforestry", "degraded_barren", "forest"},
    "cropping_system": {"cropland", "orchard", "plantation", "agroforestry"},
    "primary_crop": {"cropland", "orchard", "plantation", "agroforestry"},
    "residue_removal": {"cropland", "orchard", "plantation", "agroforestry"},
    "tillage": {"cropland", "orchard", "plantation", "agroforestry"},
    "pesticide_applications_per_season": {"cropland", "orchard", "plantation", "agroforestry", "peri_urban"},
    "fertilizer_n_kg_ha_yr": {"cropland", "orchard", "plantation", "agroforestry", "grassland", "pasture"},
    "irrigation": {"cropland", "orchard", "plantation", "agroforestry"},
    "mean_patch_size_ha": {"cropland", "orchard", "plantation", "pasture", "grassland", "agroforestry"},
    "hedgerow_density_m_per_ha": {
        "cropland", "orchard", "plantation", "pasture", "grassland", "agroforestry", "peri_urban",
    },
    "deforestation_last_5yr_pct": {
        "forest", "agroforestry", "plantation", "degraded_barren", "mangrove", "wetland", "cropland",
    },
}


def _is_relevant(field: str, profile: SiteProfile) -> bool:
    applicable = FIELD_APPLICABILITY.get(field)
    if applicable is None or profile.land_use is None:
        return True
    return profile.land_use.value in applicable


@dataclass
class FieldValue:
    field: str
    value: Any
    label: str


def _probe_values(kb: KnowledgeBase, field: str) -> list[FieldValue]:
    """Plausible low and high cases for a field, used as counterfactual probes."""
    if field in ("permanent_water_present", "invasive_species_present"):
        return [FieldValue(field, False, "no"), FieldValue(field, True, "yes")]

    if field in CATEGORICAL_SCORES:
        scale = CATEGORICAL_SCORES[field]
        worst = min(scale, key=lambda k: scale[k])
        best = max(scale, key=lambda k: scale[k])
        return [FieldValue(field, worst, worst), FieldValue(field, best, best)]

    bands = kb.bands_by_metric.get(field)
    if bands:
        low, high = bands[0], bands[-1]
        low_mid = (low.low + min(low.high, low.low * 3 + 1)) / 2
        high_mid = (high.low + min(high.high, high.low * 3 + 1)) / 2
        return [
            FieldValue(field, round(low_mid, 2), f"{low.band} ({low_mid:.2g})"),
            FieldValue(field, round(high_mid, 2), f"{high.band} ({high_mid:.2g})"),
        ]

    if field == "land_use":
        return [FieldValue(field, "cropland", "cropland"), FieldValue(field, "grassland", "grassland")]
    if field == "cropping_system":
        return [
            FieldValue(field, "monoculture", "monoculture"),
            FieldValue(field, "agroforestry", "agroforestry"),
        ]
    if field == "rainfall_pattern":
        return [FieldValue(field, "erratic", "erratic"), FieldValue(field, "unimodal", "unimodal")]
    if field == "dry_season_months":
        return [FieldValue(field, 1, "1 month"), FieldValue(field, 8, "8 months")]
    if field == "mean_annual_temp_c":
        return [FieldValue(field, 12, "12 C"), FieldValue(field, 30, "30 C")]
    if field == "hedgerow_density_m_per_ha":
        return [FieldValue(field, 0, "none"), FieldValue(field, 120, "dense")]
    if field == "species_richness_observed":
        return [FieldValue(field, 4, "very low"), FieldValue(field, 60, "high")]
    if field == "soil_nitrogen_pct":
        return [FieldValue(field, 0.04, "very low"), FieldValue(field, 0.35, "high")]
    if field == "water_table_depth_m":
        return [FieldValue(field, 1.0, "shallow"), FieldValue(field, 30.0, "deep")]
    return []


def _decision_state(kb: KnowledgeBase, profile: SiteProfile) -> tuple[set[str], float, list[str]]:
    """Reduce the current advice to a comparable fingerprint.

    Returns (candidate intervention ids, peak diagnosis severity, top diagnosis codes).
    Deliberately stops before retrieval and scoring, which is what keeps the
    simulation fast enough to run for a dozen fields on every turn.
    """
    assessments = assess_all(kb, profile)
    indices = derive_indices(assessments)
    diagnoses = diagnose(kb, profile, assessments, indices)
    candidates, _ = generate_candidates(kb, profile, diagnoses)
    ids = {c.intervention.id for c in candidates}
    peak = max((d.severity for d in diagnoses), default=0.0)
    codes = [d.code for d in diagnoses[:5]]
    return ids, peak, codes


def _divergence(
    base: tuple[set[str], float, list[str]], probe: tuple[set[str], float, list[str]]
) -> float:
    base_ids, base_peak, base_codes = base
    probe_ids, probe_peak, probe_codes = probe

    union = base_ids | probe_ids
    jaccard = (len(base_ids & probe_ids) / len(union)) if union else 1.0
    set_shift = 1.0 - jaccard

    severity_shift = abs(probe_peak - base_peak)

    code_union = set(base_codes) | set(probe_codes)
    code_shift = (
        1.0 - len(set(base_codes) & set(probe_codes)) / len(code_union) if code_union else 0.0
    )

    # Set membership change dominates: it means a different action would be advised.
    return 0.55 * set_shift + 0.25 * code_shift + 0.20 * min(severity_shift * 2, 1.0)


def _structural_prior(kb: KnowledgeBase, field: str) -> float:
    """Graph centrality plus decision-gating, normalised to roughly [0,1].

    Covers fields that matter but whose probe values happen not to flip a rule,
    for example because the relevant rule needs a second field as well.
    """
    edge_weight = sum(e.strength for e in kb.edges_from.get(field, [])) + sum(
        e.strength for e in kb.edges_to.get(field, [])
    )
    gating = sum(
        1
        for iv in kb.interventions
        for cond in (*iv.requires, *iv.contraindications)
        if cond.field == field
    )
    non_monotonic = any(
        e.sign == "non_monotonic"
        for e in (*kb.edges_from.get(field, []), *kb.edges_to.get(field, []))
    )
    prior = min(1.0, edge_weight / 4.0) * 0.5 + min(1.0, gating / 6.0) * 0.4
    if non_monotonic:
        # Without a value, the sign of the effect is literally unknown.
        prior += 0.2
    return min(1.0, prior)


def rank_questions(
    kb: KnowledgeBase,
    profile: SiteProfile,
    limit: int | None = None,
    exclude: set[str] | None = None,
) -> list[ClarifyingQuestion]:
    """Rank unknown fields by expected effect on the recommendation set."""
    limit = limit or settings.max_clarifying_questions
    exclude = exclude or set()

    unknown = [
        f
        for f in DIAGNOSTIC_FIELDS
        if profile.get_field(f) is None
        and f not in exclude
        and f in QUESTION_TEXT
        and _is_relevant(f, profile)
    ]
    if not unknown:
        return []

    # Pre-rank by structural prior so the expensive simulation is spent on the
    # fields most likely to matter.
    unknown.sort(key=lambda f: _structural_prior(kb, f), reverse=True)
    to_simulate = unknown[:MAX_FIELDS_TO_SIMULATE]

    baseline = _decision_state(kb, profile)

    scored: list[tuple[float, str]] = []
    for field in to_simulate:
        probes = _probe_values(kb, field)
        prior = _structural_prior(kb, field)
        if not probes:
            scored.append((0.35 * prior, field))
            continue
        divergences: list[float] = []
        for probe in probes:
            try:
                candidate_profile = profile.merge(
                    SiteProfile(**{field: probe.value}, provenance={field: "inferred"})  # type: ignore[arg-type]
                )
            except Exception:
                continue
            divergences.append(_divergence(baseline, _decision_state(kb, candidate_profile)))
        if not divergences:
            scored.append((0.35 * prior, field))
            continue
        # The spread between probe outcomes is the real information signal: if the
        # low and high cases give different advice, the answer decides something.
        spread = max(divergences) - min(divergences)
        expected = sum(divergences) / len(divergences)
        gain = 0.55 * expected + 0.25 * spread + 0.20 * prior
        scored.append((gain, field))

    scored.sort(key=lambda t: t[0], reverse=True)

    questions: list[ClarifyingQuestion] = []
    for gain, field in scored[:limit]:
        text, unit = QUESTION_TEXT[field]
        questions.append(
            ClarifyingQuestion(
                field=field,
                question=text,
                why_it_matters=WHY_IT_MATTERS.get(
                    field,
                    f"{metric_label(field).capitalize()} participates in the causal pathways "
                    f"behind the diagnoses on this site.",
                ),
                expected_information_gain=round(gain, 3),
                unit_or_options=unit,
            )
        )
    return questions
