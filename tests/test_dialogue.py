"""Extraction, conversational memory, clarifying-question selection and rendering.

Every test here runs with the LLM disabled, which is the point: the
conversational behaviour the brief asks for - clarifying questions, multi-turn
memory, context adaptation - is implemented in the engine, not delegated to a
model.
"""
from __future__ import annotations

import pytest

from bioai.dialogue.clarify import rank_questions
from bioai.dialogue.extract import rule_extract
from bioai.dialogue.orchestrator import Orchestrator
from bioai.dialogue.render import render_assessment, render_clarify
from bioai.dialogue.state import SessionStore
from bioai.knowledge.loader import load_knowledge_base
from bioai.reasoning.engine import ReasoningEngine
from bioai.schemas import ChatRequest, ClimateZone, CroppingSystem, LandUse, SiteProfile


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="module")
def engine():
    return ReasoningEngine()


@pytest.fixture
def orchestrator(engine):
    # A fresh store per test so sessions never leak between tests.
    return Orchestrator(engine=engine, store=SessionStore())


# --- extraction ----------------------------------------------------------------


def test_extracts_the_worked_example():
    patch = rule_extract(
        "Soil organic carbon is 0.3%, rainfall is low around 420mm, I grow monoculture "
        "wheat in a semi-arid region. pH is 5.2."
    )
    assert patch.soil_organic_carbon_pct == 0.3
    assert patch.annual_rainfall_mm == 420
    assert patch.soil_ph == 5.2
    assert patch.land_use == LandUse.CROPLAND
    assert patch.cropping_system == CroppingSystem.MONOCULTURE
    assert patch.climate_zone == ClimateZone.SEMI_ARID
    assert patch.primary_crop == "wheat"


def test_extracts_pressures_and_practices():
    patch = rule_extract(
        "I apply 220 kg N per hectare, spray 5 times a season, plough conventionally "
        "and burn the stubble. No irrigation. 40 hectares."
    )
    assert patch.fertilizer_n_kg_ha_yr == 220
    assert patch.pesticide_applications_per_season == 5
    assert patch.tillage == "conventional"
    assert patch.residue_removal == "burned"
    assert patch.irrigation == "none"
    assert patch.area_ha == 40


def test_organic_matter_is_converted_to_carbon():
    """Users frequently have organic matter, not carbon. Treating one as the
    other would misplace the site by nearly a factor of two."""
    patch = rule_extract("Soil organic matter is 3.4%")
    assert patch.soil_organic_carbon_pct == pytest.approx(3.4 / 1.724, rel=1e-3)


def test_kilometre_distances_are_converted():
    patch = rule_extract("The nearest woodland is about 1.5 km away")
    assert patch.distance_to_natural_habitat_m == 1500


def test_habitat_is_recognised_by_local_vocabulary():
    """Users do not say "semi-natural habitat". They say scrub, bush, woodland.
    A narrow vocabulary silently dropped the distance, which then made the
    isolation diagnosis unreachable."""
    for phrasing, expected in [
        ("Nearest scrub is 1.5 km away", 1500),
        ("the nearest bush patch is 600 m away", 600),
        ("nearest woodland is about 2 km", 2000),
        ("nearest remnant native vegetation 300m", 300),
    ]:
        assert rule_extract(phrasing).distance_to_natural_habitat_m == expected, phrasing


def test_negations_are_respected():
    patch = rule_extract("There is no surface water anywhere and no invasive weeds.")
    assert patch.permanent_water_present is False
    assert patch.invasive_species_present is False


def test_qualitative_rainfall_is_marked_inferred():
    """A number the system guessed must never be presented as one the user gave."""
    patch = rule_extract("Rainfall is low here and biodiversity is dropping")
    assert patch.annual_rainfall_mm is not None
    assert patch.provenance["annual_rainfall_mm"] == "inferred"


def test_decimal_geo_coordinates():
    patch = rule_extract("The plot is located at 0.35, 37.58 and grows maize")
    assert patch.geo.latitude == pytest.approx(0.35)
    assert patch.geo.longitude == pytest.approx(37.58)


def test_dms_geo_coordinates():
    patch = rule_extract("Site is at 26.9 N, 75.8 E")
    assert patch.geo.latitude == pytest.approx(26.9)
    assert patch.geo.longitude == pytest.approx(75.8)


def test_southern_and_western_hemispheres():
    patch = rule_extract("Coordinates 23.5 S, 46.6 W")
    assert patch.geo.latitude == pytest.approx(-23.5)
    assert patch.geo.longitude == pytest.approx(-46.6)


def test_unlabelled_number_pair_is_not_read_as_coordinates():
    """"I rotate 3, 4 crops" must not become a location."""
    patch = rule_extract("I rotate 3, 4 crops depending on the year")
    assert patch.geo.latitude is None


def test_implausible_values_are_dropped_not_clamped():
    patch = rule_extract("The pH is 47 and rainfall is 650 mm")
    assert patch.soil_ph is None       # outside any real pH range
    assert patch.annual_rainfall_mm == 650  # the valid sibling survives


def test_objectives_and_constraints_are_captured():
    patch = rule_extract(
        "I want to recover biodiversity without losing yield, but I have a limited "
        "budget and no machinery."
    )
    assert any("biodiversity" in o for o in patch.objectives)
    assert any("production" in o for o in patch.objectives)
    assert "capital-constrained" in patch.constraints
    assert "no machinery access" in patch.constraints


def test_extraction_never_raises_on_noise():
    for message in ("", "???", "hello", "0", "-- 999999999 %%%", "ph ph ph"):
        assert rule_extract(message) is not None


# --- clarifying questions -----------------------------------------------------


def test_questions_are_ranked_by_information_gain(kb):
    questions = rank_questions(kb, SiteProfile(), limit=5)
    assert questions
    gains = [q.expected_information_gain for q in questions]
    assert gains == sorted(gains, reverse=True)
    for question in questions:
        assert question.question.endswith("?")
        assert question.why_it_matters.strip()
        assert question.unit_or_options


def test_questions_do_not_repeat_known_fields(kb):
    profile = SiteProfile(soil_ph=6.0, soil_organic_carbon_pct=1.1, annual_rainfall_mm=700)
    fields = {q.field for q in rank_questions(kb, profile, limit=6)}
    assert "soil_ph" not in fields
    assert "soil_organic_carbon_pct" not in fields
    assert "annual_rainfall_mm" not in fields


def test_excluded_fields_are_not_reasked(kb):
    fields = {
        q.field for q in rank_questions(kb, SiteProfile(), limit=5, exclude={"soil_ph"})
    }
    assert "soil_ph" not in fields


def test_questions_adapt_to_the_site(kb):
    """A grazing site and a cropping site must not be asked the same things."""
    grazing = SiteProfile(land_use=LandUse.PASTURE, grazing_intensity_lsu_ha=1.8,
                          annual_rainfall_mm=500)
    cropping = SiteProfile(land_use=LandUse.CROPLAND, cropping_system=CroppingSystem.MONOCULTURE,
                           annual_rainfall_mm=500)
    grazing_fields = [q.field for q in rank_questions(kb, grazing, limit=5)]
    cropping_fields = [q.field for q in rank_questions(kb, cropping, limit=5)]
    assert grazing_fields != cropping_fields


def test_no_questions_when_nothing_is_left(kb):
    from bioai.dialogue.clarify import QUESTION_TEXT

    assert rank_questions(kb, SiteProfile(), exclude=set(QUESTION_TEXT)) == []


# --- multi-turn memory --------------------------------------------------------


def test_memory_accumulates_across_turns(orchestrator):
    first = orchestrator.handle(ChatRequest(message="I farm monoculture wheat."))
    session_id = first.session_id
    orchestrator.handle(
        ChatRequest(session_id=session_id, message="Soil organic carbon is 0.3%.")
    )
    third = orchestrator.handle(
        ChatRequest(session_id=session_id, message="Rainfall is 420 mm and pH is 5.2.")
    )
    profile = third.assessment.profile
    # values from all three turns must be present simultaneously
    assert profile.cropping_system == CroppingSystem.MONOCULTURE
    assert profile.soil_organic_carbon_pct == 0.3
    assert profile.annual_rainfall_mm == 420
    assert profile.soil_ph == 5.2
    assert third.turn == 3


def test_later_correction_overwrites_earlier_value(orchestrator):
    first = orchestrator.handle(ChatRequest(message="pH is 5.2 on my cropland."))
    second = orchestrator.handle(
        ChatRequest(session_id=first.session_id, message="Sorry, the pH is actually 6.4.")
    )
    assert second.assessment.profile.soil_ph == 6.4


def test_user_value_survives_a_later_inference(orchestrator):
    """An explicit number must not be displaced by a qualitative phrase later."""
    first = orchestrator.handle(ChatRequest(message="Rainfall is 1650 mm, humid, maize."))
    second = orchestrator.handle(
        ChatRequest(session_id=first.session_id, message="Rain is low compared to last year.")
    )
    assert second.assessment.profile.annual_rainfall_mm == 1650


def test_engine_asks_first_then_advises(orchestrator):
    """The brief's exact interaction: a vague opener triggers a question, not advice."""
    vague = orchestrator.handle(ChatRequest(message="Biodiversity is declining on my land"))
    assert vague.asked_for, "no clarifying question asked for a vague opener"
    assert not vague.assessment.recommendations
    assert "soil" in " ".join(q.field for q in vague.asked_for) or vague.asked_for

    detailed = orchestrator.handle(
        ChatRequest(
            session_id=vague.session_id,
            message="Soil organic carbon 0.3%, rainfall 420 mm, monoculture wheat, "
            "semi-arid, pH 5.2, nearest scrub 1.5 km, 4 sprays a season, stubble burned.",
        )
    )
    assert detailed.assessment.recommendations, "still no advice after a full description"


def test_engine_stops_asking_when_told_to(orchestrator):
    first = orchestrator.handle(ChatRequest(message="Biodiversity is declining."))
    assert first.asked_for
    second = orchestrator.handle(
        ChatRequest(
            session_id=first.session_id,
            message="I don't know any of those numbers, just tell me what to do. "
            "It is cropland with low rainfall.",
        )
    )
    assert not second.reply.startswith("## I need a few more")


def test_engine_gives_up_asking_after_two_rounds(orchestrator):
    session_id = None
    for _ in range(3):
        response = orchestrator.handle(
            ChatRequest(session_id=session_id, message="Biodiversity is declining on my land.")
        )
        session_id = response.session_id
    assert not response.reply.startswith("## I need a few more")


def test_structured_input_is_accepted_alongside_text(orchestrator):
    response = orchestrator.handle(
        ChatRequest(
            message="What should I do?",
            profile=SiteProfile(
                soil_organic_carbon_pct=0.4,
                annual_rainfall_mm=450,
                land_use=LandUse.CROPLAND,
                cropping_system=CroppingSystem.MONOCULTURE,
                primary_crop="millet",
                pesticide_applications_per_season=3,
                distance_to_natural_habitat_m=900,
            ),
        )
    )
    assert response.assessment.recommendations
    assert response.assessment.profile.primary_crop == "millet"


def test_llm_is_not_required(orchestrator):
    """The whole conversation must work with no API key present."""
    response = orchestrator.handle(
        ChatRequest(message="Cropland, 420 mm rainfall, SOC 0.3%, monoculture wheat, pH 5.2.")
    )
    assert response.llm_used is False
    assert response.reply.strip()


def test_session_isolation(orchestrator):
    a = orchestrator.handle(ChatRequest(message="pH is 5.0 on cropland."))
    b = orchestrator.handle(ChatRequest(message="pH is 8.0 on grassland."))
    assert a.session_id != b.session_id
    assert a.assessment.profile.soil_ph == 5.0
    assert b.assessment.profile.soil_ph == 8.0


def test_session_store_ttl_eviction():
    import time

    store = SessionStore(ttl_seconds=0)
    session = store.get_or_create(None)
    session.updated_at = time.time() - 10
    store.get_or_create(None)  # triggers eviction
    assert store.get(session.session_id) is None


# --- rendering ----------------------------------------------------------------


def test_assessment_renders_every_required_section(engine):
    profile = SiteProfile(
        soil_organic_carbon_pct=0.3, annual_rainfall_mm=420, land_use=LandUse.CROPLAND,
        cropping_system=CroppingSystem.MONOCULTURE, primary_crop="wheat", soil_ph=5.2,
        pesticide_applications_per_season=4, distance_to_natural_habitat_m=1500,
    )
    markdown = render_assessment(engine.assess(profile))
    for section in ("## Site assessment", "## Diagnosis", "## Recommendations"):
        assert section in markdown
    # the brief's mandatory per-recommendation fields
    for field in ("**What to do**", "**Why it works**", "**Metrics affected**",
                  "**Time horizon**", "**Confidence**", "**Evidence**"):
        assert field in markdown, f"missing {field}"
    assert "doi:" in markdown or "http" in markdown, "no locatable citation rendered"
    assert "Causal pathway" in markdown


def test_rendered_values_carry_units(engine):
    profile = SiteProfile(soil_organic_carbon_pct=0.3, annual_rainfall_mm=420,
                          land_use=LandUse.CROPLAND)
    markdown = render_assessment(engine.assess(profile))
    assert "0.3 %" in markdown
    assert "420 mm/yr" in markdown


def test_clarify_render_explains_itself(engine, kb):
    assessment = engine.assess(SiteProfile())
    markdown = render_clarify(assessment, assessment.clarifying_questions)
    assert "Expected information gain" in markdown
    assert "Why it changes the answer" in markdown


def test_render_is_ascii_safe(engine):
    """Output has to survive a Windows console and a plain-text email."""
    profile = SiteProfile(soil_organic_carbon_pct=0.3, annual_rainfall_mm=420,
                          land_use=LandUse.CROPLAND, soil_ph=5.2)
    markdown = render_assessment(engine.assess(profile), include_trace=True)
    markdown.encode("ascii")  # raises if any non-ASCII slipped in
