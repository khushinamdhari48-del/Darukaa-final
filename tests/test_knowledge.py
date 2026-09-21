"""Knowledge base integrity.

These tests protect the scientific content: they are what stops a corpus edit
from silently introducing an unsourced claim, a dangling reference or an
unciteable intervention.
"""
from __future__ import annotations

from itertools import pairwise

import pytest

from bioai.knowledge.loader import load_knowledge_base
from bioai.schemas import EVIDENCE_WEIGHT


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def test_corpus_loads_and_is_substantial(kb):
    stats = kb.stats()
    assert stats["evidence_cards"] >= 60
    assert stats["causal_edges"] >= 40
    assert stats["interventions"] >= 30


def test_every_card_has_a_real_citation(kb):
    for card in kb.cards:
        c = card.citation
        assert c.authors.strip(), f"{card.id} has no authors"
        assert 1980 <= c.year <= 2026, f"{card.id} has implausible year {c.year}"
        assert c.venue.strip(), f"{card.id} has no venue"
        assert c.type.value in EVIDENCE_WEIGHT, f"{card.id} has unweighted evidence type"
        # Peer-reviewed work must be locatable: DOI, or a URL for institutional
        # reports which frequently have no DOI.
        assert c.doi or c.url, f"{card.id} has neither DOI nor URL"


def test_dois_are_well_formed(kb):
    for card in kb.cards:
        doi = card.citation.doi
        if doi is None:
            continue
        assert doi.startswith("10."), f"{card.id} DOI does not start with '10.': {doi}"
        assert "/" in doi, f"{card.id} DOI has no registrant/suffix separator: {doi}"
        assert " " not in doi, f"{card.id} DOI contains whitespace: {doi}"


def test_every_card_has_mechanism_and_claim(kb):
    """A claim without a mechanism is an assertion, not scientific reasoning."""
    for card in kb.cards:
        assert len(card.claim.strip()) > 40, f"{card.id} claim too thin"
        assert len(card.mechanism.strip()) > 80, f"{card.id} mechanism too thin"
        assert card.text.strip(), f"{card.id} has no retrievable text chunk"


def test_quantified_effects_are_machine_readable(kb):
    """At least half the corpus must carry numbers the engine can compose,
    otherwise recommendations degrade to qualitative advice."""
    with_effects = [c for c in kb.cards if c.effects]
    assert len(with_effects) / len(kb.cards) > 0.5
    for card in with_effects:
        for effect in card.effects:
            assert effect.absolute_change or effect.relative_change_pct, (
                f"{card.id} effect on {effect.metric} has no magnitude"
            )
            low, high = effect.time_to_effect_years
            assert 0 <= low <= high, f"{card.id} effect on {effect.metric} has bad time range"
            if effect.relative_change_pct:
                lo, hi = effect.relative_change_pct
                assert lo <= hi, f"{card.id} effect on {effect.metric} has inverted range"


def test_causal_edges_are_all_sourced(kb):
    """Referential integrity is enforced at load time; assert it holds so a
    future loader change cannot quietly drop the check."""
    card_ids = set(kb.cards_by_id)
    for edge in kb.edges:
        assert edge.evidence, f"edge {edge.source}->{edge.target} is unsourced"
        for cid in edge.evidence:
            assert cid in card_ids
        assert 0 < edge.strength <= 1
        assert edge.lag_years >= 0
        assert edge.sign in ("positive", "negative", "non_monotonic")
        assert len(edge.mechanism.strip()) > 60, (
            f"edge {edge.source}->{edge.target} mechanism too thin"
        )


def test_non_monotonic_edges_declare_an_optimum(kb):
    """A non-monotonic edge with no optimum cannot have its sign resolved, which
    would make the propagated effect meaningless."""
    for edge in kb.edges:
        if edge.sign == "non_monotonic":
            assert edge.optimum is not None, f"{edge.source}->{edge.target} lacks optimum"


def test_every_intervention_is_citeable(kb):
    for iv in kb.interventions:
        assert kb.evidence_for_intervention(iv.id), f"{iv.id} has no supporting evidence"
        assert iv.targets, f"{iv.id} targets no metric"
        assert iv.addresses, f"{iv.id} addresses no diagnosis"
        assert iv.monitoring, f"{iv.id} has no monitoring plan"
        assert iv.cost_intensity in ("low", "moderate", "high")
        assert iv.time_horizon in ("short", "medium", "long")
        assert iv.novelty in ("standard", "non_obvious")


def test_intervention_graph_is_symmetric_enough(kb):
    """Conflicts must be declared on both sides, or the set-level conflict check
    would miss them depending on ranking order."""
    for iv in kb.interventions:
        for other_id in iv.conflicts:
            other = kb.interventions_by_id[other_id]
            assert iv.id in other.conflicts, (
                f"{iv.id} declares a conflict with {other_id} but not vice versa"
            )


def test_metric_bands_are_contiguous_and_ordered(kb):
    for metric, bands in kb.bands_by_metric.items():
        for previous, current in pairwise(bands):
            assert previous.high == pytest.approx(current.low), (
                f"{metric} has a gap or overlap between bands "
                f"{previous.band} and {current.band}"
            )
        for band in bands:
            assert band.low < band.high, f"{metric} band {band.band} is inverted"
            assert band.verdict in ("critical", "poor", "moderate", "good", "high")
            assert band.interpretation.strip()
            assert band.source.strip(), f"{metric} band {band.band} has no source"


def test_band_lookup_covers_the_range(kb):
    assert kb.band_for("soil_organic_carbon_pct", 0.3).verdict == "critical"
    assert kb.band_for("soil_organic_carbon_pct", 2.5).verdict == "good"
    assert kb.band_for("soil_ph", 4.2).verdict == "critical"
    assert kb.band_for("soil_ph", 6.5).verdict == "good"
    # value above the top band must still resolve, not return None
    assert kb.band_for("soil_organic_carbon_pct", 55) is not None
    assert kb.band_for("soil_organic_carbon_pct", None) is None


def test_zone_inference_from_rainfall(kb):
    assert kb.zone_for_rainfall(300).climate_zone == "arid"
    assert kb.zone_for_rainfall(500).climate_zone == "semi_arid"
    assert kb.zone_for_rainfall(1800).climate_zone == "humid"
    # montane is temperature-defined and must never be inferred from rainfall
    assert kb.zone_for_rainfall(800).climate_zone != "montane"


def test_richness_baselines_are_ordered(kb):
    for baseline in kb.richness:
        assert baseline.expected_min < baseline.expected_max
        assert baseline.source.strip()


def test_citation_short_form_handles_author_counts(kb):
    from bioai.schemas import Citation, EvidenceType

    single = Citation(
        authors="Lal, R.", year=2004, title="t", venue="v", type=EvidenceType.FIELD_STUDY
    )
    pair = Citation(
        authors="Poeplau, C. & Don, A.", year=2015, title="t", venue="v",
        type=EvidenceType.META_ANALYSIS,
    )
    many = Citation(
        authors="Newbold, T., Hudson, L. N. & Hill, S. L. L.", year=2015, title="t",
        venue="v", type=EvidenceType.GLOBAL_ASSESSMENT,
    )
    institutional = Citation(
        authors="FAO", year=2017, title="t", venue="v",
        type=EvidenceType.TECHNICAL_GUIDELINE,
    )
    assert single.short() == "Lal 2004"
    assert pair.short() == "Poeplau & Don 2015"
    assert many.short() == "Newbold et al. 2015"
    assert institutional.short() == "FAO 2017"


def test_corpus_is_weighted_toward_strong_evidence(kb):
    """A knowledge base dominated by single field studies would not support the
    confidence levels the system reports."""
    strong = sum(
        1
        for c in kb.cards
        if c.citation.type.value
        in ("meta_analysis", "systematic_review", "global_assessment")
    )
    assert strong / len(kb.cards) > 0.5
