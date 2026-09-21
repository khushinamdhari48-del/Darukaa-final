"""Retrieval layer: correctness, site-conditioning, persistence and a measured
relevance baseline.

The relevance test is a real (small) evaluation set rather than a smoke test, so
a change to the embedder, the fusion weights or the tag field shows up as a
number rather than as a vague feeling that search "seems fine".
"""
from __future__ import annotations

import pytest

from bioai.knowledge.embeddings import HashedTfidfEmbedder, build_embedder
from bioai.knowledge.lexical import BM25
from bioai.knowledge.retriever import EvidenceIndex, get_index
from bioai.knowledge.vectorstore import NumpyVectorStore
from bioai.schemas import ClimateZone, LandUse, SiteProfile

# query -> the card a domain expert would expect first
RELEVANCE_SET: dict[str, str] = {
    "trees reduce water availability in drylands": "ev-intermediate-tree-cover-recharge",
    "pollinators declining far from natural habitat": "ev-pollination-distance-decay",
    "too much nitrogen fertiliser harming plant diversity": "ev-nitrogen-deposition-plant-diversity",
    "zai planting pits water harvesting": "ev-rainwater-harvesting-drylands",
    "does no-till actually increase soil carbon": "ev-tillage-soc-stratification",
    "habitat fragmentation small isolated patches": "ev-fragmentation-lasting-impact",
    "cover crops build soil organic carbon": "ev-cover-crop-soc",
    "salinity gypsum drainage reclamation": "ev-salinity-restoration",
    "rotational grazing rest period soil hydrology": "ev-rotational-grazing-soil-hydrology",
    "flower strips improve pest control": "ev-flower-strips-hedgerows-synthesis",
    "mangrove blue carbon storage": "ev-mangrove-carbon-and-protection",
    "smaller fields more crop types biodiversity": "ev-crop-heterogeneity-multitrophic",
}


@pytest.fixture(scope="module")
def index():
    return get_index()


def test_index_builds_over_whole_corpus(index):
    assert len(index.store) == len(index.kb.cards)
    assert index.embedder.name in ("hashed_tfidf", "minilm")


def test_retrieval_returns_scored_results(index):
    results, trace = index.retrieve("soil organic carbon is very low", top_k=5)
    assert 0 < len(results) <= 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.score > 0 for r in results)
    assert trace.results and len(trace.results) == len(results)
    assert set(trace.weights) >= {"dense", "lexical", "condition"}


def test_relevance_baseline(index):
    """Guards the retrieval quality we actually measured. Loosen only with a
    deliberate decision, never to make a failing change pass."""
    at1 = at3 = 0
    misses: list[str] = []
    for query, expected in RELEVANCE_SET.items():
        ids = [r.card.id for r in index.retrieve(query, top_k=3)[0]]
        if not ids:
            misses.append(query)
            continue
        if ids[0] == expected:
            at1 += 1
        if expected in ids:
            at3 += 1
        else:
            misses.append(f"{query!r} -> got {ids}")
    total = len(RELEVANCE_SET)
    assert at3 / total >= 0.85, f"recall@3 regressed to {at3}/{total}: {misses}"
    assert at1 / total >= 0.60, f"recall@1 regressed to {at1}/{total}: {misses}"


def test_hybrid_beats_each_signal_alone(index):
    """A rare technical term must be findable even when the dense signal is weak,
    which is the whole reason BM25 is fused in."""
    results, _ = index.retrieve("Faidherbia albida reverse phenology", top_k=3)
    assert any(r.card.id == "ev-faidherbia-evergreen-agriculture" for r in results)
    assert any(r.lexical_score > 0 for r in results)


def test_site_conditioning_changes_ranking(index):
    """The same query must return different evidence for different sites - this is
    the difference between this and a generic RAG pipeline."""
    query = "how do I improve biodiversity here"
    cropland = SiteProfile(
        land_use=LandUse.CROPLAND, annual_rainfall_mm=420, climate_zone=ClimateZone.SEMI_ARID
    )
    grassland = SiteProfile(
        land_use=LandUse.GRASSLAND, annual_rainfall_mm=900, climate_zone=ClimateZone.SUBHUMID
    )
    crop_ids = [r.card.id for r in index.retrieve(query, profile=cropland, top_k=8)[0]]
    grass_ids = [r.card.id for r in index.retrieve(query, profile=grassland, top_k=8)[0]]
    assert crop_ids != grass_ids


def test_violating_evidence_is_excluded_by_default(index):
    """A card whose applicability the site contradicts must not be returned as
    support, or a recommendation could be built on inapplicable evidence."""
    cropland = SiteProfile(land_use=LandUse.CROPLAND, annual_rainfall_mm=420)
    results, _ = index.retrieve("grazing intensity effects", profile=cropland, top_k=20)
    assert all(not r.violated_conditions for r in results)


def test_violating_evidence_is_available_on_request(index):
    cropland = SiteProfile(land_use=LandUse.CROPLAND, annual_rainfall_mm=420)
    results, _ = index.retrieve(
        "grazing intensity effects", profile=cropland, top_k=20, include_violated=True
    )
    assert any(r.violated_conditions for r in results)


def test_condition_scoring_distinguishes_three_states(index):
    card = index.kb.cards_by_id["ev-cover-crop-soc"]
    satisfied = SiteProfile(land_use=LandUse.CROPLAND, annual_rainfall_mm=600)
    violated = SiteProfile(land_use=LandUse.CROPLAND, annual_rainfall_mm=150)
    unknown = SiteProfile(land_use=LandUse.CROPLAND)

    s_score, s_ok, s_bad, s_unk = index._condition_score(card, satisfied)
    v_score, v_ok, v_bad, v_unk = index._condition_score(card, violated)
    u_score, u_ok, u_bad, u_unk = index._condition_score(card, unknown)

    assert s_score == 1.0 and s_ok and not s_bad
    assert v_bad and v_score < s_score
    assert u_unk and not u_bad


def test_find_contradicted_explains_why_not(index):
    """"Not this, because X" is often the most decision-relevant output."""
    mangrove_query_site = SiteProfile(land_use=LandUse.CROPLAND, annual_rainfall_mm=420)
    contradicted = index.find_contradicted(mangrove_query_site)
    assert contradicted
    for item in contradicted:
        assert item.violated_conditions
        assert item.retrieval_path == "condition_violation"


def test_intervention_filter_guarantees_citability(index):
    """Every intervention must be able to retrieve its own supporting evidence,
    even when the free-text query does not mention it."""
    for intervention in index.kb.interventions:
        results, _ = index.retrieve(
            query="site assessment", top_k=3, intervention_filter={intervention.id}
        )
        assert results, f"{intervention.id} retrieved no evidence"
        assert all(intervention.id in r.card.interventions for r in results)


def test_index_persists_and_reloads(tmp_path):
    first = EvidenceIndex(index_dir=tmp_path).build()
    baseline = [r.card.id for r in first.retrieve("cover crops soil carbon", top_k=5)[0]]

    second = EvidenceIndex(index_dir=tmp_path)
    assert second.load() is True
    reloaded = [r.card.id for r in second.retrieve("cover crops soil carbon", top_k=5)[0]]
    assert reloaded == baseline


def test_stale_index_is_rejected(tmp_path):
    """A corpus change must force a rebuild rather than serving stale vectors."""
    index = EvidenceIndex(index_dir=tmp_path).build()
    index.kb.cards = index.kb.cards[:-1]  # simulate a corpus edit
    fresh = EvidenceIndex(kb=index.kb, index_dir=tmp_path)
    assert fresh.load() is False


def test_embedder_is_deterministic():
    docs = ["soil organic carbon and microbial biomass", "pollinator foraging distance decay"]
    a, b = HashedTfidfEmbedder(dim=256), HashedTfidfEmbedder(dim=256)
    a.fit(docs)
    b.fit(docs)
    va, vb = a.encode(docs), b.encode(docs)
    assert (va == vb).all()
    # L2-normalised, so self-similarity is 1
    assert abs(float(va[0] @ va[0]) - 1.0) < 1e-5


def test_embedder_requires_fit():
    with pytest.raises(RuntimeError):
        HashedTfidfEmbedder().encode(["anything"])


def test_unknown_embedder_falls_back_without_crashing():
    embedder = build_embedder("definitely_not_a_real_embedder")
    assert embedder.name == "hashed_tfidf"


def test_bm25_ranks_the_containing_document_first():
    bm25 = BM25()
    bm25.fit(
        ["a", "b", "c"],
        [
            "gypsum amendment displaces exchangeable sodium and restores flocculation",
            "cover crops increase soil organic carbon through rhizodeposition",
            "hedgerows export native bees into adjacent crop fields",
        ],
    )
    scores = bm25.score("exchangeable sodium gypsum")
    assert scores and max(scores, key=scores.get) == "a"
    assert all(0 <= v <= 1 for v in scores.values())


def test_bm25_empty_query_is_safe():
    bm25 = BM25()
    bm25.fit(["a"], ["some text about soil"])
    assert bm25.score("") == {}
    assert bm25.score("!!!") == {}


def test_numpy_vector_store_cosine_search(tmp_path):
    import numpy as np

    store = NumpyVectorStore(tmp_path)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    store.upsert(["x", "y", "z"], vectors, [{}, {}, {}])
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
    assert results[0][0] == "x"
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)
    assert [r[0] for r in results] == ["x", "z", "y"]


def test_empty_vector_store_returns_nothing(tmp_path):
    import numpy as np

    store = NumpyVectorStore(tmp_path)
    assert store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3) == []
