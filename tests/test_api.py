"""HTTP surface contract tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bioai.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_healthz_reports_the_knowledge_base(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["knowledge_base"]["evidence_cards"] >= 60
    assert body["retrieval"]["indexed_vectors"] == body["knowledge_base"]["evidence_cards"]
    assert "llm" in body


def test_web_client_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Biodiversity Intelligence Engine" in response.text


def test_openapi_schema_is_valid(client):
    schema = client.get("/openapi.json").json()
    assert "/chat" in schema["paths"]
    assert "/assess" in schema["paths"]


def test_chat_round_trip_keeps_session(client):
    first = client.post("/chat", json={"message": "Biodiversity is declining on my land"}).json()
    assert first["reply"].strip()
    assert first["asked_for"], "vague opener should trigger a clarifying question"
    session_id = first["session_id"]

    second = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "SOC 0.3%, rainfall 420 mm, monoculture wheat, semi-arid, pH 5.2, "
            "nearest scrub 1.5 km, 4 sprays a season.",
        },
    ).json()
    assert second["session_id"] == session_id
    assert second["turn"] == 2
    assert second["assessment"]["recommendations"]


def test_chat_requires_some_input(client):
    assert client.post("/chat", json={"message": "   "}).status_code == 422


def test_assess_returns_structured_and_rendered(client):
    response = client.post(
        "/assess",
        json={
            "profile": {
                "soil_organic_carbon_pct": 0.3,
                "annual_rainfall_mm": 420,
                "land_use": "cropland",
                "cropping_system": "monoculture",
                "primary_crop": "wheat",
                "soil_ph": 5.2,
                "pesticide_applications_per_season": 4,
                "distance_to_natural_habitat_m": 1500,
                "geo": {"latitude": 26.9, "longitude": 75.8},
            },
            "query": "biodiversity declining",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assessment = body["assessment"]
    assert assessment["diagnoses"]
    assert assessment["recommendations"]
    assert "## Recommendations" in body["report_markdown"]

    for rec in assessment["recommendations"]:
        assert rec["what_to_do"]
        assert rec["why_it_works"]
        assert rec["impacted_metrics"]
        assert rec["evidence"]
        assert rec["time_horizon"] in ("short", "medium", "long")
        assert rec["confidence"]["label"] in ("low", "moderate", "high", "very_high")


def test_assess_rejects_invalid_values(client):
    response = client.post("/assess", json={"profile": {"soil_ph": 99}})
    assert response.status_code == 422


def test_assess_accepts_an_almost_empty_profile(client):
    body = client.post("/assess", json={"profile": {"land_use": "cropland"}}).json()
    assert body["assessment"]["clarifying_questions"]


def test_explain_retrieval_exposes_the_pipeline(client):
    body = client.post(
        "/assess",
        json={
            "profile": {"soil_organic_carbon_pct": 0.3, "annual_rainfall_mm": 420,
                        "land_use": "cropland"},
            "explain_retrieval": True,
        },
    ).json()
    debug = body["assessment"]["retrieval_debug"]
    assert debug
    assert "fusion_weights" in debug[0]
    assert any("rationale_trace" in entry for entry in debug)


def test_session_inspection_and_deletion(client):
    created = client.post("/chat", json={"message": "Cropland, pH 5.4, rainfall 600 mm"}).json()
    sid = created["session_id"]

    body = client.get(f"/session/{sid}").json()
    assert body["session_id"] == sid
    assert "soil_ph" in body["known_fields"]
    assert body["transcript"]

    assert client.delete(f"/session/{sid}").status_code == 200
    assert client.get(f"/session/{sid}").status_code == 404


def test_unknown_session_is_404(client):
    assert client.get("/session/does-not-exist").status_code == 404


def test_knowledge_search_exposes_per_signal_scores(client):
    body = client.post(
        "/knowledge/search",
        json={"query": "cover crops and soil organic carbon", "top_k": 5},
    ).json()
    assert body["results"]
    first = body["results"][0]
    assert set(first["scores"]) == {"fused", "dense", "lexical", "site_condition_match"}
    assert first["citation"]
    assert "fusion_weights" in body["trace"]


def test_knowledge_search_is_site_conditioned(client):
    body = client.post(
        "/knowledge/search",
        json={
            "query": "grazing management",
            "profile": {"land_use": "cropland", "annual_rainfall_mm": 420},
            "top_k": 10,
            "include_violated": True,
        },
    ).json()
    assert any(r["site_conditions"]["violated"] for r in body["results"])


def test_knowledge_stats_and_graph(client):
    stats = client.get("/knowledge/stats").json()
    assert stats["cards_by_evidence_type"]["meta_analysis"] > 0
    assert len(stats["citations"]) >= 60
    assert stats["interventions_by_novelty"]["non_obvious"] > 0

    graph = client.get("/knowledge/graph").json()
    assert graph["nodes"] and graph["edges"]
    for edge in graph["edges"]:
        assert edge["mechanism"]
        assert edge["evidence"]


def test_intervention_catalogue_is_exposed(client):
    body = client.get("/knowledge/interventions").json()
    assert body["interventions"]
    for iv in body["interventions"]:
        assert iv["supporting_cards"], f"{iv['id']} exposed with no evidence"
