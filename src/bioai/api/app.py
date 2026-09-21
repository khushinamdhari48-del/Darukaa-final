"""FastAPI surface.

Endpoints:
  POST /chat              multi-turn conversation (text and/or structured profile)
  POST /assess            one-shot structured assessment, no dialogue policy
  GET  /session/{id}      inspect accumulated conversation memory
  DELETE /session/{id}    clear a session
  POST /knowledge/search  inspect the retrieval layer directly
  GET  /knowledge/stats   knowledge base size and composition
  GET  /knowledge/graph   the causal graph, for inspection or visualisation
  GET  /healthz           liveness plus LLM/index status
  GET  /                  minimal web client
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..dialogue.orchestrator import get_orchestrator
from ..dialogue.render import render_assessment
from ..dialogue.state import get_store
from ..knowledge.loader import load_knowledge_base
from ..knowledge.retriever import get_index
from ..llm.client import get_client
from ..reasoning.engine import get_engine
from ..schemas import ChatRequest, ChatResponse, SiteAssessment, SiteProfile

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the knowledge base and retrieval index at startup.

    Doing it here rather than lazily means a malformed evidence card fails the
    deploy instead of the first user request, and no request pays the build cost.
    """
    load_knowledge_base()
    get_index()
    get_engine()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Darukaa Biodiversity Intelligence Engine",
    version=__version__,
    description=(
        "Evidence-grounded environmental advisory system. A curated, citation-carrying "
        "knowledge base and a deterministic multi-metric reasoning engine produce every "
        "recommendation; the language model only parses input and phrases output."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssessRequest(BaseModel):
    profile: SiteProfile
    query: str = Field("", description="optional free-text context to steer retrieval")
    max_recommendations: int = Field(4, ge=1, le=10)
    explain_retrieval: bool = False
    render_markdown: bool = Field(
        True, description="include the rendered markdown report alongside the structured result"
    )


class AssessResponse(BaseModel):
    assessment: SiteAssessment
    report_markdown: str | None = None


class SearchRequest(BaseModel):
    query: str
    profile: SiteProfile | None = None
    top_k: int = Field(8, ge=1, le=40)
    include_violated: bool = Field(
        False, description="also return cards whose applicability the site contradicts"
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    kb = load_knowledge_base()
    idx = get_index()
    return {
        "status": "ok",
        "version": __version__,
        "knowledge_base": kb.stats(),
        "retrieval": {
            "embedder": idx.embedder.name,
            "vector_backend": idx.store.backend,
            "indexed_vectors": len(idx.store),
        },
        "llm": get_client().status,
        "active_sessions": len(get_store()),
        "note": (
            "The reasoning engine is deterministic and runs without an LLM. "
            "Recommendations, effect sizes and citations never originate from the model."
        ),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip() and request.profile is None:
        raise HTTPException(status_code=422, detail="provide `message`, `profile`, or both")
    return get_orchestrator().handle(request)


@app.post("/assess", response_model=AssessResponse)
def assess(request: AssessRequest) -> AssessResponse:
    """Structured-input assessment. Skips the ask/advise dialogue policy and
    always returns the full assessment for whatever data was supplied."""
    engine = get_engine()
    assessment = engine.assess(
        profile=request.profile,
        query_hint=request.query,
        max_recommendations=request.max_recommendations,
        explain_retrieval=request.explain_retrieval,
    )
    return AssessResponse(
        assessment=assessment,
        report_markdown=render_assessment(
            assessment, include_trace=request.explain_retrieval
        )
        if request.render_markdown
        else None,
    )


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {
        "session_id": session.session_id,
        "turns": session.turn_count,
        "accumulated_profile": session.profile.model_dump(exclude_none=True),
        "known_fields": list(session.profile.known()),
        "fields_asked": sorted(session.asked_fields),
        "fields_answered": sorted(session.answered_fields),
        "still_outstanding": sorted(session.unanswered_asked()),
        "transcript": [
            {"role": t.role, "content": t.content, "extracted": t.extracted}
            for t in session.turns
        ],
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    deleted = get_store().delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"deleted": session_id}


@app.post("/knowledge/search")
def knowledge_search(request: SearchRequest) -> dict[str, Any]:
    """Direct access to the retrieval layer, so the knowledge pipeline can be
    inspected independently of the reasoning that consumes it."""
    idx = get_index()
    results, trace = idx.retrieve(
        query=request.query,
        profile=request.profile,
        top_k=request.top_k,
        include_violated=request.include_violated,
    )
    return {
        "trace": trace.as_dict(),
        "results": [
            {
                "card_id": r.card.id,
                "domain": r.card.domain,
                "claim": r.card.claim.strip(),
                "mechanism": r.card.mechanism.strip(),
                "citation": r.card.citation.formatted(),
                "evidence_type": r.card.citation.type.value,
                "effects": [e.model_dump() for e in r.card.effects],
                "caveats": r.card.caveats,
                "interventions": r.card.interventions,
                "scores": {
                    "fused": r.score,
                    "dense": r.dense_score,
                    "lexical": r.lexical_score,
                    "site_condition_match": r.condition_score,
                },
                "site_conditions": {
                    "satisfied": r.satisfied_conditions,
                    "violated": r.violated_conditions,
                    "unknown": r.unknown_conditions,
                },
            }
            for r in results
        ],
    }


@app.get("/knowledge/stats")
def knowledge_stats() -> dict[str, Any]:
    kb = load_knowledge_base()
    by_type: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for card in kb.cards:
        by_type[card.citation.type.value] = by_type.get(card.citation.type.value, 0) + 1
        by_domain[card.domain] = by_domain.get(card.domain, 0) + 1
    return {
        **kb.stats(),
        "cards_by_evidence_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "cards_by_domain": dict(sorted(by_domain.items(), key=lambda kv: -kv[1])),
        "interventions_by_novelty": {
            "non_obvious": sum(1 for i in kb.interventions if i.novelty == "non_obvious"),
            "standard": sum(1 for i in kb.interventions if i.novelty == "standard"),
        },
        "citations": sorted({c.citation.formatted() for c in kb.cards}),
    }


@app.get("/knowledge/graph")
def knowledge_graph() -> dict[str, Any]:
    kb = load_knowledge_base()
    nodes = sorted({e.source for e in kb.edges} | {e.target for e in kb.edges})
    return {
        "nodes": nodes,
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "sign": e.sign,
                "strength": e.strength,
                "lag_years": e.lag_years,
                "optimum": e.optimum,
                "threshold": e.threshold,
                "saturating_at": e.saturating_at,
                "mechanism": e.mechanism,
                "evidence": list(e.evidence),
            }
            for e in kb.edges
        ],
    }


@app.get("/knowledge/interventions")
def knowledge_interventions() -> dict[str, Any]:
    kb = load_knowledge_base()
    return {
        "interventions": [
            {
                "id": i.id,
                "title": i.title,
                "what_to_do": i.what_to_do,
                "targets": list(i.targets),
                "addresses": list(i.addresses),
                "requires": [c.describe() for c in i.requires],
                "contraindications": [c.describe() for c in i.contraindications],
                "cost_intensity": i.cost_intensity,
                "reversibility": i.reversibility,
                "time_horizon": i.time_horizon,
                "novelty": i.novelty,
                "supporting_cards": [
                    c.id for c in kb.evidence_for_intervention(i.id)
                ],
            }
            for i in kb.interventions
        ]
    }


@app.exception_handler(ValueError)
def _value_error(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
