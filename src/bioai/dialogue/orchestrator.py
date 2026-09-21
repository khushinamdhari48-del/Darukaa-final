"""Turn orchestration: the policy that decides whether to ask or to advise.

    user message
      -> extract (rules, then LLM for gaps)
      -> fold into session memory
      -> assess (deterministic engine)
      -> ASK if the site is too thinly described to be defensible
         ADVISE otherwise
      -> render (deterministic markdown, optionally paraphrased by the LLM)

The ask/advise decision is a policy, not a hard gate: it relaxes as the
conversation progresses so the system cannot get stuck interrogating a user who
does not have the numbers. Asking is also never all-or-nothing - an advisory
answer still carries the outstanding questions at the end.
"""
from __future__ import annotations

import json

from ..config import settings
from ..llm.client import get_client
from ..reasoning.engine import ReasoningEngine, get_engine
from ..schemas import ChatRequest, ChatResponse, SiteAssessment
from .clarify import rank_questions
from .extract import extract
from .render import render_assessment, render_clarify
from .state import Session, SessionStore, get_store

# Below this completeness the engine asks rather than advises - unless the user
# has already been asked twice, in which case it advises with whatever it has and
# says plainly what is provisional.
ASK_BELOW_COMPLETENESS = 0.18
MAX_CONSECUTIVE_ASKS = 2

# Phrasings that mean "just tell me" - the user is overriding the ask policy.
FORCE_ADVICE_MARKERS = (
    "just tell me",
    "give me recommendations",
    "what should i do",
    "don't ask",
    "dont ask",
    "no more questions",
    "that's all i have",
    "thats all i have",
    "i don't know",
    "i dont know",
    "not sure",
    "no data",
    "assume",
    "best guess",
)


class Orchestrator:
    def __init__(
        self,
        engine: ReasoningEngine | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.engine = engine or get_engine()
        self.store = store or get_store()
        self.llm = get_client()

    # -- main entry point ---------------------------------------------------
    def handle(self, request: ChatRequest) -> ChatResponse:
        session = self.store.get_or_create(request.session_id)
        llm_used = False

        # 1. structured input, if supplied, is authoritative and merged first
        if request.profile is not None:
            session.apply_patch(request.profile)

        # 2. free text
        if request.message.strip():
            patch, used = extract(request.message, use_llm=settings.llm_available)
            llm_used = llm_used or used
            session.apply_patch(patch)
            session.add_turn("user", request.message, extracted=patch.known() or None)

        # 3. deterministic assessment over everything known so far
        assessment = self.engine.assess(
            profile=session.profile,
            query_hint=request.message,
            max_recommendations=request.max_recommendations,
            explain_retrieval=request.explain_retrieval,
            asked_fields=session.unanswered_asked(),
        )
        session.profile = assessment.profile  # keep inferred context in memory

        # 4. ask or advise
        should_ask = self._should_ask(session, assessment, request.message)
        if should_ask:
            questions = assessment.clarifying_questions or rank_questions(
                self.engine.kb, session.profile, exclude=session.unanswered_asked()
            )
            questions = questions[: settings.max_clarifying_questions]
            session.asked_fields.update(q.field for q in questions)
            reply, used = self._render_clarify(session, assessment, questions, request.message)
            llm_used = llm_used or used
            session.add_turn("assistant", reply)
            return ChatResponse(
                session_id=session.session_id,
                reply=reply,
                assessment=assessment,
                asked_for=questions,
                turn=session.turn_count,
                llm_used=llm_used,
            )

        session.asked_fields.update(q.field for q in assessment.clarifying_questions)
        reply, used = self._render_advice(session, assessment, request)
        llm_used = llm_used or used
        session.add_turn("assistant", reply)
        return ChatResponse(
            session_id=session.session_id,
            reply=reply,
            assessment=assessment,
            asked_for=assessment.clarifying_questions,
            turn=session.turn_count,
            llm_used=llm_used,
        )

    # -- policy -------------------------------------------------------------
    def _should_ask(
        self, session: Session, assessment: SiteAssessment, message: str
    ) -> bool:
        lowered = message.lower()
        if any(marker in lowered for marker in FORCE_ADVICE_MARKERS):
            return False
        if not assessment.clarifying_questions:
            return False

        asks_so_far = sum(
            1
            for turn in session.turns
            if turn.role == "assistant" and turn.content.startswith("## I need a few more")
        )
        if asks_so_far >= MAX_CONSECUTIVE_ASKS:
            return False

        if assessment.data_completeness < ASK_BELOW_COMPLETENESS:
            return True

        # Even with adequate breadth, ask when the single most valuable unknown
        # would materially change the advice and nothing is yet recommendable.
        top_gain = assessment.clarifying_questions[0].expected_information_gain
        return not assessment.recommendations and top_gain > 0.2

    # -- rendering ----------------------------------------------------------
    def _render_clarify(
        self,
        session: Session,
        assessment: SiteAssessment,
        questions: list,
        message: str,
    ) -> tuple[str, bool]:
        deterministic = render_clarify(assessment, questions)
        if not self.llm.available:
            return deterministic, False
        preliminary = "\n".join(
            f"- {d.label}: {d.explanation.strip()}" for d in assessment.diagnoses[:2]
        )
        polished = self.llm.ask_clarifying(
            message=message,
            known_json=json.dumps(assessment.profile.known(), indent=2, default=str),
            questions_json=json.dumps(
                [q.model_dump() for q in questions], indent=2, default=str
            ),
            preliminary=preliminary,
        )
        if not polished:
            return deterministic, False
        # The deterministic block is retained below the prose so no detail the
        # engine produced is lost to paraphrase.
        return (
            f"{polished.strip()}\n\n---\n\n<details><summary>Engine detail: what I have and "
            f"why these questions</summary>\n\n{deterministic}\n\n</details>"
        ), True

    def _render_advice(
        self, session: Session, assessment: SiteAssessment, request: ChatRequest
    ) -> tuple[str, bool]:
        deterministic = render_assessment(assessment, include_trace=request.explain_retrieval)
        if not self.llm.available:
            return deterministic, False
        payload = _assessment_for_llm(assessment)
        polished = self.llm.narrate(
            message=request.message,
            assessment_json=json.dumps(payload, indent=2, default=str),
            history=session.history_text(),
        )
        if not polished:
            return deterministic, False
        return (
            f"{polished.strip()}\n\n---\n\n<details><summary>Full structured assessment "
            f"(engine output, verbatim)</summary>\n\n{deterministic}\n\n</details>"
        ), True


def _assessment_for_llm(assessment: SiteAssessment) -> dict:
    """Trim the assessment to what the narration prompt needs.

    Rationale traces and retrieval debug are excluded: they would consume the
    context budget and invite the model to reason about scoring rather than
    report conclusions.
    """
    return {
        "site": assessment.profile.known(),
        "data_completeness": assessment.data_completeness,
        "derived_indices": assessment.derived_indices,
        "diagnoses": [
            {
                "label": d.label,
                "severity": d.severity,
                "limiting_factor": d.limiting_factor,
                "explanation": d.explanation,
                "interacting_metrics": d.interacting_metrics,
            }
            for d in assessment.diagnoses[:6]
        ],
        "recommendations": [
            {
                "title": r.title,
                "what_to_do": r.what_to_do,
                "why_it_works": r.why_it_works,
                "causal_pathway": r.causal_pathway,
                "impacted_metrics": [
                    {
                        "metric": m.metric,
                        "direction": m.direction.value,
                        "expected_change": m.magnitude,
                        "time_to_effect_years": list(m.time_to_effect_years),
                        "basis": "secondary_causal_model" if m.is_secondary else "primary_cited_study",
                        "pathway": m.pathway,
                    }
                    for m in r.impacted_metrics
                ],
                "time_horizon": r.time_horizon.value,
                "confidence": {
                    "label": r.confidence.label,
                    "score": r.confidence.score,
                    "supported_by": r.confidence.drivers,
                    "limited_by": r.confidence.limiters,
                },
                "evidence": [
                    {
                        "citation": e["citation"],
                        "evidence_type": e["evidence_type"],
                        "finding": e["claim"],
                        "applicability_confirmed": e["matched_site_conditions"],
                    }
                    for e in r.evidence
                ],
                "preconditions": r.preconditions,
                "risks_and_tradeoffs": r.risks_and_tradeoffs,
                "synergies": r.synergies,
                "conflicts": r.conflicts,
                "monitoring": r.monitoring,
                "cost_intensity": r.cost_intensity,
                "reversibility": r.reversibility,
                "novelty": r.novelty,
            }
            for r in assessment.recommendations
        ],
        "interactions_and_exclusions": assessment.interaction_notes,
        "outstanding_questions": [
            {"question": q.question, "why_it_matters": q.why_it_matters}
            for q in assessment.clarifying_questions
        ],
    }


_ORCHESTRATOR: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = Orchestrator()
    return _ORCHESTRATOR
