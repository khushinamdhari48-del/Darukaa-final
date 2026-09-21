"""The assessment pipeline: profile in, evidence-backed assessment out.

    infer_context -> assess_metrics -> derive_indices -> diagnose
                  -> generate candidates -> retrieve evidence -> score
                  -> rank clarifying questions

This is the deterministic core. It contains no LLM call and no network access, so
the same profile always produces the same assessment, which is what makes the
system testable and auditable. The language model, when available, only
paraphrases this output and parses user input into it.
"""
from __future__ import annotations

from ..dialogue.clarify import rank_questions
from ..knowledge.loader import KnowledgeBase, load_knowledge_base
from ..knowledge.retriever import EvidenceIndex, get_index
from ..schemas import SiteAssessment, SiteProfile
from .diagnostics import diagnose
from .metrics import assess_all, data_completeness, derive_indices, infer_context
from .recommend import build_recommendations


class ReasoningEngine:
    def __init__(self, kb: KnowledgeBase | None = None, index: EvidenceIndex | None = None) -> None:
        self.kb = kb or load_knowledge_base()
        self.index = index or get_index()

    def assess(
        self,
        profile: SiteProfile,
        query_hint: str = "",
        max_recommendations: int = 4,
        explain_retrieval: bool = False,
        asked_fields: set[str] | None = None,
    ) -> SiteAssessment:
        enriched = infer_context(self.kb, profile)
        assessments = assess_all(self.kb, enriched)
        indices = derive_indices(assessments)
        completeness = data_completeness(enriched)
        diagnoses = diagnose(self.kb, enriched, assessments, indices)

        recommendations: list = []
        rejected: list = []
        notes: list[str] = []
        if completeness >= 0.0 and diagnoses:
            recommendations, rejected, notes = build_recommendations(
                kb=self.kb,
                index=self.index,
                profile=enriched,
                diagnoses=diagnoses,
                assessments=assessments,
                indices=indices,
                completeness=completeness,
                query_hint=query_hint,
                limit=max_recommendations,
            )

        questions = rank_questions(self.kb, enriched, exclude=asked_fields or set())

        # Interventions the site rules out are reported deliberately: "not this,
        # because X" is frequently the most decision-relevant output.
        if rejected:
            notes.append(
                "Ruled out for this site: "
                + "; ".join(
                    f"{r.title} ({r.reasons[0]})" for r in rejected[:5]
                )
            )

        for name, index in indices.items():
            if 0 < index.coverage < 0.4:
                notes.append(
                    f"The {name.replace('_', ' ')} index rests on only "
                    f"{index.coverage * 100:.0f}% of its components "
                    f"(missing: {', '.join(index.components_missing[:3])}), so treat it as "
                    f"indicative rather than measured."
                )

        retrieval_debug: list[dict] = []
        if explain_retrieval:
            _, trace = self.index.retrieve(
                query=query_hint or "site assessment", profile=enriched, top_k=8
            )
            retrieval_debug.append(trace.as_dict())
            for rec in recommendations:
                retrieval_debug.append(
                    {
                        "intervention": rec.id,
                        "cited": [e["card_id"] for e in rec.evidence],
                        "rationale_trace": rec.rationale_trace,
                    }
                )

        return SiteAssessment(
            profile=enriched,
            derived_indices={name: idx.score for name, idx in indices.items() if idx.coverage > 0},
            diagnoses=diagnoses,
            recommendations=recommendations,
            clarifying_questions=questions,
            data_completeness=completeness,
            interaction_notes=notes,
            retrieval_debug=retrieval_debug,
        )

    def index_coverage(self) -> dict[str, dict]:
        """Per-index coverage detail, used by the API's /assess response."""
        return {}


_ENGINE: ReasoningEngine | None = None


def get_engine() -> ReasoningEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ReasoningEngine()
    return _ENGINE
