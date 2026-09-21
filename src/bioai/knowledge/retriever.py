"""Hybrid, site-conditioned retrieval over the evidence corpus.

Three signals are fused per card:

  1. dense       - cosine similarity between the query embedding and the card
                   embedding (semantic match on the question asked);
  2. lexical     - BM25 over the same text (exact match on rare technical terms
                   that hashing dilutes);
  3. condition   - how well the card's `applies_when` predicates match the actual
                   site profile.

The third signal is what separates this from a generic RAG pipeline. A card about
tree-crop water competition in the Sahel and a card about nitrogen deposition in
temperate heath can be equally similar to the text "biodiversity is declining",
but only one of them applies to a semi-arid wheat field. A card whose predicates
are *violated* by the site is demoted hard and its violation is reported, so a
recommendation can never be built on evidence that the site contradicts.

Retrieval is fully inspectable: `RetrievalTrace` records the per-signal scores and
the satisfied / violated / unknown predicates for every returned card.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..schemas import EvidenceCard, RetrievedEvidence, SiteProfile
from .embeddings import Embedder, HashedTfidfEmbedder, build_embedder
from .lexical import BM25
from .loader import KnowledgeBase, load_knowledge_base
from .vectorstore import NumpyVectorStore, VectorStore, build_vector_store

# A violated applicability predicate multiplies the fused score by this factor.
# It is not a hard filter: an explicitly contradicted card is still surfaced in
# the trace, because "this does not apply to you because X" is useful output.
VIOLATION_PENALTY = 0.15


@dataclass
class RetrievalTrace:
    query: str
    embedder: str
    vector_backend: str
    n_candidates: int
    weights: dict[str, float]
    results: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "embedder": self.embedder,
            "vector_backend": self.vector_backend,
            "corpus_size": self.n_candidates,
            "fusion_weights": self.weights,
            "results": self.results,
        }


class EvidenceIndex:
    """Builds, persists and queries the retrieval index."""

    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self.kb = kb or load_knowledge_base()
        self.index_dir = Path(index_dir or settings.index_dir)
        self.embedder = embedder or build_embedder()
        self.store = store or build_vector_store(self.index_dir)
        self.bm25 = BM25()
        self._built = False

    # -- build / load -------------------------------------------------------
    @property
    def _state_file(self) -> Path:
        return self.index_dir / "index_state.json"

    @staticmethod
    def _lexical_document(card: EvidenceCard) -> str:
        """BM25 document with the curated tag field weighted up.

        Tags are hand-assigned topic keys ("natural regeneration", "distance
        decay", "hydroperiod"), so a query token matching a tag is far more
        informative than the same token appearing once in prose. Repeating the
        tag field is the standard IR way to express that without a separate
        scoring term.
        """
        tags = " ".join(card.tags)
        return "\n".join([card.embedding_text(), tags, tags])

    def build(self, persist: bool = True) -> EvidenceIndex:
        cards = self.kb.cards
        documents = [c.embedding_text() for c in cards]
        self.embedder.fit(documents)
        vectors = self.embedder.encode(documents)
        self.store.upsert(
            ids=[c.id for c in cards],
            vectors=vectors,
            metadatas=[
                {
                    "id": c.id,
                    "domain": c.domain,
                    "tags": list(c.tags),
                    "interventions": list(c.interventions),
                    "citation": c.citation.short(),
                    "evidence_type": c.citation.type.value,
                    "text": c.claim,
                }
                for c in cards
            ],
        )
        self.bm25.fit([c.id for c in cards], [self._lexical_document(c) for c in cards])
        self._built = True
        if persist:
            self.persist()
        return self

    def persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.persist()
        self._state_file.write_text(
            json.dumps(
                {
                    "embedder": self.embedder.name,
                    "embedder_state": getattr(self.embedder, "state", lambda: {})(),
                    "bm25_state": self.bm25.state(),
                    "card_ids": [c.id for c in self.kb.cards],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> bool:
        """Restore a previously built index. Returns False when the persisted
        index is missing or stale relative to the current corpus."""
        if not self._state_file.exists():
            return False
        state = json.loads(self._state_file.read_text(encoding="utf-8"))
        if state.get("embedder") != self.embedder.name:
            return False
        if state.get("card_ids") != [c.id for c in self.kb.cards]:
            return False  # corpus changed; rebuild
        if isinstance(self.store, NumpyVectorStore) and not self.store.load():
            return False
        if isinstance(self.embedder, HashedTfidfEmbedder):
            self.embedder.load_state(state["embedder_state"])
        self.bm25.load_state(state["bm25_state"])
        self._built = True
        return True

    def ensure_built(self) -> EvidenceIndex:
        if self._built:
            return self
        if not self.load():
            self.build()
        return self

    # -- condition matching -------------------------------------------------
    def _condition_score(
        self, card: EvidenceCard, profile: SiteProfile | None
    ) -> tuple[float, list[str], list[str], list[str]]:
        """Score how well a card's applicability predicates match the site.

        Returns (score, satisfied, violated, unknown). With no predicates the
        card is universally applicable and scores neutral-positive (0.5) rather
        than 1.0, so that a card demonstrably matched to the site outranks a
        generic one.
        """
        satisfied: list[str] = []
        violated: list[str] = []
        unknown: list[str] = []
        if not card.applies_when:
            return 0.5, satisfied, violated, unknown
        if profile is None:
            return 0.5, satisfied, violated, [c.describe() for c in card.applies_when]

        for condition in card.applies_when:
            verdict = condition.evaluate(profile.get_field(condition.field))
            if verdict is True:
                satisfied.append(condition.describe())
            elif verdict is False:
                violated.append(condition.describe() + (f" ({condition.note})" if condition.note else ""))
            else:
                unknown.append(condition.describe())

        total = len(card.applies_when)
        # Unknowns are treated as half-credit: absence of evidence about the site
        # should neither reward nor punish the card as strongly as a real match.
        score = (len(satisfied) + 0.5 * len(unknown)) / total
        return score, satisfied, violated, unknown

    # -- query --------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        profile: SiteProfile | None = None,
        top_k: int | None = None,
        intervention_filter: set[str] | None = None,
        domain_filter: set[str] | None = None,
        include_violated: bool = False,
    ) -> tuple[list[RetrievedEvidence], RetrievalTrace]:
        self.ensure_built()
        top_k = top_k or settings.top_k

        query_vector = self.embedder.encode([query])[0]
        # Retrieve a wide dense candidate set, then re-rank with all three
        # signals: the fusion can promote a card the dense pass ranked low.
        dense_hits = dict(self.store.search(query_vector, top_k=max(top_k * 4, 40)))
        lexical_hits = self.bm25.score(query)

        candidate_ids = set(dense_hits) | set(lexical_hits)
        if intervention_filter:
            candidate_ids = {
                cid
                for cid in candidate_ids
                if intervention_filter & set(self.kb.cards_by_id[cid].interventions)
            }
            # Make sure every requested intervention gets a shot at citation even
            # if the free-text query did not mention it.
            for iv in intervention_filter:
                candidate_ids.update(c.id for c in self.kb.evidence_for_intervention(iv))
        if domain_filter:
            candidate_ids = {cid for cid in candidate_ids if self.kb.cards_by_id[cid].domain in domain_filter}

        # Raw cosine values from a sparse hashed embedding occupy a narrow band
        # (typically 0.05-0.35), so fusing them directly lets the condition term
        # dominate and flattens the ranking. Min-max normalising the dense signal
        # across the candidate set restores its discriminative power without
        # changing the relative order it produced.
        dense_norm = _minmax(dense_hits)

        scored: list[RetrievedEvidence] = []
        for cid in candidate_ids:
            card = self.kb.cards_by_id[cid]
            dense = float(dense_norm.get(cid, 0.0))
            lexical = float(lexical_hits.get(cid, 0.0))
            cond, satisfied, violated, unknown = self._condition_score(card, profile)
            fused = (
                settings.w_dense * max(dense, 0.0)
                + settings.w_lexical * lexical
                + settings.w_condition * cond
            )
            if violated:
                fused *= VIOLATION_PENALTY
            scored.append(
                RetrievedEvidence(
                    card=card,
                    dense_score=round(dense, 4),
                    lexical_score=round(lexical, 4),
                    condition_score=round(cond, 4),
                    score=round(fused, 4),
                    satisfied_conditions=satisfied,
                    violated_conditions=violated,
                    unknown_conditions=unknown,
                    retrieval_path=_path_label(dense, lexical),
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)

        # Site-applicable and site-contradicted evidence are ranked separately.
        # Because a violation multiplies the score by VIOLATION_PENALTY, mixing
        # them in one list would bury the contradicted cards below the cut and
        # `include_violated` would silently do nothing. Partitioning makes the
        # flag mean what it says: the contradicted cards are appended, after the
        # applicable ones, still carrying their penalised score so the ranking
        # signal is preserved.
        applicable = [r for r in scored if not r.violated_conditions]
        contradicted = [r for r in scored if r.violated_conditions]
        top = applicable[:top_k]
        if include_violated:
            top = top + contradicted[: max(3, top_k // 4)]

        trace = RetrievalTrace(
            query=query,
            embedder=self.embedder.name,
            vector_backend=self.store.backend,
            n_candidates=len(self.kb.cards),
            weights={
                "dense": settings.w_dense,
                "lexical": settings.w_lexical,
                "condition": settings.w_condition,
                "violation_penalty": VIOLATION_PENALTY,
            },
            results=[
                {
                    "card_id": r.card.id,
                    "citation": r.card.citation.short(),
                    "evidence_type": r.card.citation.type.value,
                    "score": r.score,
                    "dense": r.dense_score,
                    "lexical": r.lexical_score,
                    "condition": r.condition_score,
                    "path": r.retrieval_path,
                    "satisfied": r.satisfied_conditions,
                    "violated": r.violated_conditions,
                    "unknown": r.unknown_conditions,
                }
                for r in top
            ],
        )
        return top, trace

    def find_contradicted(
        self, profile: SiteProfile, intervention_ids: set[str] | None = None
    ) -> list[RetrievedEvidence]:
        """Cards whose applicability predicates the site actively violates.

        Used to state why an otherwise popular measure is *not* recommended here,
        which is often the most useful thing the system can say.
        """
        out: list[RetrievedEvidence] = []
        for card in self.kb.cards:
            if intervention_ids and not (intervention_ids & set(card.interventions)):
                continue
            cond, satisfied, violated, unknown = self._condition_score(card, profile)
            if violated:
                out.append(
                    RetrievedEvidence(
                        card=card,
                        condition_score=round(cond, 4),
                        satisfied_conditions=satisfied,
                        violated_conditions=violated,
                        unknown_conditions=unknown,
                        retrieval_path="condition_violation",
                    )
                )
        return out


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Scale to [0,1] over the candidate set. Positive values only: a negative
    cosine means the card points away from the query and should contribute
    nothing rather than becoming the new floor."""
    positive = {k: v for k, v in scores.items() if v > 0}
    if not positive:
        return {}
    low, high = min(positive.values()), max(positive.values())
    if high - low < 1e-9:
        return {k: 1.0 for k in positive}
    return {k: (v - low) / (high - low) for k, v in positive.items()}


def _path_label(dense: float, lexical: float) -> str:
    if dense > 0 and lexical > 0:
        return "hybrid"
    if dense > 0:
        return "dense_only"
    if lexical > 0:
        return "lexical_only"
    return "intervention_link"


_INDEX: EvidenceIndex | None = None


def get_index() -> EvidenceIndex:
    """Process-wide singleton so the index is built at most once."""
    global _INDEX
    if _INDEX is None:
        _INDEX = EvidenceIndex().ensure_built()
    return _INDEX
