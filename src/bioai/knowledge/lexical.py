"""BM25 lexical scoring.

Dense retrieval alone under-performs on this corpus because the decisive tokens
are rare technical terms - "sodicity", "hydroperiod", "zai", "Faidherbia",
"neonicotinoid". Feature hashing dilutes exactly those, so BM25 is run alongside
and the two scores are fused. Implemented directly to keep the dependency
footprint at zero.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2]


class BM25:
    """Okapi BM25 with the standard k1/b parameterisation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self._doc_tokens: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_length = 0.0
        self._idf: dict[str, float] = {}

    def fit(self, ids: Sequence[str], documents: Sequence[str]) -> None:
        self.ids = list(ids)
        self._doc_tokens = [Counter(tokenize(doc)) for doc in documents]
        self._doc_lengths = [sum(c.values()) for c in self._doc_tokens]
        n_docs = max(len(documents), 1)
        self._avg_length = (sum(self._doc_lengths) / n_docs) if n_docs else 0.0
        doc_freq: Counter[str] = Counter()
        for counts in self._doc_tokens:
            doc_freq.update(counts.keys())
        # Standard BM25 IDF, floored so that very common terms cannot go negative
        # and flip the sign of a match.
        self._idf = {
            term: max(math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)), 0.01)
            for term, freq in doc_freq.items()
        }

    def score(self, query: str) -> dict[str, float]:
        query_terms = tokenize(query)
        if not query_terms or not self.ids:
            return {}
        raw: dict[str, float] = {}
        for idx, doc_id in enumerate(self.ids):
            counts = self._doc_tokens[idx]
            length = self._doc_lengths[idx] or 1
            total = 0.0
            for term in query_terms:
                freq = counts.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self._avg_length or 1))
                total += self._idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            if total > 0:
                raw[doc_id] = total
        if not raw:
            return {}
        # Normalise to [0,1] so the fusion weights in settings mean the same thing
        # regardless of query length.
        peak = max(raw.values())
        return {k: v / peak for k, v in raw.items()}

    def state(self) -> dict:
        return {
            "ids": self.ids,
            "doc_tokens": [dict(c) for c in self._doc_tokens],
            "doc_lengths": self._doc_lengths,
            "avg_length": self._avg_length,
            "idf": self._idf,
            "k1": self.k1,
            "b": self.b,
        }

    def load_state(self, state: dict) -> None:
        self.ids = state["ids"]
        self._doc_tokens = [Counter(d) for d in state["doc_tokens"]]
        self._doc_lengths = state["doc_lengths"]
        self._avg_length = state["avg_length"]
        self._idf = state["idf"]
        self.k1 = state.get("k1", 1.5)
        self.b = state.get("b", 0.75)
