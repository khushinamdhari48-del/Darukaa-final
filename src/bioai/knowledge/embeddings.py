"""Pluggable text embedding.

Default: `HashedTfidfEmbedder`, a deterministic, dependency-free embedder built
on feature hashing of word unigrams/bigrams plus character 4-grams, weighted by
corpus IDF and L2-normalised. It needs no model download, no network and no GPU,
which is what lets the whole system run offline and reproducibly.

Optional: `MiniLMEmbedder`, used when sentence-transformers is installed and
`BIOAI_EMBEDDER=minilm`. It gives better paraphrase matching at the cost of a
~90 MB model download.

Both satisfy the same protocol, so the retriever is indifferent to which is active.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np

from ..config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Domain stopwords: extremely common in this corpus and therefore uninformative.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for with by from as is are
    was were be been being it its their there which who whom whose while when where how why what
    can could may might will would should must not no nor so such also more most less least very
    into over under between among across during per about
    """.split()
)


class Embedder(Protocol):
    name: str
    dim: int

    def fit(self, documents: Sequence[str]) -> None: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _tokenize(text: str) -> list[str]:
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]
    features = list(words)
    features += [f"{a}_{b}" for a, b in pairwise(words)]
    # character 4-grams on longer words catch morphological variants
    # ("mycorrhiza" / "mycorrhizal", "infiltration" / "infiltrability")
    for word in words:
        if len(word) >= 7:
            features += [f"#{word[i:i + 4]}" for i in range(len(word) - 3)]
    return features


def _hash_index(feature: str, dim: int) -> tuple[int, float]:
    """Signed feature hashing: a stable bucket plus a +/-1 sign, which keeps the
    expected inner product unbiased despite collisions."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashedTfidfEmbedder:
    """Feature-hashed TF-IDF. Deterministic across runs and platforms."""

    name = "hashed_tfidf"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0
        self._fitted = False

    def fit(self, documents: Sequence[str]) -> None:
        n_docs = max(len(documents), 1)
        doc_freq: dict[str, int] = {}
        for doc in documents:
            for feature in set(_tokenize(doc)):
                doc_freq[feature] = doc_freq.get(feature, 0) + 1
        self._idf = {
            feature: math.log((n_docs + 1) / (freq + 1)) + 1.0 for feature, freq in doc_freq.items()
        }
        # Unseen features at query time get the IDF of a feature appearing once.
        self._default_idf = math.log((n_docs + 1) / 1.0) + 1.0
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("HashedTfidfEmbedder.fit() must be called before encode()")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for feature in _tokenize(text):
                counts[feature] = counts.get(feature, 0) + 1
            for feature, count in counts.items():
                idx, sign = _hash_index(feature, self.dim)
                tf = 1.0 + math.log(count)
                out[row, idx] += sign * tf * self._idf.get(feature, self._default_idf)
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out

    def state(self) -> dict:
        return {"idf": self._idf, "default_idf": self._default_idf, "dim": self.dim}

    def load_state(self, state: dict) -> None:
        self._idf = state["idf"]
        self._default_idf = state["default_idf"]
        self.dim = state["dim"]
        self._fitted = True


class MiniLMEmbedder:
    """sentence-transformers all-MiniLM-L6-v2. Optional."""

    name = "minilm"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, documents: Sequence[str]) -> None:  # pretrained; nothing to fit
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def state(self) -> dict:
        return {"dim": self.dim}

    def load_state(self, state: dict) -> None:
        return None


def build_embedder(name: str | None = None) -> Embedder:
    """Resolve the configured embedder, falling back to the offline default with a
    warning rather than failing if the optional dependency is missing."""
    requested = (name or settings.embedder).lower()
    if requested == "minilm":
        try:
            return MiniLMEmbedder()
        except ImportError:
            import warnings

            warnings.warn(
                "BIOAI_EMBEDDER=minilm requires sentence-transformers "
                "(pip install -r requirements-optional.txt); "
                "falling back to the offline hashed_tfidf embedder.",
                RuntimeWarning,
                stacklevel=2,
            )
    return HashedTfidfEmbedder()
