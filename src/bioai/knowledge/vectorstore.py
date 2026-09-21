"""Persistent vector store with a swappable backend.

`NumpyVectorStore` is the default: an .npz matrix plus a JSON sidecar, exact
cosine search, no dependencies. At this corpus size (tens to low thousands of
cards) exact search is both faster and more accurate than an ANN index, so the
default is also the right engineering choice, not only the portable one.

`ChromaVectorStore` is used when chromadb is installed and
`BIOAI_VECTOR_BACKEND=chroma`, for reviewers who want to see a "real" vector DB
in the loop. Both implement the same three operations the retriever needs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    backend: str

    def upsert(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None: ...
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
    def persist(self) -> None: ...
    def __len__(self) -> int: ...


class NumpyVectorStore:
    backend = "numpy"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._matrix_file = self.path / "vectors.npz"
        self._meta_file = self.path / "vectors.meta.json"
        self.ids: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.metadatas: list[dict] = []

    def upsert(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        if len(ids) != vectors.shape[0] or len(ids) != len(metadatas):
            raise ValueError("ids, vectors and metadatas must be the same length")
        self.ids = list(ids)
        self.matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        self.metadatas = list(metadatas)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.matrix.size == 0:
            return []
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        # Stored vectors are already L2-normalised, so the dot product is cosine.
        scores = self.matrix @ vector
        k = min(top_k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def persist(self) -> None:
        np.savez_compressed(self._matrix_file, matrix=self.matrix)
        self._meta_file.write_text(
            json.dumps({"ids": self.ids, "metadatas": self.metadatas}, indent=2),
            encoding="utf-8",
        )

    def load(self) -> bool:
        if not (self._matrix_file.exists() and self._meta_file.exists()):
            return False
        with np.load(self._matrix_file) as data:
            self.matrix = data["matrix"]
        meta = json.loads(self._meta_file.read_text(encoding="utf-8"))
        self.ids = meta["ids"]
        self.metadatas = meta["metadatas"]
        return True

    def __len__(self) -> int:
        return len(self.ids)


class ChromaVectorStore:
    backend = "chroma"

    def __init__(self, path: Path, collection: str = "bioai_evidence") -> None:
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path / "chroma"))
        # Embeddings are supplied by our own embedder, so Chroma's default
        # embedding function is deliberately bypassed.
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=[v.tolist() for v in np.asarray(vectors, dtype=np.float32)],
            metadatas=[{k: _flatten(v) for k, v in m.items()} for m in metadatas],
            documents=[m.get("text", "") for m in metadatas],
        )

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query, dtype=np.float32).reshape(-1).tolist()],
            n_results=min(top_k, count),
        )
        ids = result["ids"][0]
        distances = result.get("distances", [[0.0] * len(ids)])[0]
        # Chroma returns cosine distance; convert back to similarity.
        return [(cid, 1.0 - float(dist)) for cid, dist in zip(ids, distances, strict=False)]

    def persist(self) -> None:  # PersistentClient writes through
        return None

    def __len__(self) -> int:
        return int(self._collection.count())


def _flatten(value: object) -> str | int | float | bool:
    """Chroma metadata values must be scalars."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def build_vector_store(path: Path, backend: str | None = None) -> VectorStore:
    from ..config import settings

    requested = (backend or settings.vector_backend).lower()
    if requested == "chroma":
        try:
            return ChromaVectorStore(path)
        except ImportError:
            import warnings

            warnings.warn(
                "BIOAI_VECTOR_BACKEND=chroma requires chromadb "
                "(pip install -r requirements-optional.txt); "
                "falling back to the built-in numpy vector store.",
                RuntimeWarning,
                stacklevel=2,
            )
    return NumpyVectorStore(path)
