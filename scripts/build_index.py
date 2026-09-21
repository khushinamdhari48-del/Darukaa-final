#!/usr/bin/env python
"""Build and persist the retrieval index.

The index is built lazily on first use, so this script is only needed to
pre-warm a container image or to force a rebuild after editing the corpus.

    python scripts/build_index.py
    python scripts/build_index.py --force --backend chroma --embedder minilm
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bioai.config import settings
from bioai.knowledge.embeddings import build_embedder
from bioai.knowledge.loader import load_knowledge_base
from bioai.knowledge.retriever import EvidenceIndex
from bioai.knowledge.vectorstore import build_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="delete any existing index first")
    parser.add_argument("--embedder", help="hashed_tfidf (default) or minilm")
    parser.add_argument("--backend", help="numpy (default) or chroma")
    parser.add_argument("--index-dir", help="override the index directory")
    args = parser.parse_args()

    index_dir = Path(args.index_dir or settings.index_dir)
    if args.force and index_dir.exists():
        shutil.rmtree(index_dir)
        print(f"removed existing index at {index_dir}")

    started = time.perf_counter()
    kb = load_knowledge_base()
    print("knowledge base validated:")
    for key, value in kb.stats().items():
        print(f"  {key.replace('_', ' '):24s} {value}")

    index = EvidenceIndex(
        kb=kb,
        embedder=build_embedder(args.embedder),
        store=build_vector_store(index_dir, args.backend),
        index_dir=index_dir,
    ).build()

    elapsed = time.perf_counter() - started
    print(
        f"\nindexed {len(index.store)} evidence cards in {elapsed:.2f}s "
        f"(embedder={index.embedder.name}, vector store={index.store.backend})"
    )
    print(f"persisted to {index_dir}")

    # Smoke-test the round trip so a broken index is caught here, not in production.
    results, trace = index.retrieve("cover crops and soil organic carbon", top_k=3)
    print("\nsanity check - top 3 for 'cover crops and soil organic carbon':")
    for result in results:
        print(f"  {result.score:.3f}  {result.card.id}  [{result.card.citation.short()}]")
    if not results:
        print("ERROR: index returned no results")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
