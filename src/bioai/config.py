"""Runtime configuration. Everything is overridable by environment variable so
the system can be deployed without code changes, and every default is chosen so
that the system runs with no network access and no API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
CORPUS_DIR = PACKAGE_ROOT / "knowledge" / "corpus"
DATASET_DIR = PACKAGE_ROOT / "knowledge" / "datasets"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- knowledge layer ---
    corpus_dir: Path = CORPUS_DIR
    dataset_dir: Path = DATASET_DIR
    index_dir: Path = field(
        default_factory=lambda: Path(os.getenv("BIOAI_INDEX_DIR", str(PROJECT_ROOT / ".index")))
    )
    embedder: str = field(default_factory=lambda: os.getenv("BIOAI_EMBEDDER", "hashed_tfidf"))
    vector_backend: str = field(default_factory=lambda: os.getenv("BIOAI_VECTOR_BACKEND", "numpy"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("BIOAI_EMBEDDING_DIM", "512")))

    # --- hybrid retrieval weights (dense / lexical / site-condition match) ---
    w_dense: float = field(default_factory=lambda: _env_float("BIOAI_W_DENSE", 0.45))
    w_lexical: float = field(default_factory=lambda: _env_float("BIOAI_W_LEXICAL", 0.30))
    w_condition: float = field(default_factory=lambda: _env_float("BIOAI_W_CONDITION", 0.25))
    top_k: int = field(default_factory=lambda: int(os.getenv("BIOAI_TOP_K", "12")))

    # --- LLM layer (optional) ---
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("BIOAI_MODEL", "claude-sonnet-5"))
    extraction_model: str = field(
        default_factory=lambda: os.getenv("BIOAI_EXTRACTION_MODEL", "claude-haiku-4-5-20251001")
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("BIOAI_MAX_TOKENS", "2000")))
    llm_enabled: bool = field(default_factory=lambda: _env_flag("BIOAI_LLM_ENABLED", True))

    # --- dialogue ---
    max_clarifying_questions: int = field(
        default_factory=lambda: int(os.getenv("BIOAI_MAX_QUESTIONS", "3"))
    )
    min_completeness_for_recommendations: float = field(
        default_factory=lambda: _env_float("BIOAI_MIN_COMPLETENESS", 0.12)
    )
    session_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("BIOAI_SESSION_TTL", "86400"))
    )

    @property
    def llm_available(self) -> bool:
        """True only when a key is present AND the LLM layer has not been
        switched off. Every code path must work when this is False."""
        return bool(self.anthropic_api_key) and self.llm_enabled


settings = Settings()
