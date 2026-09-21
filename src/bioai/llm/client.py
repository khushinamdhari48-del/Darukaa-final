"""Anthropic client wrapper with a hard offline guarantee.

Every method returns `None` (never raises, never blocks) when no API key is
configured, when the SDK is absent, or when the call fails. Callers treat `None`
as "use the deterministic path", so the system degrades in quality rather than
in function.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import settings
from ..schemas import SiteProfile
from . import prompts

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMClient:
    def __init__(self) -> None:
        self._client: Any | None = None
        self._init_error: str | None = None
        self.calls = 0
        if not settings.llm_available:
            self._init_error = (
                "ANTHROPIC_API_KEY not set" if not settings.anthropic_api_key else "LLM disabled"
            )
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        except ImportError:
            self._init_error = "anthropic SDK not installed"
        except Exception as exc:  # pragma: no cover - defensive
            self._init_error = f"client init failed: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def status(self) -> str:
        if self.available:
            return f"available (model={settings.model})"
        return f"unavailable ({self._init_error}); deterministic engine in use"

    # -- low level ----------------------------------------------------------
    def _complete(
        self, system: str, user: str, model: str | None = None, max_tokens: int | None = None
    ) -> str | None:
        if self._client is None:
            return None
        try:
            self.calls += 1
            response = self._client.messages.create(
                model=model or settings.model,
                max_tokens=max_tokens or settings.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
        except Exception as exc:
            logger.warning("LLM call failed, falling back to deterministic path: %s", exc)
            return None

    # -- extraction ---------------------------------------------------------
    def extract_profile(
        self, message: str, already_found: set[str] | None = None
    ) -> SiteProfile | None:
        already = ""
        if already_found:
            already = (
                "\nFields already extracted by the rule-based parser (you may omit these, "
                f"they will not be overwritten): {', '.join(sorted(already_found))}\n"
            )
        raw = self._complete(
            system=prompts.EXTRACTION_SYSTEM,
            user=prompts.EXTRACTION_USER.format(message=message, already=already),
            model=settings.extraction_model,
            max_tokens=1200,
        )
        if not raw:
            return None
        match = _JSON_BLOCK.search(raw)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("LLM extraction returned unparsable JSON; ignoring")
            return None
        if not isinstance(payload, dict):
            return None

        # Anything the model returns is validated field-by-field against the
        # schema, so a hallucinated key or out-of-range value is discarded rather
        # than entering the assessment.
        safe: dict[str, Any] = {}
        provenance: dict[str, str] = {}
        for key, value in payload.items():
            if value is None:
                continue
            try:
                SiteProfile(**{key: value})
            except Exception:
                logger.debug("discarding LLM-extracted field %s=%r", key, value)
                continue
            safe[key] = value
            provenance[key] = "user"
        if not safe:
            return None
        try:
            return SiteProfile(**safe, provenance=provenance)
        except Exception:
            return None

    # -- narration ----------------------------------------------------------
    def narrate(self, message: str, assessment_json: str, history: str) -> str | None:
        return self._complete(
            system=prompts.NARRATION_SYSTEM,
            user=prompts.NARRATION_USER.format(
                message=message or "(no new message; assessment requested directly)",
                history=history or "(first turn)",
                assessment=assessment_json,
            ),
            max_tokens=settings.max_tokens,
        )

    def ask_clarifying(
        self, message: str, known_json: str, questions_json: str, preliminary: str
    ) -> str | None:
        return self._complete(
            system=prompts.CLARIFY_SYSTEM,
            user=prompts.CLARIFY_USER.format(
                message=message,
                known=known_json,
                questions=questions_json,
                preliminary=preliminary or "(nothing conclusive yet)",
            ),
            max_tokens=900,
        )


_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
