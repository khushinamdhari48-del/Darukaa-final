"""Conversation memory.

A session holds the cumulative `SiteProfile` (not the raw transcript) as its
primary state. That is the important design choice: memory is structured, so
turn 7 can reason over a number given in turn 2 without re-reading the
transcript, and a later correction ("actually pH is 6.1") overwrites the slot
rather than contradicting it in context.

The transcript is kept only for tone continuity and for auditing.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from ..config import settings
from ..schemas import SiteProfile


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    extracted: dict | None = None


@dataclass
class Session:
    session_id: str
    profile: SiteProfile = field(default_factory=SiteProfile)
    turns: list[Turn] = field(default_factory=list)
    asked_fields: set[str] = field(default_factory=set)
    answered_fields: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    def add_turn(self, role: str, content: str, extracted: dict | None = None) -> None:
        self.turns.append(Turn(role=role, content=content, extracted=extracted))
        self.updated_at = time.time()

    def apply_patch(self, patch: SiteProfile) -> list[str]:
        """Fold a new patch into memory. Returns the fields that actually changed,
        which is what lets the assistant acknowledge specifically what it heard."""
        before = self.profile.known()
        self.profile = self.profile.merge(patch)
        after = self.profile.known()
        changed = [
            key for key, value in after.items() if before.get(key) != value
        ]
        self.answered_fields.update(changed)
        self.updated_at = time.time()
        return changed

    def history_text(self, max_turns: int = 6) -> str:
        recent = self.turns[-max_turns * 2 :]
        if not recent:
            return ""
        lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            content = turn.content.strip()
            if len(content) > 600:
                content = content[:600] + " [...]"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def unanswered_asked(self) -> set[str]:
        """Fields already asked about that the user has still not supplied. These
        are excluded from re-asking so the assistant does not nag."""
        return self.asked_fields - self.answered_fields


class SessionStore:
    """In-memory, thread-safe, TTL-expiring session store.

    Deliberately not a database: sessions are short-lived conversational state,
    and keeping the interface this narrow means swapping in Redis or Postgres is
    a single-class change if the deployment ever needs horizontal scaling.
    """

    def __init__(self, ttl_seconds: int | None = None) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        # `or` would silently discard a deliberate ttl of 0.
        self._ttl = settings.session_ttl_seconds if ttl_seconds is None else ttl_seconds

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            self._evict_expired()
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            new_id = session_id or uuid.uuid4().hex[:12]
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def reset(self, session_id: str) -> Session:
        with self._lock:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict_expired(self) -> None:
        cutoff = time.time() - self._ttl
        expired = [sid for sid, s in self._sessions.items() if s.updated_at < cutoff]
        for sid in expired:
            del self._sessions[sid]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


_STORE: SessionStore | None = None


def get_store() -> SessionStore:
    global _STORE
    if _STORE is None:
        _STORE = SessionStore()
    return _STORE
