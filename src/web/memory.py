import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CachedAlignedEntity:
    label: str
    raw_entity: str
    aligned_entity: str
    param_name: str
    matched: bool
    score_gap: float | None = None
    source_mode: str | None = None
    updated_at: float = field(default_factory=time.time)


@dataclass
class FailureMemory:
    trace_id: str
    failure_stage: str | None
    primary_failure_tag: str
    cypher_excerpt: str
    executed_params_excerpt: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class EntityAlignmentCache:
    def __init__(self, *, match_strategy: str = "exact"):
        self.match_strategy = match_strategy
        self._items: dict[str, CachedAlignedEntity] = {}

    def lookup(self, label: str, raw_entity: str) -> CachedAlignedEntity | None:
        if self.match_strategy != "exact":
            raise ValueError(f"Unsupported entity cache match strategy: {self.match_strategy}")
        return self._items.get(InMemoryQASessionStore.make_entity_cache_key(label, raw_entity))

    def upsert(self, item: CachedAlignedEntity) -> None:
        self._items[InMemoryQASessionStore.make_entity_cache_key(item.label, item.raw_entity)] = item

    def keys(self) -> list[str]:
        return list(self._items.keys())

    def values(self) -> list[CachedAlignedEntity]:
        return list(self._items.values())


@dataclass
class QATaskMemory:
    entity_cache: EntityAlignmentCache = field(default_factory=EntityAlignmentCache)
    recent_failures: list[FailureMemory] = field(default_factory=list)


@dataclass
class QASession:
    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    last_trace_id: str | None = None
    task_memory: QATaskMemory = field(default_factory=QATaskMemory)


class InMemoryQASessionStore:
    def __init__(self, ttl_seconds: int = 1800, max_turns: int = 4):
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._sessions: dict[str, tuple[QASession, float]] = {}

    def get_or_create(self, session_id: str | None) -> QASession:
        self._cleanup()
        if session_id:
            cached = self._sessions.get(session_id)
            if cached is not None:
                session, _ = cached
                self._sessions[session_id] = (session, time.time())
                return session

        session = QASession(session_id=session_id or uuid.uuid4().hex)
        self._sessions[session.session_id] = (session, time.time())
        return session

    def save_turn(self, session: QASession, user_message: str, assistant_message: str) -> None:
        self.commit_turn(
            session,
            user_message=user_message,
            assistant_message=assistant_message,
        )

    def commit_turn(
        self,
        session: QASession,
        *,
        user_message: str,
        assistant_message: str,
        trace_id: str | None = None,
        entity_cache_updates: list[CachedAlignedEntity] | None = None,
        recent_failure: FailureMemory | None = None,
    ) -> None:
        session.history.append(
            {
                "user": user_message,
                "assistant": assistant_message,
            }
        )
        if len(session.history) > self.max_turns:
            session.history = session.history[-self.max_turns :]
        session.last_trace_id = trace_id
        for item in entity_cache_updates or []:
            session.task_memory.entity_cache.upsert(item)
        if recent_failure is not None:
            session.task_memory.recent_failures.append(recent_failure)
            session.task_memory.recent_failures = session.task_memory.recent_failures[-5:]
        self._sessions[session.session_id] = (session, time.time())

    @staticmethod
    def make_entity_cache_key(label: str, raw_entity: str) -> str:
        normalized = " ".join(str(raw_entity or "").strip().lower().split())
        return f"{label}::{normalized}"

    def _cleanup(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, (_, touched_at) in self._sessions.items()
            if now - touched_at > self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
