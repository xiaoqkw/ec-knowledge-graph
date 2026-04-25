import time
import uuid
from dataclasses import dataclass, field


@dataclass
class QASession:
    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)


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
        session.history.append(
            {
                "user": user_message,
                "assistant": assistant_message,
            }
        )
        if len(session.history) > self.max_turns:
            session.history = session.history[-self.max_turns :]
        self._sessions[session.session_id] = (session, time.time())

    def _cleanup(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, (_, touched_at) in self._sessions.items()
            if now - touched_at > self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
