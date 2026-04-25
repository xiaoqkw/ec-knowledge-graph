import time
import uuid
from dataclasses import replace

from dialogue.types import DialogueState


class InMemorySessionStore:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, tuple[DialogueState, float]] = {}

    def get_or_create(self, session_id: str | None) -> DialogueState:
        self._cleanup()
        if session_id:
            cached = self._sessions.get(session_id)
            if cached is not None:
                state, _ = cached
                self._sessions[session_id] = (state, time.time())
                return state

        new_state = DialogueState(session_id=session_id or uuid.uuid4().hex)
        self._sessions[new_state.session_id] = (new_state, time.time())
        return new_state

    def save(self, state: DialogueState) -> None:
        self._sessions[state.session_id] = (state, time.time())

    def reset(self, state: DialogueState) -> DialogueState:
        state.clear()
        self.save(state)
        return state

    def snapshot(self, session_id: str) -> DialogueState | None:
        cached = self._sessions.get(session_id)
        if cached is None:
            return None
        return replace(cached[0])

    def _cleanup(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, (_, touched_at) in self._sessions.items()
            if now - touched_at > self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
