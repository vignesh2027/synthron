"""Session manager — track active user sessions and their state."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from synthron.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Session:
    """Represents a user session with an orchestrator."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    task_count: int = 0
    total_tokens: int = 0
    active_task: str = ""
    status: str = "idle"  # idle | running | error
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update last-active timestamp."""
        self.last_active = time.time()

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_s(self) -> float:
        return time.time() - self.last_active

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "task_count": self.task_count,
            "total_tokens": self.total_tokens,
            "active_task": self.active_task,
            "status": self.status,
        }


class SessionManager:
    """Manages all active Synthron sessions.

    Sessions are identified by a short UUID. Idle sessions (>1h) are auto-expired.
    """

    SESSION_TTL_SECONDS = 3600  # 1 hour idle expiry

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, metadata: dict | None = None) -> Session:
        """Create a new session.

        Args:
            metadata: Optional metadata to attach.

        Returns:
            New Session object.
        """
        session = Session(metadata=metadata or {})
        self._sessions[session.id] = session
        logger.debug(f"[session] Created: {session.id}")
        return session

    def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID, or None if expired/not found."""
        session = self._sessions.get(session_id)
        if session and session.idle_s > self.SESSION_TTL_SECONDS:
            self.delete(session_id)
            return None
        if session:
            session.touch()
        return session

    def get_or_create(self, session_id: str = "") -> Session:
        """Get existing session or create a new one.

        Args:
            session_id: Existing session ID, or empty to create new.

        Returns:
            Session object.
        """
        if session_id:
            session = self.get(session_id)
            if session:
                return session
        return self.create()

    def update(self, session_id: str, **kwargs: Any) -> None:
        """Update session fields."""
        session = self._sessions.get(session_id)
        if session:
            for k, v in kwargs.items():
                if hasattr(session, k):
                    setattr(session, k, v)
            session.touch()

    def delete(self, session_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(session_id, None)
        logger.debug(f"[session] Deleted: {session_id}")

    def expire_idle(self) -> int:
        """Remove all idle sessions past TTL.

        Returns:
            Number of sessions expired.
        """
        expired = [
            sid for sid, s in self._sessions.items()
            if s.idle_s > self.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self.delete(sid)
        return len(expired)

    def all_sessions(self) -> list[Session]:
        """Return all active sessions."""
        self.expire_idle()
        return list(self._sessions.values())

    def stats(self) -> dict[str, Any]:
        """Return session statistics."""
        sessions = self.all_sessions()
        running = [s for s in sessions if s.status == "running"]
        return {
            "total_sessions": len(sessions),
            "running": len(running),
            "idle": len(sessions) - len(running),
            "total_tasks": sum(s.task_count for s in sessions),
            "total_tokens": sum(s.total_tokens for s in sessions),
        }


# Global session manager
session_manager = SessionManager()
