"""Event bus — pub/sub agent communication and dashboard streaming."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class AgentEvent:
    """A Synthron agent event broadcast on the event bus."""

    event_type: str               # thought | action | result | score | error | plan | done
    agent_name: str
    agent_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.event_type,
            "agent": self.agent_name,
            "agent_type": self.agent_type,
            "content": self.content,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "ts": self.timestamp,
        }


# Callback type: sync or async
EventCallback = Callable[[AgentEvent], Any]


class EventBus:
    """Async pub/sub event bus for inter-agent communication.

    Agents publish events (thoughts, tool calls, results).
    Dashboard, logger, and self-improvement system subscribe.
    Supports both sync and async callbacks.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._global_subscribers: list[EventCallback] = []
        self._history: list[AgentEvent] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()

    def subscribe(self, callback: EventCallback, event_types: list[str] | None = None) -> None:
        """Subscribe to events.

        Args:
            callback: Sync or async function called with each AgentEvent.
            event_types: List of event types to filter, or None for all events.
        """
        if event_types is None:
            self._global_subscribers.append(callback)
        else:
            for et in event_types:
                self._subscribers.setdefault(et, []).append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        """Unsubscribe a callback from all event types."""
        if callback in self._global_subscribers:
            self._global_subscribers.remove(callback)
        for callbacks in self._subscribers.values():
            if callback in callbacks:
                callbacks.remove(callback)

    async def publish(self, event: AgentEvent) -> None:
        """Publish an event to all relevant subscribers.

        Args:
            event: The AgentEvent to broadcast.
        """
        async with self._lock:
            # Store in history
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Notify subscribers (don't block on slow subscribers)
        callbacks = list(self._global_subscribers) + self._subscribers.get(event.event_type, [])
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(event))
                else:
                    cb(event)
            except Exception:
                pass

    async def emit(
        self,
        event_type: str,
        agent_name: str,
        agent_type: str,
        content: str,
        session_id: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Convenience method to create and publish an event.

        Args:
            event_type: Event category string.
            agent_name: Name of the emitting agent.
            agent_type: Type of the emitting agent.
            content: Event message content.
            session_id: Session identifier.
            metadata: Optional additional metadata.
        """
        event = AgentEvent(
            event_type=event_type,
            agent_name=agent_name,
            agent_type=agent_type,
            content=content,
            session_id=session_id,
            metadata=metadata or {},
        )
        await self.publish(event)

    def get_history(
        self,
        session_id: str = "",
        event_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[AgentEvent]:
        """Return recent event history, optionally filtered.

        Args:
            session_id: Filter by session, or empty for all sessions.
            event_types: Filter by event type, or None for all.
            limit: Maximum number of events to return.

        Returns:
            List of AgentEvent objects.
        """
        events = self._history
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        return events[-limit:]

    def clear_history(self, session_id: str = "") -> None:
        """Clear event history for a session or all sessions."""
        if session_id:
            self._history = [e for e in self._history if e.session_id != session_id]
        else:
            self._history.clear()

    def stats(self) -> dict:
        """Return bus statistics."""
        from collections import Counter
        type_counts = Counter(e.event_type for e in self._history)
        return {
            "total_events": len(self._history),
            "global_subscribers": len(self._global_subscribers),
            "typed_subscribers": {k: len(v) for k, v in self._subscribers.items()},
            "event_type_counts": dict(type_counts),
        }


# Global singleton event bus
event_bus = EventBus()
