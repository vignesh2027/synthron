"""Stream handler — bridges the EventBus to WebSocket and SSE clients."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

from ..orchestrator.event_bus import EventBus, AgentEvent

logger = logging.getLogger(__name__)


class StreamHandler:
    """
    Subscribes to the EventBus and fans out events to multiple
    WebSocket send callbacks or SSE queues.

    Usage:
        handler = StreamHandler(event_bus)
        handler.add_client(session_id, send_fn)
        # ... later
        handler.remove_client(session_id)
    """

    def __init__(self, event_bus: EventBus):
        self._bus = event_bus
        self._clients: dict[str, Callable] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

        self._bus.subscribe(self._on_event)

    def _on_event(self, event: dict) -> None:
        """Called synchronously by EventBus. Schedules async fan-out."""
        payload = json.dumps(event, default=str)
        for session_id, queue in list(self._queues.items()):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("StreamHandler: queue full for session %s", session_id)

    async def add_sse_client(self, session_id: str) -> asyncio.Queue:
        """Register an SSE client and return its event queue."""
        async with self._lock:
            queue: asyncio.Queue = asyncio.Queue(maxsize=256)
            self._queues[session_id] = queue
        return queue

    async def remove_client(self, session_id: str) -> None:
        async with self._lock:
            self._queues.pop(session_id, None)
            self._clients.pop(session_id, None)

    async def add_ws_client(self, session_id: str, send_fn: Callable) -> None:
        """Register a WebSocket client with a send coroutine."""
        async with self._lock:
            self._clients[session_id] = send_fn
            # Also add an internal queue for buffering
            self._queues[session_id] = asyncio.Queue(maxsize=256)

        # Start drain loop for this client
        asyncio.create_task(self._ws_drain_loop(session_id))

    async def _ws_drain_loop(self, session_id: str) -> None:
        """Drains the queue and sends events to the WebSocket client."""
        queue = self._queues.get(session_id)
        send_fn = self._clients.get(session_id)
        if not queue or not send_fn:
            return

        while session_id in self._clients:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                try:
                    await send_fn(payload)
                except Exception as e:
                    logger.warning("WS send failed for %s: %s", session_id, e)
                    await self.remove_client(session_id)
                    break
            except asyncio.TimeoutError:
                # Send a ping/keepalive
                try:
                    await send_fn(json.dumps({"type": "ping"}))
                except Exception:
                    await self.remove_client(session_id)
                    break

    async def broadcast(self, event: dict) -> None:
        """Manually emit an event to all connected clients."""
        self._bus.emit(event)

    @property
    def client_count(self) -> int:
        return len(self._queues)

    def get_session_ids(self) -> list[str]:
        return list(self._queues.keys())


_stream_handler: StreamHandler | None = None


def get_stream_handler(event_bus: EventBus | None = None) -> StreamHandler:
    """Get or create the global StreamHandler singleton."""
    global _stream_handler
    if _stream_handler is None:
        if event_bus is None:
            from ..orchestrator.event_bus import event_bus as global_bus
            event_bus = global_bus
        _stream_handler = StreamHandler(event_bus)
    return _stream_handler
