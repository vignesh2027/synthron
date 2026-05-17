"""WebSocket streaming routes for real-time agent event feeds."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from synthron.orchestrator.event_bus import event_bus, AgentEvent
from synthron.utils.logger import get_logger

router = APIRouter(tags=["stream"])
logger = get_logger(__name__)


class ConnectionManager:
    """Manage active WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self._connections.setdefault(session_id, []).append(websocket)
        logger.debug(f"[ws] Client connected: session={session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        conns = self._connections.get(session_id, [])
        if websocket in conns:
            conns.remove(websocket)
        logger.debug(f"[ws] Client disconnected: session={session_id}")

    async def broadcast(self, session_id: str, message: dict) -> None:
        """Send a message to all connections for a session."""
        conns = self._connections.get(session_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.remove(ws)

    async def broadcast_all(self, message: dict) -> None:
        """Send to all connected clients."""
        for session_id in list(self._connections.keys()):
            await self.broadcast(session_id, message)


manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time agent event streaming.

    Events are broadcast as JSON objects:
    {
        "type": "thought" | "action" | "result" | "score" | "done",
        "agent": "planner",
        "content": "Breaking task into 4 subtasks...",
        "ts": 1735000000.0
    }
    """
    await manager.connect(websocket, session_id)

    async def forward_event(event: AgentEvent) -> None:
        if event.session_id == session_id or not event.session_id:
            await manager.broadcast(session_id, event.to_dict())

    event_bus.subscribe(forward_event)

    try:
        while True:
            # Keep connection alive, handle pings
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        event_bus.unsubscribe(forward_event)


@router.websocket("/ws/events/all")
async def websocket_all_events(websocket: WebSocket) -> None:
    """WebSocket endpoint that streams ALL agent events (dashboard use)."""
    await websocket.accept()

    async def forward_all(event: AgentEvent) -> None:
        try:
            await websocket.send_json(event.to_dict())
        except Exception:
            pass

    event_bus.subscribe(forward_all)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        event_bus.unsubscribe(forward_all)
