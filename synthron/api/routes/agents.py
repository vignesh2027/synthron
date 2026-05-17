"""Agent management API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from synthron.orchestrator.session_manager import session_manager
from synthron.providers.smart_router import router as smart_router
from synthron.utils.logger import get_logger

router = APIRouter(prefix="/agents", tags=["agents"])
logger = get_logger(__name__)


@router.get("/status")
async def get_agents_status() -> dict:
    """Return status of all active agents and sessions."""
    return {
        "sessions": session_manager.stats(),
        "router": smart_router.status(),
        "provider_stats": smart_router.provider_stats(),
    }


@router.get("/providers")
async def get_providers() -> dict:
    """Return available provider information."""
    from synthron.utils.config import settings
    available = settings.providers.available_providers()
    return {
        "available": available,
        "configured": len(available),
        "router_status": smart_router.status(),
    }


@router.get("/sessions")
async def list_sessions() -> dict:
    """List all active sessions."""
    sessions = session_manager.all_sessions()
    return {
        "total": len(sessions),
        "sessions": [s.to_dict() for s in sessions],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session."""
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    session_manager.delete(session_id)
    return {"deleted": session_id}
