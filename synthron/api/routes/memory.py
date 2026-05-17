"""Memory inspection API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from synthron.api.models import MemoryQueryRequest, MemoryQueryResponse
from synthron.memory.memory_manager import MemoryManager
from synthron.utils.logger import get_logger

router = APIRouter(prefix="/memory", tags=["memory"])
logger = get_logger(__name__)

_memory: MemoryManager | None = None


async def get_memory() -> MemoryManager:
    global _memory
    if _memory is None:
        _memory = MemoryManager()
        await _memory.initialize()
    return _memory


@router.post("/recall", response_model=MemoryQueryResponse)
async def recall_memories(request: MemoryQueryRequest) -> MemoryQueryResponse:
    """Semantic search across long-term memory."""
    mem = await get_memory()
    results = await mem.recall(request.query, top_k=request.top_k)
    return MemoryQueryResponse(
        query=request.query,
        results=results,
        total=len(results),
    )


@router.get("/stats")
async def get_memory_stats() -> dict:
    """Return memory system statistics."""
    mem = await get_memory()
    return await mem.memory_stats()


@router.get("/episodes/recent")
async def get_recent_episodes(limit: int = 10) -> dict:
    """Return recent task episodes."""
    mem = await get_memory()
    episodes = await mem._episodic.get_recent(limit=limit)
    return {"episodes": episodes, "total": len(episodes)}


@router.get("/patterns")
async def get_learned_patterns() -> dict:
    """Return learned behavioral patterns."""
    mem = await get_memory()
    patterns = await mem.get_patterns()
    return {"patterns": patterns, "total": len(patterns)}
