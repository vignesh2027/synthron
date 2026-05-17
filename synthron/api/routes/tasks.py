"""Task management API routes."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from synthron.api.models import TaskRequest, TaskResponse
from synthron.orchestrator.orchestrator import Orchestrator
from synthron.utils.logger import get_logger

router = APIRouter(prefix="/tasks", tags=["tasks"])
logger = get_logger(__name__)

# Global orchestrator instance (initialized on first request)
_orchestrator: Orchestrator | None = None
_task_store: dict[str, dict[str, Any]] = {}


async def get_orchestrator() -> Orchestrator:
    """Get or initialize the global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
        await _orchestrator.initialize()
    return _orchestrator


@router.post("/run", response_model=TaskResponse)
async def run_task(request: TaskRequest) -> TaskResponse:
    """Execute a task synchronously and return the result.

    Args:
        request: Task execution request.

    Returns:
        TaskResponse with output and metadata.
    """
    orch = await get_orchestrator()
    task_id = str(uuid.uuid4())[:12]

    logger.info(f"[api/tasks] POST /run task_id={task_id} task={request.task[:60]}")

    try:
        result = await orch.run(
            task=request.task,
            session_id=request.session_id,
        )
        return TaskResponse(
            task_id=task_id,
            session_id=request.session_id,
            status="completed",
            output=result.output,
            success=result.success,
            total_tokens=result.total_tokens,
            total_time_s=result.total_time_s,
            providers_used=result.providers_used,
            retry_count=result.retry_count,
            error=result.error,
        )
    except Exception as exc:
        logger.error(f"[api/tasks] Task failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/async", response_model=dict)
async def run_task_async(
    request: TaskRequest, background_tasks: BackgroundTasks
) -> dict:
    """Queue a task for async execution. Poll /tasks/{task_id} for result."""
    task_id = str(uuid.uuid4())[:12]
    _task_store[task_id] = {"status": "queued", "task": request.task}

    async def run_bg() -> None:
        orch = await get_orchestrator()
        _task_store[task_id]["status"] = "running"
        try:
            result = await orch.run(request.task, session_id=request.session_id)
            _task_store[task_id].update({
                "status": "completed",
                "output": result.output,
                "success": result.success,
                "tokens": result.total_tokens,
                "time_s": result.total_time_s,
            })
        except Exception as exc:
            _task_store[task_id].update({"status": "failed", "error": str(exc)})

    background_tasks.add_task(run_bg)
    return {"task_id": task_id, "status": "queued"}


@router.get("/{task_id}", response_model=dict)
async def get_task_status(task_id: str) -> dict:
    """Get the status of an async task."""
    task = _task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {"task_id": task_id, **task}


@router.get("/stream/{session_id}")
async def stream_task(session_id: str, task: str) -> StreamingResponse:
    """Stream task output as Server-Sent Events."""
    orch = await get_orchestrator()

    async def event_generator():
        async for chunk in orch.stream(task, session_id=session_id):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
