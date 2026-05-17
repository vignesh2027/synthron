"""Pydantic schemas for the Synthron REST API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=10000, description="Task to execute")
    session_id: str = Field(default="", description="Optional session ID for context continuity")
    stream: bool = Field(default=False, description="Stream responses via WebSocket")
    tools: list[str] = Field(default_factory=list, description="Tool names to enable (empty = all)")
    provider: str = Field(default="", description="Force a specific provider (optional)")


class TaskResponse(BaseModel):
    task_id: str
    session_id: str
    status: str = "queued"
    output: str = ""
    success: bool = True
    total_tokens: int = 0
    total_time_s: float = 0.0
    providers_used: list[str] = []
    retry_count: int = 0
    error: str = ""


class AgentStatusResponse(BaseModel):
    name: str
    agent_type: str
    status: str
    runs: int
    total_tokens: int
    avg_latency_ms: float


class MemoryQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str = ""


class MemoryQueryResponse(BaseModel):
    query: str
    results: list[dict[str, Any]] = []
    total: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    providers: list[str] = []
    memory_stats: dict[str, Any] = {}
    performance: dict[str, Any] = {}


class StreamEvent(BaseModel):
    type: str
    agent: str
    agent_type: str = ""
    content: str
    session_id: str = ""
    ts: float = 0.0
