"""Abstract base agent and core data models for all Synthron agents."""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable

from pydantic import BaseModel, Field

from synthron.providers.base_provider import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    Message,
)
from synthron.providers.smart_router import router as global_router
from synthron.utils.config import settings
from synthron.utils.exceptions import AgentTimeoutError, MaxRetriesExceededError
from synthron.utils.logger import AgentLogger, get_logger
from synthron.utils.token_counter import count_tokens

logger = get_logger(__name__)


# ─── Data Models ──────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class SubTask(BaseModel):
    """A single decomposed unit of work produced by the PlannerAgent."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    index: int = 0
    title: str
    description: str
    tool_hint: str = ""  # suggested tool to use
    depends_on: list[str] = Field(default_factory=list)  # subtask IDs this depends on
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, higher = sooner

    class Config:
        use_enum_values = True


class TaskPlan(BaseModel):
    """Full execution plan produced by PlannerAgent."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_task: str
    subtasks: list[SubTask]
    complexity: int = Field(default=5, ge=1, le=10)
    estimated_time_s: float = 60.0
    created_at: float = Field(default_factory=time.time)

    @property
    def total_subtasks(self) -> int:
        return len(self.subtasks)

    def get_ready(self, completed_ids: set[str]) -> list[SubTask]:
        """Return subtasks whose dependencies are all satisfied."""
        return [
            st for st in self.subtasks
            if st.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in st.depends_on)
        ]


class SubTaskResult(BaseModel):
    """Result from ExecutorAgent for a single subtask."""

    subtask_id: str
    subtask_title: str
    output: str
    success: bool = True
    tool_used: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str = ""
    attempt: int = 1


class CriticScore(BaseModel):
    """Quality score and feedback from CriticAgent."""

    subtask_id: str
    score: float = Field(ge=0.0, le=1.0)
    verdict: str = ""  # PASS | WARN | FAIL
    feedback: str = ""
    improvement_hint: str = ""
    should_retry: bool = False

    @classmethod
    def from_score(cls, subtask_id: str, score: float, feedback: str = "") -> "CriticScore":
        threshold_pass = settings.agents.critic_pass_threshold
        threshold_warn = settings.agents.critic_warn_threshold
        if score >= threshold_pass:
            verdict = "PASS"
            should_retry = False
        elif score >= threshold_warn:
            verdict = "WARN"
            should_retry = False
        else:
            verdict = "FAIL"
            should_retry = True
        return cls(
            subtask_id=subtask_id,
            score=score,
            verdict=verdict,
            feedback=feedback,
            should_retry=should_retry,
        )


class Thought(BaseModel):
    """An agent's internal reasoning step."""

    content: str
    agent_name: str
    agent_type: str
    timestamp: float = Field(default_factory=time.time)


class AgentResult(BaseModel):
    """Final result from any agent."""

    agent_name: str
    agent_type: str
    task: str
    output: str
    success: bool = True
    subtask_results: list[SubTaskResult] = Field(default_factory=list)
    critic_scores: list[CriticScore] = Field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    provider_used: str = ""
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class FinalResult(BaseModel):
    """Top-level result returned to the user after full orchestration."""

    task: str
    output: str
    success: bool = True
    agent_results: list[AgentResult] = Field(default_factory=list)
    total_tokens: int = 0
    total_time_s: float = 0.0
    providers_used: list[str] = Field(default_factory=list)
    retry_count: int = 0
    error: str = ""

    def __str__(self) -> str:
        return self.output


# ─── Abstract Base Agent ──────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base class for all Synthron agents.

    All agents follow the Think → Act → Reflect loop:
    1. think()   — reason about what to do next
    2. act()     — execute a tool or generate text
    3. reflect() — evaluate the result and decide next step

    Agents are async-first; use `await agent.run(task)` to invoke.
    """

    name: str = "base_agent"
    role: str = "base"
    agent_type: str = "default"

    def __init__(
        self,
        name: str = "",
        provider: BaseProvider | None = None,
        tools: list[Any] | None = None,
        system_prompt: str = "",
        max_retries: int | None = None,
        timeout: float | None = None,
        stream_callbacks: list[Callable[[dict], None]] | None = None,
    ) -> None:
        if name:
            self.name = name

        self._provider = provider  # None → use smart router
        self._tools: list[Any] = tools or []
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_retries = max_retries if max_retries is not None else settings.agents.max_retries
        self._timeout = timeout or settings.agents.agent_timeout
        self._stream_callbacks = stream_callbacks or []

        self._log = AgentLogger(self.name, self.agent_type)
        self._message_history: list[Message] = []
        self._total_tokens = 0
        self._total_latency_ms = 0.0
        self._run_count = 0

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Execute the agent's primary function for the given task.

        Args:
            task: The task description or input.
            context: Optional shared context dict from the orchestrator.

        Returns:
            AgentResult with output and metadata.
        """

    def _default_system_prompt(self) -> str:
        """Return the default system prompt for this agent type. Override per agent."""
        return (
            "You are a helpful AI agent. Think step by step. "
            "Be concise, accurate, and complete."
        )

    # ── Core loop helpers ──────────────────────────────────────────────────────

    async def think(self, context: str) -> Thought:
        """Generate an internal reasoning step.

        Args:
            context: Current task context for the thought.

        Returns:
            Thought object with reasoning content.
        """
        thought_text = f"Thinking about: {context[:200]}"
        self._log.thought(thought_text)
        thought = Thought(
            content=thought_text,
            agent_name=self.name,
            agent_type=self.agent_type,
        )
        await self._emit_event("thought", thought_text)
        return thought

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResponse:
        """Call the LLM provider (via router or direct provider).

        Args:
            prompt: The user-side prompt text.
            system: Optional system prompt override.
            temperature: Temperature override; uses agent default if None.
            max_tokens: Maximum output tokens.

        Returns:
            GenerationResponse with content and usage stats.
        """
        messages = list(self._message_history) + [Message(role="user", content=prompt)]
        request = GenerationRequest(
            messages=messages,
            system_prompt=system or self._system_prompt,
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens,
        )

        try:
            if self._provider is not None:
                response = await asyncio.wait_for(
                    self._provider.generate(request),
                    timeout=self._timeout,
                )
            else:
                response = await asyncio.wait_for(
                    global_router.generate(request, agent_type=self.agent_type),
                    timeout=self._timeout,
                )

            self._total_tokens += response.total_tokens
            self._total_latency_ms += response.latency_ms

            # Maintain rolling message history
            self._message_history.append(Message(role="user", content=prompt))
            self._message_history.append(
                Message(role="assistant", content=response.content)
            )
            # Trim history to last 20 messages
            if len(self._message_history) > 20:
                self._message_history = self._message_history[-20:]

            return response

        except asyncio.TimeoutError:
            raise AgentTimeoutError(self.name, self._timeout)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream tokens from the LLM.

        Args:
            prompt: The prompt text.

        Yields:
            Token chunks as strings.
        """
        messages = [Message(role="user", content=prompt)]
        request = GenerationRequest(
            messages=messages,
            system_prompt=self._system_prompt,
        )

        if self._provider is not None:
            async for chunk in self._provider.generate_stream(request):
                yield chunk
        else:
            async for chunk in global_router.generate_stream(
                request, agent_type=self.agent_type
            ):
                yield chunk

    def add_tool(self, tool: Any) -> None:
        """Register a tool with this agent."""
        self._tools.append(tool)

    def get_tool(self, name: str) -> Any | None:
        """Retrieve a tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def clear_history(self) -> None:
        """Clear message history (start fresh conversation)."""
        self._message_history.clear()

    async def _emit_event(self, event_type: str, content: str) -> None:
        """Emit a streaming event to all registered callbacks."""
        event = {
            "agent": self.name,
            "agent_type": self.agent_type,
            "event_type": event_type,
            "content": content,
            "timestamp": time.time(),
        }
        for cb in self._stream_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        """Register a streaming event callback."""
        self._stream_callbacks.append(callback)

    @property
    def stats(self) -> dict[str, Any]:
        """Return cumulative agent statistics."""
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "runs": self._run_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": round(self._total_latency_ms, 1),
            "avg_latency_ms": (
                round(self._total_latency_ms / self._run_count, 1)
                if self._run_count else 0
            ),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
