"""
Synthron — The Neural Fabric for Autonomous AI Agents

Quickstart:
    from synthron import run
    result = await run("Research the top AI companies in 2026")
    print(result)

Advanced:
    from synthron import Synthron
    agent = Synthron(providers=["gemini", "groq"], tools=["web_search", "code_executor"])
    result = await agent.run("Build a Python web scraper for Hacker News")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Synthron Contributors"

from synthron.orchestrator.orchestrator import Orchestrator
from synthron.agents.base_agent import (
    FinalResult,
    TaskPlan,
    SubTask,
    SubTaskResult,
    CriticScore,
)
from synthron.providers.smart_router import router
from synthron.utils.config import settings
from synthron.utils.logger import get_logger

_default_orchestrator: Orchestrator | None = None


async def run(
    task: str,
    session_id: str = "",
    providers: list[str] | None = None,
    tools: list[str] | None = None,
) -> str:
    """Run a task with Synthron in one line.

    This is the simplest way to use Synthron. It initializes automatically
    on first call and reuses the same orchestrator for subsequent calls.

    Args:
        task: Natural language task description.
        session_id: Optional session ID for context continuity.
        providers: List of provider names to use (default: all configured).
        tools: List of tool names to enable (default: all).

    Returns:
        Task output as a string.

    Example:
        result = await run("Summarize recent AI news")
        print(result)
    """
    global _default_orchestrator

    if _default_orchestrator is None:
        tool_instances = None
        if tools:
            from synthron.tools import get_tools_by_names
            tool_instances = get_tools_by_names(tools)
        _default_orchestrator = Orchestrator(tools=tool_instances)

    result = await _default_orchestrator.run(task, session_id=session_id)
    return result.output or result.error


class Synthron:
    """High-level Synthron client for developer use.

    Usage:
        agent = Synthron(providers=["gemini", "groq"], tools=["web_search"])
        result = await agent.run("Your task here")
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        tools: list[str] | None = None,
        max_executors: int = 3,
        critic_threshold: float = 0.8,
        dashboard: bool = False,
    ) -> None:
        tool_instances = None
        if tools:
            from synthron.tools import get_tools_by_names
            tool_instances = get_tools_by_names(tools)
        else:
            from synthron.tools import get_default_tools
            tool_instances = get_default_tools()

        self._orchestrator = Orchestrator(
            tools=tool_instances,
            max_executors=max_executors,
            critic_threshold=critic_threshold,
            dashboard=dashboard,
        )
        self._initialized = False

    async def run(self, task: str, session_id: str = "", stream: bool = False) -> FinalResult:
        """Run a task through the full Synthron pipeline.

        Args:
            task: Task to execute.
            session_id: Optional session ID.
            stream: If True, stream events to callbacks.

        Returns:
            FinalResult with output and metadata.
        """
        if not self._initialized:
            await self._orchestrator.initialize()
            self._initialized = True
        return await self._orchestrator.run(task, session_id=session_id, stream=stream)

    def subscribe(self, callback) -> None:
        """Subscribe to real-time agent events."""
        self._orchestrator.subscribe(callback)

    def status(self) -> dict:
        """Return orchestrator status."""
        return self._orchestrator.status()

    async def __aenter__(self) -> "Synthron":
        if not self._initialized:
            await self._orchestrator.initialize()
            self._initialized = True
        return self

    async def __aexit__(self, *args) -> None:
        pass


__all__ = [
    "__version__",
    "run",
    "Synthron",
    "Orchestrator",
    "FinalResult",
    "TaskPlan",
    "SubTask",
    "SubTaskResult",
    "CriticScore",
    "settings",
    "router",
]
