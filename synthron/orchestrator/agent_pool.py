"""Agent pool — dynamic agent spawning and lifecycle management."""

from __future__ import annotations

import asyncio
from typing import Any

from synthron.agents.base_agent import BaseAgent
from synthron.agents.critic_agent import CriticAgent
from synthron.agents.executor_agent import ExecutorAgent
from synthron.agents.memory_agent import MemoryAgent
from synthron.agents.planner_agent import PlannerAgent
from synthron.agents.researcher_agent import ResearcherAgent
from synthron.agents.coder_agent import CoderAgent
from synthron.tools import get_default_tools
from synthron.utils.config import settings
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class AgentPool:
    """Manages a pool of reusable agents with dynamic spawning.

    Agents are created lazily and reused across tasks.
    Executor agents can be scaled up to max_executors for parallel work.
    """

    def __init__(
        self,
        max_executors: int | None = None,
        stream_callbacks: list | None = None,
        tools: list | None = None,
    ) -> None:
        self.max_executors = max_executors or settings.agents.max_parallel_executors
        self._stream_callbacks = stream_callbacks or []
        self._tools = tools or get_default_tools()

        # Singleton agents
        self._planner: PlannerAgent | None = None
        self._critic: CriticAgent | None = None
        self._memory_agent: MemoryAgent | None = None
        self._researcher: ResearcherAgent | None = None
        self._coder: CoderAgent | None = None

        # Executor pool (up to max_executors)
        self._executors: list[ExecutorAgent] = []
        self._executor_lock = asyncio.Lock()

    def _common_kwargs(self) -> dict[str, Any]:
        return {"stream_callbacks": list(self._stream_callbacks)}

    @property
    def planner(self) -> PlannerAgent:
        """Return (or create) the singleton PlannerAgent."""
        if self._planner is None:
            self._planner = PlannerAgent(**self._common_kwargs())
            logger.debug("[pool] PlannerAgent created")
        return self._planner

    @property
    def critic(self) -> CriticAgent:
        """Return (or create) the singleton CriticAgent."""
        if self._critic is None:
            self._critic = CriticAgent(**self._common_kwargs())
            logger.debug("[pool] CriticAgent created")
        return self._critic

    @property
    def memory_agent(self) -> MemoryAgent:
        """Return (or create) the singleton MemoryAgent."""
        if self._memory_agent is None:
            self._memory_agent = MemoryAgent(**self._common_kwargs())
            logger.debug("[pool] MemoryAgent created")
        return self._memory_agent

    @property
    def researcher(self) -> ResearcherAgent:
        """Return (or create) the singleton ResearcherAgent."""
        if self._researcher is None:
            self._researcher = ResearcherAgent(
                tools=self._tools, **self._common_kwargs()
            )
            logger.debug("[pool] ResearcherAgent created")
        return self._researcher

    @property
    def coder(self) -> CoderAgent:
        """Return (or create) the singleton CoderAgent."""
        if self._coder is None:
            self._coder = CoderAgent(
                tools=self._tools, **self._common_kwargs()
            )
            logger.debug("[pool] CoderAgent created")
        return self._coder

    async def get_executor(self) -> ExecutorAgent:
        """Return an available executor agent, spawning one if needed.

        Returns:
            An ExecutorAgent ready to process a subtask.
        """
        async with self._executor_lock:
            # Find a free executor (not currently running)
            for executor in self._executors:
                if executor._run_count == 0 or True:  # simple: all are reusable
                    return executor

            # Spawn new executor if pool not full
            if len(self._executors) < self.max_executors:
                executor_id = len(self._executors) + 1
                executor = ExecutorAgent(
                    executor_id=executor_id,
                    tools=self._tools,
                    **self._common_kwargs(),
                )
                self._executors.append(executor)
                logger.debug(f"[pool] Spawned ExecutorAgent #{executor_id}")
                return executor

            # Pool full — return the least-used executor
            return min(self._executors, key=lambda e: e._total_tokens)

    def spawn_executor(self) -> ExecutorAgent:
        """Synchronously spawn a new ExecutorAgent."""
        executor_id = len(self._executors) + 1
        executor = ExecutorAgent(
            executor_id=executor_id,
            tools=self._tools,
            **self._common_kwargs(),
        )
        self._executors.append(executor)
        return executor

    def add_stream_callback(self, callback: Any) -> None:
        """Add a streaming callback to all pooled agents."""
        self._stream_callbacks.append(callback)
        for agent in self._all_agents():
            if hasattr(agent, "subscribe"):
                agent.subscribe(callback)

    def _all_agents(self) -> list[BaseAgent]:
        """Return all initialized agents."""
        agents: list[BaseAgent] = list(self._executors)
        for a in [self._planner, self._critic, self._memory_agent, self._researcher, self._coder]:
            if a is not None:
                agents.append(a)
        return agents

    def stats(self) -> dict[str, Any]:
        """Return pool statistics."""
        return {
            "executors": len(self._executors),
            "max_executors": self.max_executors,
            "tools_registered": len(self._tools),
            "agents": {
                agent.name: agent.stats for agent in self._all_agents()
            },
        }
