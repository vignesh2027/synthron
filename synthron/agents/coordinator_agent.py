"""Coordinator Agent — orchestrates multi-agent teams for complex tasks."""

from __future__ import annotations

import asyncio
from typing import Any

from synthron.agents.base_agent import AgentResult, BaseAgent, FinalResult
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

_COORDINATOR_SYSTEM = """You are SYNTHRON's CoordinatorAgent — the master orchestrator of multi-agent teams.

Your role: Assign the right specialized agents to the right subtasks, monitor progress,
resolve conflicts, and synthesize all results into a unified final answer.

COORDINATION PRINCIPLES:
1. Assign tasks based on agent specialization (researcher → research, coder → code).
2. Run independent tasks in parallel for maximum speed.
3. Pass results from one agent as context to dependent agents.
4. Detect and resolve conflicts between agent outputs.
5. Synthesize everything into a cohesive, well-structured final response.

SYNTHESIS: When merging results, create a unified narrative. Don't just concatenate.
Ensure consistency, fill gaps, and highlight the most important findings."""


class CoordinatorAgent(BaseAgent):
    """Coordinates a team of specialized agents for complex multi-faceted tasks.

    Manages parallel execution, result passing, and final synthesis.
    Powered by Gemini (best coordination and synthesis).
    """

    name = "coordinator"
    role = "coordinator"
    agent_type = "coordinator"

    def __init__(
        self,
        sub_agents: dict[str, BaseAgent] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("system_prompt", _COORDINATOR_SYSTEM)
        super().__init__(**kwargs)
        self._sub_agents: dict[str, BaseAgent] = sub_agents or {}

    def _default_system_prompt(self) -> str:
        return _COORDINATOR_SYSTEM

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a specialized agent with the coordinator."""
        self._sub_agents[name] = agent
        logger.debug(f"[coordinator] Registered agent: {name}")

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Coordinate all sub-agents to complete a complex task.

        Args:
            task: The high-level task.
            context: Optional shared context.

        Returns:
            AgentResult with synthesized output from all agents.
        """
        self._run_count += 1
        self._log.thought(f"Coordinating task: {task[:80]}")
        await self._emit_event("coordinating", f"Starting coordination: {task[:80]}")

        # Determine which agents to use
        agent_assignments = await self._plan_assignments(task)

        # Execute agents in dependency order (parallel where possible)
        all_results: dict[str, AgentResult] = {}
        total_tokens = 0

        for agent_names in agent_assignments:
            # Run independent agents in parallel
            tasks = []
            for agent_name in agent_names:
                if agent_name in self._sub_agents:
                    agent = self._sub_agents[agent_name]
                    # Pass results from completed agents as context
                    ctx = {
                        k: v.output[:500] for k, v in all_results.items()
                    }
                    tasks.append(self._run_agent(agent, task, ctx))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for agent_name, result in zip(agent_names, results):
                    if isinstance(result, Exception):
                        self._log.warning(f"Agent '{agent_name}' failed: {result}")
                    elif isinstance(result, AgentResult):
                        all_results[agent_name] = result
                        total_tokens += result.total_tokens

        # Synthesize all results
        final_output = await self._synthesize(task, all_results)
        await self._emit_event("synthesis_done", "All agents completed, results synthesized")

        return AgentResult(
            agent_name=self.name,
            agent_type=self.agent_type,
            task=task,
            output=final_output,
            success=True,
            total_tokens=total_tokens + self._total_tokens,
            total_latency_ms=self._total_latency_ms,
            metadata={
                "agents_used": list(all_results.keys()),
                "agent_results": {k: v.output[:200] for k, v in all_results.items()},
            },
        )

    async def _run_agent(
        self, agent: BaseAgent, task: str, context: dict[str, Any]
    ) -> AgentResult:
        """Run a sub-agent with error handling."""
        try:
            self._log.action(f"delegate → {agent.name}", task[:60])
            return await agent.run(task, context=context)
        except Exception as exc:
            self._log.error(f"Sub-agent '{agent.name}' failed: {exc}")
            return AgentResult(
                agent_name=agent.name,
                agent_type=agent.agent_type,
                task=task,
                output="",
                success=False,
                error=str(exc),
            )

    async def _plan_assignments(self, task: str) -> list[list[str]]:
        """Determine which agents to use and their execution order.

        Returns a list of batches, where each batch runs in parallel.

        Args:
            task: Task description.

        Returns:
            List of batches (each batch is a list of agent names to run in parallel).
        """
        available = list(self._sub_agents.keys())
        if not available:
            return []

        prompt = (
            f"Task: {task}\n\n"
            f"Available agents: {', '.join(available)}\n\n"
            f"Which agents should handle this task? List them in execution order. "
            f"Agents on the same line run in parallel. Separate parallel groups with |.\n"
            f"Example: researcher | coder\ncoder\n\n"
            f"Reply with agent names only, one group per line."
        )

        try:
            response = await self.generate(prompt, max_tokens=200, temperature=0.3)
            lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
            batches = []
            for line in lines:
                batch = [a.strip() for a in line.split("|") if a.strip() in available]
                if batch:
                    batches.append(batch)
            return batches if batches else [[available[0]]]
        except Exception:
            # Fallback: use all available agents sequentially
            return [[name] for name in available]

    async def _synthesize(self, task: str, results: dict[str, AgentResult]) -> str:
        """Synthesize all agent outputs into a unified response.

        Args:
            task: Original task.
            results: Dict of agent_name → AgentResult.

        Returns:
            Synthesized final output string.
        """
        if not results:
            return "No results to synthesize."

        results_text = "\n\n".join(
            f"## {name.upper()} AGENT RESULT:\n{r.output[:2000]}"
            for name, r in results.items()
            if r.output
        )

        prompt = (
            f"Original task: {task}\n\n"
            f"Results from specialized agents:\n{results_text}\n\n"
            f"Synthesize these into one comprehensive, cohesive response. "
            f"Integrate all information seamlessly. Be thorough but organized."
        )

        try:
            response = await self.generate(prompt, max_tokens=4096, temperature=0.5)
            return response.content
        except Exception as exc:
            self._log.error(f"Synthesis failed: {exc}")
            # Fallback: concatenate results
            return "\n\n".join(
                f"### {name}\n{r.output}" for name, r in results.items() if r.output
            )
