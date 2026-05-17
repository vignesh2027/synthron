"""Planner Agent — decomposes tasks into executable subtask DAGs."""

from __future__ import annotations

import json
import re
from typing import Any

from synthron.agents.base_agent import (
    AgentResult,
    BaseAgent,
    SubTask,
    TaskPlan,
    TaskStatus,
)
from synthron.utils.config import settings
from synthron.utils.exceptions import PlanningError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

_PLANNER_SYSTEM = """You are SYNTHRON's PlannerAgent — a world-class task decomposition specialist.

Your job: receive a user task and break it into a precise, ordered list of subtasks that can be
executed by specialized executor agents using tools (web_search, code_executor, file_tool, api_caller,
data_analyzer, calculator, browser_tool).

RULES:
1. Decompose into 2-8 focused subtasks. Never create unnecessary steps.
2. Each subtask must be independently executable by an agent with tools.
3. Identify dependencies: if subtask B needs subtask A's output, mark depends_on: ["A_id"].
4. Assign a tool_hint: the best tool for each subtask.
5. Assign complexity: integer 1-10 for the full task.
6. Be concrete — no vague "research XYZ" without specifying what to find.

OUTPUT FORMAT (strict JSON, no markdown fences):
{
  "complexity": 7,
  "estimated_time_s": 120,
  "subtasks": [
    {
      "index": 1,
      "title": "Short title",
      "description": "Precise instructions for the executor",
      "tool_hint": "web_search",
      "depends_on": []
    },
    ...
  ]
}

Do NOT output anything except the JSON object."""

_REPLAN_SYSTEM = """You are SYNTHRON's PlannerAgent performing a REPLAN after a subtask failed.

Given the original plan, the failed subtask, and the failure reason, produce a revised plan.
Fix only what is broken. Preserve successful subtasks.
Output the same JSON format as the original plan."""


class PlannerAgent(BaseAgent):
    """Decomposes a complex user task into an ordered DAG of subtasks.

    Uses Gemini 2.5 Flash by default (large context, best planning).
    Falls back via smart router on rate limit.
    """

    name = "planner"
    role = "planner"
    agent_type = "planner"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", _PLANNER_SYSTEM)
        super().__init__(**kwargs)

    def _default_system_prompt(self) -> str:
        return _PLANNER_SYSTEM

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Execute the planner for a given task.

        Args:
            task: Raw user task string.
            context: Optional context (ignored by planner).

        Returns:
            AgentResult with TaskPlan embedded in metadata.
        """
        self._run_count += 1
        self._log.thought(f"Analyzing task: {task[:100]}...")
        await self._emit_event("thought", f"Planning task: {task[:100]}")

        try:
            plan = await self.plan(task)
            self._log.result(
                f"Plan ready: {plan.total_subtasks} subtasks, complexity={plan.complexity}"
            )
            await self._emit_event(
                "plan_ready",
                f"{plan.total_subtasks} subtasks created (complexity={plan.complexity})",
            )
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                task=task,
                output=f"Plan with {plan.total_subtasks} subtasks",
                success=True,
                total_tokens=self._total_tokens,
                total_latency_ms=self._total_latency_ms,
                metadata={"plan": plan.model_dump()},
            )
        except Exception as exc:
            self._log.error(f"Planning failed: {exc}")
            raise PlanningError(self.name, task, str(exc)) from exc

    async def plan(self, task: str) -> TaskPlan:
        """Generate a full TaskPlan for the given task.

        Args:
            task: Raw user task string.

        Returns:
            TaskPlan with ordered subtasks.
        """
        self._log.thought("Decomposing task into subtasks...")

        prompt = f"Task: {task}\n\nCreate the execution plan now."
        response = await self.generate(prompt, max_tokens=2048, temperature=0.3)

        plan = self._parse_plan_response(response.content, task)
        return plan

    async def replan(self, failed_subtask: SubTask, reason: str, original_plan: TaskPlan) -> TaskPlan:
        """Generate a revised plan after a subtask failure.

        Args:
            failed_subtask: The subtask that failed.
            reason: Human-readable failure reason.
            original_plan: The original plan for context.

        Returns:
            Revised TaskPlan with the failed subtask addressed.
        """
        self._log.warning(f"Replanning after failure in subtask '{failed_subtask.title}'")
        await self._emit_event("replan", f"Replanning due to: {reason}")

        original_json = json.dumps(
            [st.model_dump() for st in original_plan.subtasks], indent=2
        )
        prompt = (
            f"Original task: {original_plan.original_task}\n\n"
            f"Original plan:\n{original_json}\n\n"
            f"Failed subtask: '{failed_subtask.title}'\n"
            f"Failure reason: {reason}\n\n"
            f"Produce a revised plan that fixes the failure."
        )

        response = await self.generate(prompt, system=_REPLAN_SYSTEM, max_tokens=2048, temperature=0.2)
        return self._parse_plan_response(response.content, original_plan.original_task)

    async def estimate_complexity(self, task: str) -> int:
        """Estimate task complexity on a 1-10 scale.

        Args:
            task: Task description.

        Returns:
            Integer complexity score 1-10.
        """
        prompt = (
            f"Rate the complexity of this task from 1 (trivial) to 10 (extremely complex).\n"
            f"Task: {task}\n\n"
            f"Reply with only a single integer."
        )
        resp = await self.generate(prompt, max_tokens=10, temperature=0.1)
        digits = re.findall(r"\d+", resp.content)
        if digits:
            return max(1, min(10, int(digits[0])))
        return 5

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Try multiple strategies to extract a JSON object from raw LLM output."""
        # Strategy 1: direct parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: find first { to last }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 3: strip ALL backtick fences line by line then retry
        lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _parse_plan_response(self, raw: str, task: str) -> TaskPlan:
        """Parse the LLM's JSON plan response into a TaskPlan."""
        data = self._extract_json(raw)
        if data is None:
            # Fallback — treat as single subtask instead of crashing
            data = {
                "complexity": 3,
                "estimated_time_s": 30,
                "subtasks": [{"index": 1, "title": "Execute task", "description": task,
                               "tool_hint": "web_search", "depends_on": []}],
            }

        raw_subtasks = data.get("subtasks", [])
        if not raw_subtasks:
            # Fallback: treat the whole task as one subtask
            raw_subtasks = [
                {
                    "index": 1,
                    "title": "Execute task",
                    "description": task,
                    "tool_hint": "web_search",
                    "depends_on": [],
                }
            ]

        subtasks = []
        for i, st_data in enumerate(raw_subtasks):
            subtasks.append(
                SubTask(
                    index=st_data.get("index", i + 1),
                    title=st_data.get("title", f"Subtask {i+1}"),
                    description=st_data.get("description", ""),
                    tool_hint=st_data.get("tool_hint", ""),
                    depends_on=st_data.get("depends_on", []),
                    status=TaskStatus.PENDING,
                )
            )

        complexity = max(1, min(10, int(data.get("complexity", 5))))
        estimated_time = float(data.get("estimated_time_s", len(subtasks) * 30))

        plan = TaskPlan(
            original_task=task,
            subtasks=subtasks,
            complexity=complexity,
            estimated_time_s=estimated_time,
        )

        logger.debug(
            f"[planner] Plan parsed: {plan.total_subtasks} subtasks, "
            f"complexity={plan.complexity}"
        )
        return plan
