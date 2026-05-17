"""Executor Agent — runs subtasks using tools, powered by Cerebras for speed."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from synthron.agents.base_agent import (
    AgentResult,
    BaseAgent,
    SubTask,
    SubTaskResult,
    TaskStatus,
)
from synthron.utils.config import settings
from synthron.utils.exceptions import ExecutionError, ToolError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

_EXECUTOR_SYSTEM = """You are SYNTHRON's ExecutorAgent — a precise, tool-wielding specialist.

You receive ONE subtask and must complete it using the available tools.

PROCESS:
1. Read the subtask description carefully.
2. Select the best tool for the job.
3. Execute with the tool.
4. Return a clear, complete result.

TOOLS AVAILABLE: {tool_names}

OUTPUT FORMAT:
- Start with the direct result/answer.
- Include all relevant data, numbers, or content found.
- Be specific. If the task asks for numbers, provide them.
- Do NOT say "I cannot" — always attempt the task.

If no specific tool is needed (simple reasoning/writing), just produce the output directly."""


class ExecutorAgent(BaseAgent):
    """Executes individual subtasks using registered tools.

    Assigned to Cerebras (2100 tok/s) for maximum throughput.
    Supports parallel execution of independent subtasks.
    """

    name = "executor"
    role = "executor"
    agent_type = "executor"

    def __init__(self, executor_id: int = 1, **kwargs: Any) -> None:
        name = f"executor-{executor_id}"
        kwargs.setdefault("name", name)
        super().__init__(**kwargs)
        self.executor_id = executor_id

    def _default_system_prompt(self) -> str:
        tool_names = ", ".join(t.name for t in self._tools) if self._tools else "none"
        return _EXECUTOR_SYSTEM.format(tool_names=tool_names)

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Execute a task string directly (used when called as standalone agent).

        Args:
            task: Task description string.
            context: Optional context dict.

        Returns:
            AgentResult with output.
        """
        self._run_count += 1
        subtask = SubTask(
            title="Direct execution",
            description=task,
            tool_hint="",
        )
        result = await self.execute(subtask, context=context)

        return AgentResult(
            agent_name=self.name,
            agent_type=self.agent_type,
            task=task,
            output=result.output,
            success=result.success,
            subtask_results=[result],
            total_tokens=self._total_tokens,
            total_latency_ms=self._total_latency_ms,
            error=result.error,
        )

    async def execute(
        self,
        subtask: SubTask,
        context: dict[str, Any] | None = None,
        attempt: int = 1,
    ) -> SubTaskResult:
        """Execute a single subtask, using tools as needed.

        Args:
            subtask: The SubTask to execute.
            context: Shared context from orchestrator (previous results, etc.)
            attempt: Retry attempt number for logging.

        Returns:
            SubTaskResult with output and metadata.
        """
        start_ts = time.perf_counter()
        self._log.thought(f"Executing: '{subtask.title}'")
        await self._emit_event("executing", f"Subtask: {subtask.title}")

        subtask.status = TaskStatus.RUNNING

        # Build execution prompt
        context_str = self._format_context(context)
        prompt = self._build_prompt(subtask, context_str)

        # Try tool first if a hint is given
        tool_result: str | None = None
        tool_used = ""
        if subtask.tool_hint:
            tool = self.get_tool(subtask.tool_hint)
            if tool:
                try:
                    self._log.action(tool.name, subtask.description[:80])
                    tool_result = await asyncio.wait_for(
                        tool.run(subtask.description, context=context),
                        timeout=settings.agents.tool_timeout,
                    )
                    tool_used = tool.name
                    self._log.result(f"Tool '{tool.name}' returned {len(str(tool_result))} chars")
                except asyncio.TimeoutError:
                    self._log.warning(f"Tool '{subtask.tool_hint}' timed out — falling back to LLM")
                except ToolError as exc:
                    self._log.warning(f"Tool '{subtask.tool_hint}' error: {exc} — using LLM fallback")
                except Exception as exc:
                    self._log.warning(f"Unexpected tool error: {exc} — using LLM fallback")

        # Augment prompt with tool result if available
        if tool_result:
            final_prompt = (
                f"{prompt}\n\n"
                f"[TOOL RESULT from {tool_used}]:\n{tool_result}\n\n"
                f"Now synthesize this into a complete, structured answer for the subtask."
            )
        else:
            # Try all tools if no specific hint worked
            if not tool_used and self._tools:
                best_tool = await self.select_tool(subtask)
                if best_tool:
                    try:
                        tool_result = await asyncio.wait_for(
                            best_tool.run(subtask.description, context=context),
                            timeout=settings.agents.tool_timeout,
                        )
                        tool_used = best_tool.name
                    except Exception:
                        pass

            if tool_result:
                final_prompt = (
                    f"{prompt}\n\n"
                    f"[TOOL RESULT from {tool_used}]:\n{tool_result}\n\n"
                    f"Synthesize this into a complete answer."
                )
            else:
                final_prompt = prompt

        try:
            response = await self.generate(final_prompt, max_tokens=4096, temperature=0.5)
            elapsed_ms = (time.perf_counter() - start_ts) * 1000

            subtask.status = TaskStatus.COMPLETED
            self._log.result(f"'{subtask.title}' complete ({elapsed_ms:.0f}ms)")
            await self._emit_event("subtask_done", f"Completed: {subtask.title}")

            return SubTaskResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output=response.content,
                success=True,
                tool_used=tool_used,
                tokens_used=response.total_tokens,
                latency_ms=elapsed_ms,
                attempt=attempt,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000
            subtask.status = TaskStatus.FAILED
            self._log.error(f"Execution failed for '{subtask.title}': {exc}")
            await self._emit_event("subtask_failed", f"Failed: {subtask.title} — {exc}")

            return SubTaskResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output="",
                success=False,
                tool_used=tool_used,
                tokens_used=0,
                latency_ms=elapsed_ms,
                error=str(exc),
                attempt=attempt,
            )

    async def select_tool(self, subtask: SubTask) -> Any | None:
        """Ask the LLM which tool best fits this subtask.

        Args:
            subtask: The subtask to select a tool for.

        Returns:
            The best matching tool, or None.
        """
        if not self._tools:
            return None

        tool_names = [t.name for t in self._tools]
        prompt = (
            f"Given this subtask: '{subtask.description}'\n"
            f"Which tool is most appropriate? Options: {', '.join(tool_names)}\n"
            f"Reply with ONLY the tool name, nothing else."
        )

        try:
            response = await self.generate(prompt, max_tokens=20, temperature=0.1)
            selected = response.content.strip().lower()
            tool = self.get_tool(selected)
            if tool:
                return tool
            # Fuzzy match
            for t in self._tools:
                if t.name in selected or selected in t.name:
                    return t
        except Exception:
            pass

        return None

    async def handle_tool_error(self, error: Exception, subtask: SubTask) -> SubTaskResult:
        """Handle a tool error by falling back to pure LLM generation.

        Args:
            error: The exception from the tool.
            subtask: The subtask that was being executed.

        Returns:
            SubTaskResult from LLM fallback.
        """
        self._log.warning(f"Tool error, using LLM fallback: {error}")
        prompt = (
            f"A tool failed with error: {error}\n"
            f"Complete this subtask using your own knowledge instead:\n{subtask.description}"
        )
        try:
            response = await self.generate(prompt, max_tokens=2048)
            return SubTaskResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output=response.content,
                success=True,
                tool_used="llm_fallback",
                tokens_used=response.total_tokens,
            )
        except Exception as fallback_exc:
            return SubTaskResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output="",
                success=False,
                error=f"Tool error: {error}. LLM fallback also failed: {fallback_exc}",
            )

    def _build_prompt(self, subtask: SubTask, context_str: str) -> str:
        """Build the execution prompt for a subtask."""
        tool_names = ", ".join(t.name for t in self._tools) or "none"
        parts = [
            f"SUBTASK #{subtask.index}: {subtask.title}",
            f"DESCRIPTION: {subtask.description}",
        ]
        if context_str:
            parts.append(f"CONTEXT FROM PREVIOUS STEPS:\n{context_str}")
        if subtask.tool_hint:
            parts.append(f"SUGGESTED TOOL: {subtask.tool_hint}")
        parts.append(f"AVAILABLE TOOLS: {tool_names}")
        parts.append("\nProvide a complete, detailed result:")
        return "\n".join(parts)

    def _format_context(self, context: dict[str, Any] | None) -> str:
        """Format the context dict into a readable string."""
        if not context:
            return ""
        lines = []
        for key, value in list(context.items())[:5]:  # limit context size
            if isinstance(value, str) and len(value) > 500:
                value = value[:500] + "..."
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
