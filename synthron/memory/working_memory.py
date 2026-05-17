"""Working memory — active task context window for the current run."""

from __future__ import annotations

import time
from typing import Any

from synthron.agents.base_agent import SubTaskResult


class WorkingMemory:
    """Active in-memory store for the current task execution.

    Holds: task description, partial results, shared context,
    agent outputs, and intermediate data between steps.

    This is ephemeral — it exists only for the duration of one orchestrator run.
    """

    def __init__(self, task: str, session_id: str = "") -> None:
        self.task = task
        self.session_id = session_id
        self.created_at = time.time()

        self._context: dict[str, Any] = {}
        self._results: dict[str, SubTaskResult] = {}  # subtask_id → result
        self._agent_outputs: dict[str, str] = {}       # agent_name → latest output
        self._shared_data: dict[str, Any] = {}         # arbitrary shared key-value store
        self._notes: list[str] = []                    # free-form notes from any agent

    # ── Context management ────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)

    def update(self, data: dict[str, Any]) -> None:
        """Bulk update context."""
        self._context.update(data)

    def context_snapshot(self) -> dict[str, Any]:
        """Return a copy of the current context."""
        return dict(self._context)

    # ── Subtask results ───────────────────────────────────────────────────────

    def store_result(self, result: SubTaskResult) -> None:
        """Store a subtask result."""
        self._results[result.subtask_id] = result
        self._agent_outputs[f"subtask_{result.subtask_title}"] = result.output[:500]

    def get_result(self, subtask_id: str) -> SubTaskResult | None:
        """Retrieve a subtask result by ID."""
        return self._results.get(subtask_id)

    def all_results(self) -> list[SubTaskResult]:
        """Return all stored subtask results."""
        return list(self._results.values())

    def completed_ids(self) -> set[str]:
        """Return IDs of all completed subtasks."""
        return {r.subtask_id for r in self._results.values() if r.success}

    # ── Agent outputs ─────────────────────────────────────────────────────────

    def set_agent_output(self, agent_name: str, output: str) -> None:
        """Store latest output from a named agent."""
        self._agent_outputs[agent_name] = output

    def get_agent_output(self, agent_name: str) -> str:
        """Retrieve latest output from a named agent."""
        return self._agent_outputs.get(agent_name, "")

    # ── Shared data store ─────────────────────────────────────────────────────

    def share(self, key: str, value: Any) -> None:
        """Share data between agents."""
        self._shared_data[key] = value

    def shared(self, key: str, default: Any = None) -> Any:
        """Retrieve shared data."""
        return self._shared_data.get(key, default)

    # ── Notes ─────────────────────────────────────────────────────────────────

    def add_note(self, note: str) -> None:
        """Add a free-form note (e.g., from critic agent)."""
        self._notes.append(note)

    def get_notes(self) -> list[str]:
        """Retrieve all notes."""
        return list(self._notes)

    # ── Context building ──────────────────────────────────────────────────────

    def build_context_for_executor(self, max_results: int = 5) -> dict[str, Any]:
        """Build a context dict to pass to an executor agent.

        Includes recent subtask results and shared data.

        Args:
            max_results: Maximum number of previous results to include.

        Returns:
            Context dict ready for executor.
        """
        context: dict[str, Any] = {}

        # Include recent successful results
        results = [r for r in self.all_results() if r.success][-max_results:]
        for r in results:
            context[f"result_{r.subtask_title[:30]}"] = r.output[:400]

        # Include shared data
        context.update({k: str(v)[:200] for k, v in self._shared_data.items()})

        # Include notes
        if self._notes:
            context["critic_notes"] = " | ".join(self._notes[-3:])

        return context

    def summary(self) -> dict[str, Any]:
        """Return a summary of working memory state."""
        return {
            "task": self.task[:100],
            "session_id": self.session_id,
            "age_s": round(time.time() - self.created_at, 1),
            "results_count": len(self._results),
            "completed_count": len(self.completed_ids()),
            "context_keys": list(self._context.keys()),
            "notes_count": len(self._notes),
        }
