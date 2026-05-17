"""Workflow engine — DAG-based parallel task execution."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from synthron.agents.base_agent import SubTask, SubTaskResult, TaskPlan, TaskStatus
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowEngine:
    """Execute a TaskPlan as a DAG with parallel execution of independent subtasks.

    Algorithm:
    1. Build dependency graph from subtask.depends_on
    2. Find all subtasks with no pending dependencies (ready set)
    3. Execute ready subtasks in parallel (up to max_concurrent)
    4. As subtasks complete, unlock dependents
    5. Repeat until all subtasks complete or fail
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(
        self,
        plan: TaskPlan,
        executor_fn: Callable[[SubTask, dict], Any],
        context: dict[str, Any] | None = None,
        on_complete: Callable[[SubTaskResult], Any] | None = None,
    ) -> list[SubTaskResult]:
        """Execute a task plan as a parallel DAG.

        Args:
            plan: The TaskPlan with ordered subtasks.
            executor_fn: Async function(subtask, context) → SubTaskResult.
            context: Shared mutable context dict (updated as results arrive).
            on_complete: Optional callback called after each subtask completes.

        Returns:
            List of SubTaskResult objects in completion order.
        """
        ctx = dict(context or {})
        results: list[SubTaskResult] = []
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()
        lock = asyncio.Lock()
        pending = {st.id: st for st in plan.subtasks}

        logger.info(
            f"[workflow] Starting DAG execution: {len(plan.subtasks)} subtasks, "
            f"max_concurrent={self.max_concurrent}"
        )

        async def run_subtask(subtask: SubTask) -> None:
            async with self._semaphore:
                logger.debug(f"[workflow] Starting subtask #{subtask.index}: '{subtask.title}'")
                t_start = time.perf_counter()

                try:
                    result: SubTaskResult = await executor_fn(subtask, dict(ctx))
                except Exception as exc:
                    logger.error(f"[workflow] Subtask '{subtask.title}' raised: {exc}")
                    result = SubTaskResult(
                        subtask_id=subtask.id,
                        subtask_title=subtask.title,
                        output="",
                        success=False,
                        error=str(exc),
                        latency_ms=(time.perf_counter() - t_start) * 1000,
                    )

                elapsed = (time.perf_counter() - t_start) * 1000
                result.latency_ms = elapsed

                async with lock:
                    results.append(result)
                    if result.success:
                        completed_ids.add(subtask.id)
                        subtask.status = TaskStatus.COMPLETED
                        # Update shared context with result
                        ctx[f"result_{subtask.title[:30]}"] = result.output[:400]
                    else:
                        failed_ids.add(subtask.id)
                        subtask.status = TaskStatus.FAILED

                logger.debug(
                    f"[workflow] Subtask '{subtask.title}' → "
                    f"{'✅' if result.success else '❌'} ({elapsed:.0f}ms)"
                )

                if on_complete:
                    try:
                        if asyncio.iscoroutinefunction(on_complete):
                            await on_complete(result)
                        else:
                            on_complete(result)
                    except Exception:
                        pass

        # Run DAG loop
        launched: set[str] = set()
        running_tasks: list[asyncio.Task] = []

        while True:
            # Find newly-ready subtasks
            ready = [
                st for st_id, st in pending.items()
                if st_id not in launched
                and st_id not in completed_ids
                and st_id not in failed_ids
                and all(dep in completed_ids for dep in st.depends_on)
            ]

            for st in ready:
                launched.add(st.id)
                task = asyncio.create_task(run_subtask(st))
                running_tasks.append(task)

            # Check if we're done
            all_done = all(
                st.id in completed_ids or st.id in failed_ids
                for st in plan.subtasks
            )

            if all_done:
                break

            # Check for deadlock: nothing running and nothing ready
            running = [t for t in running_tasks if not t.done()]
            if not running and not ready:
                # Some subtasks have unresolvable dependencies (dep chain failed)
                stuck = [
                    st for st in plan.subtasks
                    if st.id not in completed_ids and st.id not in failed_ids
                ]
                logger.warning(
                    f"[workflow] DAG deadlock: {len(stuck)} subtasks cannot run "
                    f"(failed deps). Skipping."
                )
                for st in stuck:
                    results.append(SubTaskResult(
                        subtask_id=st.id,
                        subtask_title=st.title,
                        output="",
                        success=False,
                        error="Dependencies failed — skipped.",
                    ))
                break

            # Wait for any running task to complete before checking for new ready ones
            if running:
                await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

        # Await any remaining tasks
        remaining = [t for t in running_tasks if not t.done()]
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)

        logger.info(
            f"[workflow] DAG complete: {len(completed_ids)} done, "
            f"{len(failed_ids)} failed"
        )
        return results

    def topological_order(self, plan: TaskPlan) -> list[list[SubTask]]:
        """Return subtasks grouped into parallel execution waves.

        Returns a list of waves where each wave's subtasks can run in parallel.

        Args:
            plan: TaskPlan to analyze.

        Returns:
            List of waves (each wave is a list of SubTask).
        """
        id_map = {st.id: st for st in plan.subtasks}
        completed: set[str] = set()
        waves: list[list[SubTask]] = []
        remaining = list(plan.subtasks)

        while remaining:
            wave = [
                st for st in remaining
                if all(dep in completed for dep in st.depends_on)
            ]
            if not wave:
                # Remaining tasks have unresolvable deps — add them as final wave
                waves.append(remaining)
                break
            waves.append(wave)
            completed.update(st.id for st in wave)
            remaining = [st for st in remaining if st.id not in completed]

        return waves
