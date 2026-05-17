"""Synthron Orchestrator — master controller for all agent operations."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Callable

from synthron.agents.base_agent import (
    AgentResult,
    CriticScore,
    FinalResult,
    SubTask,
    SubTaskResult,
    TaskPlan,
    TaskStatus,
)
from synthron.memory.memory_manager import MemoryManager
from synthron.orchestrator.agent_pool import AgentPool
from synthron.orchestrator.event_bus import AgentEvent, event_bus
from synthron.orchestrator.session_manager import Session, session_manager
from synthron.orchestrator.workflow_engine import WorkflowEngine
from synthron.providers.smart_router import router
from synthron.self_improvement.failure_analyzer import FailureAnalyzer
from synthron.self_improvement.performance_tracker import PerformanceTracker
from synthron.utils.config import settings
from synthron.utils.exceptions import MaxRetriesExceededError, PlanningError
from synthron.utils.logger import get_logger, print_banner

logger = get_logger(__name__)


class Orchestrator:
    """Synthron's master orchestrator — coordinates all agents end-to-end.

    Full pipeline for one user task:
    ┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌────────┐
    │  Planner │ →  │ Execute │ →  │    Critic    │ →  │ Memory │
    │ (Gemini) │    │ (Cerebras│   │  (DeepSeek)  │    │(store) │
    └──────────┘    └─────────┘    └──────────────┘    └────────┘
         │              │                  │                 │
         ↓              ↓                  ↓                 ↓
      TaskPlan    SubTaskResults      CriticScores     EpisodicDB
                                        (retry if FAIL)

    Usage:
        orch = Orchestrator()
        result = await orch.run("Research AI companies 2026")
    """

    def __init__(
        self,
        tools: list | None = None,
        max_executors: int | None = None,
        critic_threshold: float | None = None,
        max_retries: int | None = None,
        dashboard: bool = False,
        stream_callbacks: list[Callable] | None = None,
    ) -> None:
        self._critic_threshold = critic_threshold or settings.agents.critic_pass_threshold
        self._max_retries = max_retries or settings.agents.max_retries
        self._dashboard = dashboard
        self._stream_callbacks: list[Callable] = stream_callbacks or []

        self._pool = AgentPool(
            max_executors=max_executors,
            stream_callbacks=self._stream_callbacks,
            tools=tools,
        )
        self._memory = MemoryManager()
        self._workflow = WorkflowEngine(
            max_concurrent=max_executors or settings.agents.max_parallel_executors
        )
        self._failure_analyzer = FailureAnalyzer()
        self._perf_tracker = PerformanceTracker()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all subsystems."""
        if self._initialized:
            return

        logger.info("[orchestrator] Initializing Synthron...")
        print_banner()

        await asyncio.gather(
            router.initialize(),
            self._memory.initialize(),
            return_exceptions=True,
        )

        # Wire memory into memory agent
        self._pool.memory_agent.attach_memory(self._memory)

        # Wire event bus callbacks
        event_bus.subscribe(self._on_event)

        self._initialized = True
        logger.info(f"[orchestrator] Ready. Providers: {list(router._providers.keys())}")

    async def run(
        self,
        task: str,
        session_id: str = "",
        stream: bool = False,
    ) -> FinalResult:
        """Run a complete task through the full Synthron pipeline.

        Args:
            task: User task string.
            session_id: Existing session ID, or empty to create new.
            stream: If True, stream events to registered callbacks.

        Returns:
            FinalResult with complete output and metadata.
        """
        if not self._initialized:
            await self.initialize()

        session = session_manager.get_or_create(session_id)
        session.active_task = task
        session.status = "running"
        session.task_count += 1

        working = self._memory.start_working(task)
        run_start = time.perf_counter()
        total_tokens = 0
        retry_count = 0
        providers_used: set[str] = set()

        await self._emit("task_start", "orchestrator", "coordinator", f"Task: {task[:100]}", session.id)
        logger.info(f"[orchestrator] ▶ Task: {task[:100]}")

        try:
            # ── STEP 1: Check memory for similar past runs ──────────────────
            past_context = await self._memory.get_context_for_task(task)
            if past_context:
                logger.debug("[orchestrator] Past context found, injecting into planner")

            # ── STEP 2: Planning ────────────────────────────────────────────
            planner = self._pool.planner
            plan: TaskPlan = await self._plan_with_retry(task, past_context, session.id)
            total_tokens += planner._total_tokens

            await self._emit(
                "plan_created", "planner", "planner",
                f"Plan: {plan.total_subtasks} subtasks (complexity={plan.complexity})",
                session.id,
            )

            # ── STEP 3: Execute + Critique loop ─────────────────────────────
            all_results: list[SubTaskResult] = []
            all_scores: list[CriticScore] = []

            async def execute_subtask(subtask: SubTask, ctx: dict) -> SubTaskResult:
                nonlocal total_tokens, retry_count, providers_used

                executor = await self._pool.get_executor()
                attempt = 1
                improvement_hint = ""

                while attempt <= self._max_retries + 1:
                    result = await executor.execute(
                        subtask, context=ctx, attempt=attempt
                    )
                    total_tokens += result.tokens_used
                    if result.tool_used:
                        providers_used.add(result.tool_used)

                    # ── Critique ───────────────────────────────────────────
                    critic = self._pool.critic
                    score = await critic.score(subtask, result)
                    all_scores.append(score)
                    total_tokens += critic._total_tokens - (total_tokens - total_tokens)

                    if score.verdict == "PASS" or not score.should_retry:
                        return result

                    if attempt > self._max_retries:
                        logger.warning(
                            f"[orchestrator] '{subtask.title}' max retries reached "
                            f"(score={score.score:.2f})"
                        )
                        await self._failure_analyzer.record(subtask, result, score)
                        return result

                    # ── Retry with improvement hint ────────────────────────
                    retry_count += 1
                    improvement_hint = await critic.suggest_improvement(score)
                    subtask.description = (
                        f"{subtask.description}\n\n[IMPROVEMENT NEEDED]: {improvement_hint}"
                    )
                    subtask.status = TaskStatus.PENDING
                    await self._emit(
                        "retry", "critic", "critic",
                        f"Retrying '{subtask.title}' (attempt {attempt+1}): {improvement_hint[:80]}",
                        session.id,
                    )
                    attempt += 1

                return result

            # ── DAG execution ────────────────────────────────────────────────
            all_results = await self._workflow.execute(
                plan=plan,
                executor_fn=execute_subtask,
                context=working.build_context_for_executor(),
                on_complete=lambda r: working.store_result(r),
            )

            # ── STEP 4: Merge results into final output ──────────────────────
            final_output = await self._merge_results(task, plan, all_results)
            elapsed_s = time.perf_counter() - run_start
            success = any(r.success for r in all_results)

            # ── STEP 5: Memory persistence ───────────────────────────────────
            await self._memory.remember_task_result(task, final_output[:2000], success)
            episode_results = [r.model_dump() for r in all_results]
            await self._memory.store_episode(
                task=task,
                plan=plan.model_dump(),
                results=episode_results,
                success=success,
                tokens=total_tokens,
                time_s=elapsed_s,
            )

            # ── STEP 6: Performance tracking ──────────────────────────────────
            await self._perf_tracker.record(
                task=task,
                success=success,
                tokens=total_tokens,
                time_s=elapsed_s,
                retry_count=retry_count,
                scores=[s.score for s in all_scores],
            )

            result = FinalResult(
                task=task,
                output=final_output,
                success=success,
                total_tokens=total_tokens,
                total_time_s=round(elapsed_s, 2),
                providers_used=list(providers_used),
                retry_count=retry_count,
            )

            session.status = "idle"
            session.total_tokens += total_tokens

            await self._emit(
                "task_done", "orchestrator", "coordinator",
                f"Done in {elapsed_s:.1f}s | {total_tokens:,} tokens",
                session.id,
            )
            logger.info(
                f"[orchestrator] ✅ Task complete | {elapsed_s:.1f}s | "
                f"{total_tokens:,} tokens | retries={retry_count}"
            )
            return result

        except Exception as exc:
            elapsed_s = time.perf_counter() - run_start
            session.status = "error"
            logger.error(f"[orchestrator] ❌ Task failed: {exc}")
            await self._emit("task_error", "orchestrator", "coordinator", str(exc), session.id)

            return FinalResult(
                task=task,
                output="",
                success=False,
                total_tokens=total_tokens,
                total_time_s=round(elapsed_s, 2),
                retry_count=retry_count,
                error=str(exc),
            )

    async def stream(self, task: str, session_id: str = "") -> AsyncIterator[str]:
        """Run task and stream token-by-token output.

        Args:
            task: User task string.
            session_id: Optional session ID.

        Yields:
            String chunks as they are generated.
        """
        if not self._initialized:
            await self.initialize()

        yield f"[Synthron] Planning: {task[:60]}...\n\n"

        # Run in background and stream events
        result_holder: list[FinalResult] = []
        run_task = asyncio.create_task(self.run(task, session_id=session_id, stream=True))

        # Stream event messages while task runs
        seen_events = set()
        while not run_task.done():
            events = event_bus.get_history(session_id=session_id, limit=50)
            for e in events:
                if e.id not in seen_events:
                    seen_events.add(e.id)
                    yield f"[{e.agent_name.upper()}] {e.content}\n"
            await asyncio.sleep(0.1)

        result = await run_task
        if result.output:
            yield f"\n{'='*60}\n## FINAL OUTPUT\n{'='*60}\n\n{result.output}"
        else:
            yield f"\n[Error]: {result.error}"

    async def _plan_with_retry(
        self, task: str, context: str, session_id: str
    ) -> TaskPlan:
        """Plan with retry on failure."""
        planner = self._pool.planner
        if context:
            task_with_context = f"{task}\n\n[Relevant context from memory]:\n{context[:1000]}"
        else:
            task_with_context = task

        for attempt in range(1, 3):
            try:
                return await planner.plan(task_with_context)
            except Exception as exc:
                if attempt == 2:
                    raise PlanningError("planner", task, str(exc)) from exc
                logger.warning(f"[orchestrator] Planning attempt {attempt} failed: {exc}")
                await asyncio.sleep(1)

    async def _merge_results(
        self, task: str, plan: TaskPlan, results: list[SubTaskResult]
    ) -> str:
        """Merge all subtask results into a cohesive final output."""
        successful = [r for r in results if r.success and r.output]

        if not successful:
            return "Task could not be completed. All subtasks failed."

        if len(successful) == 1:
            return successful[0].output

        # Synthesize with planner (Gemini — large context)
        sections = "\n\n".join(
            f"### {r.subtask_title}\n{r.output[:1500]}"
            for r in successful
        )
        synthesis_prompt = (
            f"Original task: {task}\n\n"
            f"Results from {len(successful)} subtasks:\n{sections}\n\n"
            f"Synthesize all these results into one comprehensive, well-structured final report. "
            f"Integrate everything seamlessly. Format with headers and bullets where appropriate."
        )

        try:
            response = await self._pool.planner.generate(
                synthesis_prompt, max_tokens=4096, temperature=0.4
            )
            return response.content
        except Exception as exc:
            logger.warning(f"[orchestrator] Synthesis failed: {exc}, concatenating instead")
            return "\n\n".join(
                f"## {r.subtask_title}\n{r.output}" for r in successful
            )

    async def _emit(
        self,
        event_type: str,
        agent_name: str,
        agent_type: str,
        content: str,
        session_id: str = "",
    ) -> None:
        """Emit an event to the bus and stream callbacks."""
        await event_bus.emit(event_type, agent_name, agent_type, content, session_id)
        for cb in self._stream_callbacks:
            try:
                event_dict = {
                    "type": event_type,
                    "agent": agent_name,
                    "content": content,
                    "session_id": session_id,
                }
                if asyncio.iscoroutinefunction(cb):
                    await cb(event_dict)
                else:
                    cb(event_dict)
            except Exception:
                pass

    def _on_event(self, event: AgentEvent) -> None:
        """Handle incoming events from the event bus (logging)."""
        logger.debug(f"[bus] {event.agent_name}: {event.content[:80]}")

    def subscribe(self, callback: Callable) -> None:
        """Register a streaming callback."""
        self._stream_callbacks.append(callback)
        self._pool.add_stream_callback(callback)

    def status(self) -> dict[str, Any]:
        """Return orchestrator health and stats."""
        return {
            "initialized": self._initialized,
            "router": router.status(),
            "pool": self._pool.stats(),
            "memory": {"session_id": self._memory.session_id},
            "sessions": session_manager.stats(),
            "event_bus": event_bus.stats(),
        }
