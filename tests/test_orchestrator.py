"""Tests for orchestrator, event bus, workflow engine, and agent pool."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ────────────────────────── EventBus ──────────────────────────

class TestEventBus:
    def setup_method(self):
        from synthron.orchestrator.event_bus import EventBus
        self.bus = EventBus()

    def test_subscribe_and_emit(self):
        received = []

        def handler(event):
            received.append(event)

        self.bus.subscribe(handler)
        self.bus.emit({"type": "test", "content": "hello"})
        assert len(received) == 1
        assert received[0]["content"] == "hello"

    def test_typed_subscription(self):
        received = []

        def handler(event):
            received.append(event)

        self.bus.subscribe(handler, event_types=["action"])
        self.bus.emit({"type": "thought", "content": "thinking..."})
        self.bus.emit({"type": "action", "content": "doing..."})
        # Only action events should arrive
        assert len(received) == 1
        assert received[0]["content"] == "doing..."

    def test_multiple_subscribers(self):
        a_received = []
        b_received = []

        self.bus.subscribe(lambda e: a_received.append(e))
        self.bus.subscribe(lambda e: b_received.append(e))

        self.bus.emit({"type": "test", "content": "broadcast"})
        assert len(a_received) == 1
        assert len(b_received) == 1

    def test_event_history(self):
        for i in range(5):
            self.bus.emit({"type": "test", "content": f"event_{i}"})
        history = self.bus.get_history(limit=3)
        assert len(history) == 3
        assert history[-1]["content"] == "event_4"

    def test_emit_does_not_crash_on_bad_handler(self):
        def bad_handler(event):
            raise RuntimeError("boom")

        self.bus.subscribe(bad_handler)
        # Should not raise
        self.bus.emit({"type": "test", "content": "safe"})

    def test_unsubscribe(self):
        received = []

        def handler(event):
            received.append(event)

        self.bus.subscribe(handler)
        self.bus.unsubscribe(handler)
        self.bus.emit({"type": "test", "content": "ignored"})
        assert len(received) == 0

    def test_clear_history(self):
        self.bus.emit({"type": "test", "content": "x"})
        self.bus.clear_history()
        assert len(self.bus.get_history()) == 0


# ────────────────────────── WorkingMemory ──────────────────────────

class TestWorkingMemoryInOrchestrator:
    def test_working_memory_build_context(self):
        from synthron.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set_context("You are a helpful assistant.")
        wm.add_tool_result("web_search", {"query": "Python", "results": ["result1"]})
        wm.add_agent_output("planner", "I will research Python.")

        context = wm.build_context_for_executor()
        assert "context" in context.lower() or "assistant" in context.lower()

    def test_working_memory_shared_data(self):
        from synthron.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set_shared("key1", "value1")
        wm.set_shared("key2", 42)

        assert wm.get_shared("key1") == "value1"
        assert wm.get_shared("key2") == 42
        assert wm.get_shared("missing") is None

    def test_working_memory_notes(self):
        from synthron.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.add_note("Important finding: X is better than Y")
        wm.add_note("Remember to check Z")

        notes = wm.get_notes()
        assert len(notes) == 2
        assert "X is better than Y" in notes[0]

    def test_working_memory_clear(self):
        from synthron.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set_context("context")
        wm.add_note("note")
        wm.clear()

        assert wm.get_notes() == []


# ────────────────────────── TaskQueue ──────────────────────────

class TestTaskQueue:
    @pytest.mark.asyncio
    async def test_basic_enqueue_dequeue(self):
        from synthron.orchestrator.task_queue import AsyncTaskQueue, QueueEntry
        import time

        queue = AsyncTaskQueue(max_concurrent=2)

        entry = QueueEntry(
            task_id="t1",
            task="do something",
            priority=1,
            created_at=time.time(),
        )
        await queue.enqueue(entry)

        dequeued = await queue.dequeue()
        assert dequeued is not None
        assert dequeued.task_id == "t1"

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        from synthron.orchestrator.task_queue import AsyncTaskQueue, QueueEntry
        import time

        queue = AsyncTaskQueue(max_concurrent=5)

        for priority, task_id in [(3, "low"), (1, "high"), (2, "medium")]:
            entry = QueueEntry(
                task_id=task_id,
                task=f"task_{task_id}",
                priority=priority,
                created_at=time.time(),
            )
            await queue.enqueue(entry)

        first = await queue.dequeue()
        assert first.task_id == "high"  # priority 1 = highest

    @pytest.mark.asyncio
    async def test_queue_size(self):
        from synthron.orchestrator.task_queue import AsyncTaskQueue, QueueEntry
        import time

        queue = AsyncTaskQueue(max_concurrent=5)
        assert queue.size() == 0

        await queue.enqueue(QueueEntry("t1", "task", 1, time.time()))
        assert queue.size() == 1


# ────────────────────────── SessionManager ──────────────────────────

class TestSessionManager:
    def test_create_session(self):
        from synthron.orchestrator.session_manager import SessionManager

        mgr = SessionManager(ttl_seconds=3600)
        session = mgr.create_session()
        assert session.session_id
        assert session.created_at > 0

    def test_get_session(self):
        from synthron.orchestrator.session_manager import SessionManager

        mgr = SessionManager(ttl_seconds=3600)
        s1 = mgr.create_session()
        s2 = mgr.get_session(s1.session_id)
        assert s2 is not None
        assert s2.session_id == s1.session_id

    def test_session_not_found(self):
        from synthron.orchestrator.session_manager import SessionManager

        mgr = SessionManager(ttl_seconds=3600)
        result = mgr.get_session("nonexistent-session-id")
        assert result is None

    def test_delete_session(self):
        from synthron.orchestrator.session_manager import SessionManager

        mgr = SessionManager(ttl_seconds=3600)
        s = mgr.create_session()
        mgr.delete_session(s.session_id)
        assert mgr.get_session(s.session_id) is None

    def test_list_sessions(self):
        from synthron.orchestrator.session_manager import SessionManager

        mgr = SessionManager(ttl_seconds=3600)
        mgr.create_session()
        mgr.create_session()
        sessions = mgr.list_sessions()
        assert len(sessions) >= 2

    def test_expired_session_cleanup(self):
        from synthron.orchestrator.session_manager import SessionManager
        import time

        mgr = SessionManager(ttl_seconds=1)  # 1 second TTL
        s = mgr.create_session()
        time.sleep(1.1)
        mgr.cleanup_expired()
        assert mgr.get_session(s.session_id) is None


# ────────────────────────── WorkflowEngine ──────────────────────────

class TestWorkflowEngine:
    @pytest.mark.asyncio
    async def test_simple_dag_execution(self):
        from synthron.orchestrator.workflow_engine import WorkflowEngine
        from synthron.agents.base_agent import SubTask, TaskStatus

        call_log = []

        async def mock_executor(subtask):
            call_log.append(subtask.id)
            return {"output": f"done_{subtask.id}", "success": True}

        engine = WorkflowEngine()

        subtasks = [
            SubTask(id="t1", task="step 1", dependencies=[]),
            SubTask(id="t2", task="step 2", dependencies=["t1"]),
            SubTask(id="t3", task="step 3", dependencies=["t1"]),
            SubTask(id="t4", task="step 4", dependencies=["t2", "t3"]),
        ]

        results = await engine.execute(subtasks, executor_fn=mock_executor)

        # t1 must run first
        assert call_log.index("t1") < call_log.index("t2")
        assert call_log.index("t1") < call_log.index("t3")
        # t4 must run last
        assert call_log.index("t2") < call_log.index("t4")
        assert call_log.index("t3") < call_log.index("t4")

    @pytest.mark.asyncio
    async def test_independent_tasks_run_in_parallel(self):
        from synthron.orchestrator.workflow_engine import WorkflowEngine
        from synthron.agents.base_agent import SubTask
        import time

        start_times = {}

        async def mock_executor(subtask):
            start_times[subtask.id] = time.time()
            await asyncio.sleep(0.05)  # simulate work
            return {"output": "done", "success": True}

        engine = WorkflowEngine()

        # t1, t2, t3 have no dependencies — should run in parallel
        subtasks = [
            SubTask(id="t1", task="parallel 1", dependencies=[]),
            SubTask(id="t2", task="parallel 2", dependencies=[]),
            SubTask(id="t3", task="parallel 3", dependencies=[]),
        ]

        start = time.time()
        results = await engine.execute(subtasks, executor_fn=mock_executor)
        elapsed = time.time() - start

        # Should complete in ~0.05s (parallel), not ~0.15s (sequential)
        assert elapsed < 0.12, f"Parallel tasks took {elapsed:.3f}s, expected < 0.12s"

    @pytest.mark.asyncio
    async def test_failed_subtask_skips_dependents(self):
        from synthron.orchestrator.workflow_engine import WorkflowEngine
        from synthron.agents.base_agent import SubTask

        executed = []

        async def mock_executor(subtask):
            executed.append(subtask.id)
            if subtask.id == "t1":
                return {"output": "", "success": False, "error": "simulated failure"}
            return {"output": "done", "success": True}

        engine = WorkflowEngine()

        subtasks = [
            SubTask(id="t1", task="failing task", dependencies=[]),
            SubTask(id="t2", task="depends on t1", dependencies=["t1"]),
        ]

        results = await engine.execute(subtasks, executor_fn=mock_executor)

        assert "t1" in executed
        assert "t2" not in executed  # should be skipped due to t1 failure


# ────────────────────────── AgentPool ──────────────────────────

class TestAgentPool:
    def test_get_planner(self):
        from synthron.orchestrator.agent_pool import AgentPool

        pool = AgentPool(max_executors=2)
        planner = pool.get_planner()
        assert planner is not None

        # Singleton — same instance each call
        planner2 = pool.get_planner()
        assert planner is planner2

    def test_get_critic(self):
        from synthron.orchestrator.agent_pool import AgentPool

        pool = AgentPool(max_executors=2)
        critic = pool.get_critic()
        assert critic is not None

    @pytest.mark.asyncio
    async def test_get_executor_creates_up_to_max(self):
        from synthron.orchestrator.agent_pool import AgentPool

        pool = AgentPool(max_executors=2)

        e1 = await pool.get_executor()
        e2 = await pool.get_executor()
        # Third call should still work (reuse or wait)
        assert e1 is not None
        assert e2 is not None


# ────────────────────────── Orchestrator Integration ──────────────────────────

class TestOrchestratorIntegration:
    @pytest.mark.asyncio
    async def test_orchestrator_status(self):
        """Orchestrator.status() returns expected keys without full initialization."""
        from synthron.orchestrator.orchestrator import Orchestrator

        orch = Orchestrator(max_executors=1)
        status = orch.status()

        assert "router" in status
        assert "memory" in status or "session" in status or "initialized" in status

    @pytest.mark.asyncio
    async def test_orchestrator_subscribe(self):
        """subscribe() wires up event handler."""
        from synthron.orchestrator.orchestrator import Orchestrator

        orch = Orchestrator(max_executors=1)
        events = []

        def handler(e):
            events.append(e)

        orch.subscribe(handler)
        orch._emit({"type": "test", "content": "hello"})
        assert len(events) == 1
