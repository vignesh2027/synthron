"""Tests for Synthron memory subsystems."""

from __future__ import annotations

import pytest
import asyncio

from synthron.memory.short_term import InMemoryBuffer
from synthron.memory.working_memory import WorkingMemory
from synthron.agents.base_agent import SubTaskResult


class TestInMemoryBuffer:
    @pytest.fixture
    def buffer(self):
        return InMemoryBuffer(max_size=5)

    async def test_add_and_get(self, buffer):
        await buffer.add("user", "Hello")
        items = await buffer.get_all()
        assert len(items) == 1
        assert items[0]["content"] == "Hello"

    async def test_max_size_enforced(self, buffer):
        for i in range(10):
            await buffer.add("user", f"Message {i}")
        items = await buffer.get_all()
        assert len(items) == 5  # max_size=5

    async def test_get_last_n(self, buffer):
        for i in range(4):
            await buffer.add("user", f"Message {i}")
        last_2 = await buffer.get_last_n(2)
        assert len(last_2) == 2
        assert last_2[-1]["content"] == "Message 3"

    async def test_clear(self, buffer):
        await buffer.add("user", "test")
        await buffer.clear()
        items = await buffer.get_all()
        assert len(items) == 0

    async def test_to_messages(self, buffer):
        await buffer.add("user", "Hello")
        await buffer.add("assistant", "Hi there!")
        msgs = buffer.to_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"


class TestWorkingMemory:
    @pytest.fixture
    def wm(self):
        return WorkingMemory(task="Test task", session_id="test-123")

    def test_set_and_get(self, wm):
        wm.set("key1", "value1")
        assert wm.get("key1") == "value1"

    def test_get_missing_returns_default(self, wm):
        assert wm.get("missing", "default") == "default"

    def test_store_and_retrieve_result(self, wm):
        result = SubTaskResult(
            subtask_id="abc123",
            subtask_title="Test",
            output="Result output",
            success=True,
        )
        wm.store_result(result)
        retrieved = wm.get_result("abc123")
        assert retrieved is result

    def test_completed_ids(self, wm):
        result = SubTaskResult(
            subtask_id="abc123",
            subtask_title="Test",
            output="Done",
            success=True,
        )
        wm.store_result(result)
        assert "abc123" in wm.completed_ids()

    def test_add_and_get_notes(self, wm):
        wm.add_note("Important observation")
        wm.add_note("Another note")
        notes = wm.get_notes()
        assert len(notes) == 2
        assert "Important observation" in notes

    def test_build_context_for_executor(self, wm):
        result = SubTaskResult(
            subtask_id="xyz",
            subtask_title="Step 1",
            output="Some output data",
            success=True,
        )
        wm.store_result(result)
        ctx = wm.build_context_for_executor()
        assert isinstance(ctx, dict)

    def test_summary(self, wm):
        summary = wm.summary()
        assert "task" in summary
        assert "session_id" in summary
        assert summary["results_count"] == 0
