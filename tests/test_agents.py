"""Tests for Synthron agents."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synthron.agents.base_agent import SubTask, SubTaskResult, TaskStatus
from synthron.agents.planner_agent import PlannerAgent
from synthron.agents.executor_agent import ExecutorAgent
from synthron.agents.critic_agent import CriticAgent


@pytest.fixture
def mock_provider():
    """Mock LLM provider."""
    provider = AsyncMock()
    provider.generate = AsyncMock()
    provider.is_available = MagicMock(return_value=True)
    return provider


@pytest.fixture
def planner(mock_provider):
    return PlannerAgent(provider=mock_provider)


@pytest.fixture
def executor(mock_provider):
    return ExecutorAgent(provider=mock_provider)


@pytest.fixture
def critic(mock_provider):
    return CriticAgent(provider=mock_provider)


class TestPlannerAgent:
    async def test_plan_returns_task_plan(self, planner, mock_provider):
        from synthron.providers.base_provider import GenerationResponse
        mock_provider.generate.return_value = GenerationResponse(
            content='{"complexity": 5, "estimated_time_s": 60, "subtasks": ['
                    '{"index": 1, "title": "Search web", "description": "Search for AI news",'
                    '"tool_hint": "web_search", "depends_on": []}]}',
            model="test",
            provider="test",
            total_tokens=100,
        )
        plan = await planner.plan("Research AI news 2026")
        assert plan.total_subtasks == 1
        assert plan.subtasks[0].title == "Search web"
        assert plan.complexity == 5

    async def test_estimate_complexity_returns_int(self, planner, mock_provider):
        from synthron.providers.base_provider import GenerationResponse
        mock_provider.generate.return_value = GenerationResponse(
            content="7", model="test", provider="test", total_tokens=5
        )
        complexity = await planner.estimate_complexity("Write a complex ML model")
        assert 1 <= complexity <= 10

    async def test_plan_fallback_on_bad_json(self, planner, mock_provider):
        from synthron.providers.base_provider import GenerationResponse
        mock_provider.generate.return_value = GenerationResponse(
            content="This is not JSON at all", model="test", provider="test", total_tokens=10
        )
        plan = await planner.plan("Simple task")
        # Should fall back to single subtask
        assert plan.total_subtasks >= 1


class TestExecutorAgent:
    async def test_execute_returns_result(self, executor, mock_provider):
        from synthron.providers.base_provider import GenerationResponse
        mock_provider.generate.return_value = GenerationResponse(
            content="The answer is 42.",
            model="test",
            provider="test",
            total_tokens=50,
        )
        subtask = SubTask(
            title="Answer question",
            description="What is the answer to life?",
            tool_hint="",
        )
        result = await executor.execute(subtask)
        assert result.success is True
        assert "42" in result.output
        assert result.subtask_id == subtask.id

    async def test_execute_handles_provider_error(self, executor, mock_provider):
        from synthron.utils.exceptions import ProviderError
        mock_provider.generate.side_effect = ProviderError("API error", provider="test")
        subtask = SubTask(title="Fail task", description="This will fail")
        result = await executor.execute(subtask)
        assert result.success is False
        assert result.error != ""


class TestCriticAgent:
    async def test_score_pass(self, critic, mock_provider):
        from synthron.providers.base_provider import GenerationResponse
        mock_provider.generate.return_value = GenerationResponse(
            content='{"score": 0.92, "verdict": "PASS", "feedback": "Excellent.", "improvement_hint": ""}',
            model="test",
            provider="test",
            total_tokens=30,
        )
        subtask = SubTask(title="Test", description="Test subtask")
        result = SubTaskResult(
            subtask_id=subtask.id,
            subtask_title=subtask.title,
            output="A comprehensive and detailed answer that covers all aspects.",
            success=True,
        )
        score = await critic.score(subtask, result)
        assert score.score >= 0.8
        assert score.verdict == "PASS"
        assert score.should_retry is False

    async def test_score_fail_triggers_retry(self, critic, mock_provider):
        from synthron.providers.base_provider import GenerationResponse
        mock_provider.generate.return_value = GenerationResponse(
            content='{"score": 0.3, "verdict": "FAIL", "feedback": "Too vague.", "improvement_hint": "Add details"}',
            model="test",
            provider="test",
            total_tokens=25,
        )
        subtask = SubTask(title="Test", description="Test subtask")
        result = SubTaskResult(
            subtask_id=subtask.id,
            subtask_title=subtask.title,
            output="short answer",
            success=True,
        )
        score = await critic.score(subtask, result)
        assert score.score < 0.5
        assert score.verdict == "FAIL"
        assert score.should_retry is True

    async def test_score_empty_result_is_zero(self, critic):
        subtask = SubTask(title="Empty", description="Empty test")
        result = SubTaskResult(
            subtask_id=subtask.id,
            subtask_title=subtask.title,
            output="",
            success=False,
        )
        score = await critic.score(subtask, result)
        assert score.score == 0.0


class TestSubTask:
    def test_subtask_default_status(self):
        st = SubTask(title="Test", description="Test desc")
        assert st.status == TaskStatus.PENDING

    def test_subtask_id_is_short(self):
        st = SubTask(title="Test", description="desc")
        assert len(st.id) == 8
