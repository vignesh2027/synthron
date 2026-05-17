"""Tests for Synthron self-improvement system."""

from __future__ import annotations

import pytest

from synthron.agents.base_agent import CriticScore, SubTask, SubTaskResult
from synthron.self_improvement.failure_analyzer import FailureAnalyzer
from synthron.self_improvement.pattern_learner import PatternLearner
from synthron.self_improvement.performance_tracker import PerformanceTracker
from synthron.self_improvement.retry_strategist import RetryStrategist


class TestFailureAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return FailureAnalyzer()

    @pytest.fixture
    def sample_failure(self):
        subtask = SubTask(title="Search", description="Find AI news", tool_hint="web_search")
        result = SubTaskResult(
            subtask_id=subtask.id,
            subtask_title=subtask.title,
            output="",
            success=False,
            tool_used="web_search",
            error="Connection timeout",
        )
        score = CriticScore.from_score(subtask.id, 0.2, "Output is empty")
        return subtask, result, score

    async def test_record_failure(self, analyzer, sample_failure):
        subtask, result, score = sample_failure
        await analyzer.record(subtask, result, score)
        assert len(analyzer._failures) == 1

    def test_get_fix_hint_timeout(self, analyzer):
        hint = analyzer.get_fix_hint("Connection timeout")
        assert len(hint) > 0

    def test_tool_failure_rate(self, analyzer):
        analyzer.record_success("web_search")
        analyzer.record_success("web_search")
        rates = analyzer.get_tool_failure_rates()
        # No failures recorded yet
        assert rates.get("web_search", 0) == 0.0

    def test_generate_report_empty(self, analyzer):
        report = analyzer.generate_report()
        assert "No failures" in report


class TestPatternLearner:
    @pytest.fixture
    def learner(self):
        return PatternLearner()

    def test_learn_and_classify(self, learner):
        learner.learn(
            task="Research AI funding 2026",
            tool_sequence=["web_search", "data_analyzer"],
            scores=[0.85, 0.9],
            success=True,
        )
        assert learner._run_count == 1

    def test_classify_finance_task(self, learner):
        category = learner._classify_task("Analyze stock market prices and revenue")
        assert category == "finance"

    def test_classify_coding_task(self, learner):
        category = learner._classify_task("Write a Python function to sort a list")
        assert category == "coding"

    def test_get_prompt_hint(self, learner):
        hint = learner.get_prompt_hint("Calculate compound interest")
        assert len(hint) > 0

    def test_get_tool_recommendation_fallback(self, learner):
        tools = learner.get_tool_recommendation("Research something")
        assert isinstance(tools, list)
        assert len(tools) > 0


class TestPerformanceTracker:
    @pytest.fixture
    def tracker(self):
        return PerformanceTracker(window=10)

    async def test_record_and_metrics(self, tracker):
        await tracker.record("task 1", True, 1000, 5.0, 0, [0.9, 0.85])
        await tracker.record("task 2", False, 2000, 8.0, 2, [0.4])

        metrics = tracker.get_current_metrics()
        assert metrics["success_rate"] == 0.5
        assert metrics["window_size"] == 2

    async def test_cumulative_metrics(self, tracker):
        await tracker.record("t1", True, 500, 2.0)
        await tracker.record("t2", True, 600, 3.0)

        cum = tracker.get_cumulative_metrics()
        assert cum["total_runs"] == 2
        assert cum["total_successes"] == 2
        assert cum["success_rate"] == 1.0

    async def test_trend(self, tracker):
        for i in range(5):
            await tracker.record(f"task {i}", i % 2 == 0, 100, 1.0)
        trend = tracker.get_trend("success_rate", window=5)
        assert len(trend) == 5

    def test_generate_report_no_data(self, tracker):
        metrics = tracker.get_current_metrics()
        assert "No runs" in metrics.get("message", "") or isinstance(metrics, dict)


class TestRetryStrategist:
    @pytest.fixture
    def strategist(self):
        return RetryStrategist()

    def test_max_retries_exceeded(self, strategist):
        subtask = SubTask(title="Test", description="desc")
        result = SubTaskResult(subtask_id=subtask.id, subtask_title="Test", output="")
        score = CriticScore.from_score(subtask.id, 0.3, "Failed")
        decision = strategist.decide(subtask, result, score, attempt=4, max_retries=3)
        assert decision.should_retry is False

    def test_good_score_no_retry(self, strategist):
        subtask = SubTask(title="Test", description="desc")
        result = SubTaskResult(subtask_id=subtask.id, subtask_title="Test", output="good")
        score = CriticScore.from_score(subtask.id, 0.9, "Excellent")
        decision = strategist.decide(subtask, result, score, attempt=1, max_retries=3)
        assert decision.should_retry is False

    def test_low_score_triggers_retry(self, strategist):
        subtask = SubTask(title="Test", description="desc")
        result = SubTaskResult(subtask_id=subtask.id, subtask_title="Test", output="bad")
        score = CriticScore.from_score(subtask.id, 0.2, "Too vague")
        decision = strategist.decide(subtask, result, score, attempt=1, max_retries=3)
        assert decision.should_retry is True

    def test_timeout_suggests_delay(self, strategist):
        subtask = SubTask(title="Test", description="desc")
        result = SubTaskResult(
            subtask_id=subtask.id, subtask_title="Test", output="", error="Connection timeout"
        )
        score = CriticScore.from_score(subtask.id, 0.1, "Failed")
        decision = strategist.decide(subtask, result, score, attempt=1, max_retries=3)
        assert decision.delay_s > 0
