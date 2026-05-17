"""Pattern Learner — build a knowledge base from past successful runs."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any

from synthron.utils.logger import get_logger

logger = get_logger(__name__)

# Task classification keywords
TASK_CATEGORIES = {
    "finance": ["price", "stock", "financial", "revenue", "profit", "market cap", "valuation"],
    "research": ["research", "study", "analyze", "investigate", "survey", "report"],
    "coding": ["code", "python", "javascript", "script", "function", "bug", "debug", "implement"],
    "data": ["csv", "data", "dataset", "pandas", "analysis", "statistics", "chart"],
    "web": ["website", "scrape", "url", "web page", "browser", "search"],
    "writing": ["write", "draft", "summarize", "essay", "document", "email"],
    "math": ["calculate", "compute", "formula", "equation", "percentage", "average"],
}

# Tool recommendations per task category
TOOL_RECOMMENDATIONS: dict[str, list[str]] = {
    "finance": ["web_search", "data_analyzer", "calculator"],
    "research": ["web_search", "browser_tool"],
    "coding": ["code_executor", "file_tool"],
    "data": ["data_analyzer", "code_executor"],
    "web": ["browser_tool", "web_search"],
    "writing": [],  # LLM only
    "math": ["calculator", "code_executor"],
}


class LearnedPattern:
    """A behavioral pattern extracted from past successful runs."""

    def __init__(
        self,
        key: str,
        description: str,
        tool_sequence: list[str],
        task_category: str,
    ) -> None:
        self.key = key
        self.description = description
        self.tool_sequence = tool_sequence
        self.task_category = task_category
        self.observed = 1
        self.success_count = 0
        self.failure_count = 0
        self.avg_score = 0.0
        self.last_seen = time.time()

    def record(self, success: bool, score: float) -> None:
        self.observed += 1
        self.last_seen = time.time()
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.avg_score = (self.avg_score * (self.observed - 1) + score) / self.observed

    @property
    def reliability(self) -> float:
        """Return success rate as reliability score."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "tool_sequence": self.tool_sequence,
            "category": self.task_category,
            "observed": self.observed,
            "reliability": round(self.reliability, 3),
            "avg_score": round(self.avg_score, 3),
        }


class PatternLearner:
    """Learns behavioral patterns from successful task runs.

    After 10+ runs, builds a knowledge base that guides future decisions:
    - Which tools to use for which task categories
    - Which tool sequences work best
    - Task-specific prompting strategies
    """

    MIN_OBSERVATIONS_TO_TRUST = 5

    def __init__(self) -> None:
        self._patterns: dict[str, LearnedPattern] = {}
        self._category_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0, "total": 0}
        )
        self._run_count = 0

    def learn(
        self,
        task: str,
        tool_sequence: list[str],
        scores: list[float],
        success: bool,
    ) -> None:
        """Extract and record patterns from a completed task run.

        Args:
            task: Task description.
            tool_sequence: List of tools used in order.
            scores: Critic scores per subtask.
            success: Whether the overall task succeeded.
        """
        self._run_count += 1
        avg_score = sum(scores) / len(scores) if scores else 0.0
        category = self._classify_task(task)

        # Update category stats
        stats = self._category_stats[category]
        stats["total"] += 1
        if success:
            stats["success"] += 1
        else:
            stats["failure"] += 1

        # Build pattern key
        tool_key = "+".join(sorted(set(tool_sequence))) if tool_sequence else "llm_only"
        pattern_key = f"{category}:{tool_key}"

        if pattern_key in self._patterns:
            self._patterns[pattern_key].record(success, avg_score)
        else:
            pattern = LearnedPattern(
                key=pattern_key,
                description=f"{category} task using {', '.join(set(tool_sequence)) or 'LLM only'}",
                tool_sequence=tool_sequence,
                task_category=category,
            )
            pattern.record(success, avg_score)
            self._patterns[pattern_key] = pattern

        logger.debug(
            f"[pattern_learner] Learned: {pattern_key} "
            f"(success={success}, score={avg_score:.2f})"
        )

    def get_tool_recommendation(self, task: str) -> list[str]:
        """Recommend tools based on learned patterns and task category.

        Args:
            task: Task description.

        Returns:
            Ordered list of recommended tool names.
        """
        category = self._classify_task(task)

        # Check learned patterns first
        category_patterns = [
            p for p in self._patterns.values()
            if p.task_category == category
            and p.observed >= self.MIN_OBSERVATIONS_TO_TRUST
        ]

        if category_patterns:
            best = max(category_patterns, key=lambda p: p.reliability)
            if best.reliability > 0.7:
                logger.debug(
                    f"[pattern_learner] Using learned pattern: {best.key} "
                    f"(reliability={best.reliability:.2f})"
                )
                return best.tool_sequence

        # Fallback to hardcoded recommendations
        return TOOL_RECOMMENDATIONS.get(category, ["web_search"])

    def get_prompt_hint(self, task: str) -> str:
        """Return a learned prompt hint for the given task.

        Args:
            task: Task description.

        Returns:
            Hint string to prepend to the agent prompt.
        """
        category = self._classify_task(task)
        hints = {
            "finance": "Always include specific numbers, percentages, and data sources.",
            "research": "Verify facts from at least 2 independent sources.",
            "coding": "Test the code in the executor before returning it.",
            "data": "Show sample data and computed statistics in your output.",
            "web": "Extract key content and summarize — don't return raw HTML.",
            "writing": "Structure with clear sections, headers, and bullet points.",
            "math": "Show the calculation steps, not just the final answer.",
        }
        return hints.get(category, "")

    def get_category_report(self) -> dict[str, Any]:
        """Return success rate per task category."""
        report = {}
        for cat, stats in self._category_stats.items():
            total = stats["total"]
            if total > 0:
                report[cat] = {
                    "total": total,
                    "success_rate": round(stats["success"] / total, 3),
                    "failures": stats["failure"],
                }
        return report

    def get_best_patterns(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Return the most reliable learned patterns."""
        mature = [
            p for p in self._patterns.values()
            if p.observed >= self.MIN_OBSERVATIONS_TO_TRUST
        ]
        sorted_patterns = sorted(mature, key=lambda p: p.reliability, reverse=True)
        return [p.to_dict() for p in sorted_patterns[:top_n]]

    def _classify_task(self, task: str) -> str:
        """Classify a task into a category based on keywords.

        Args:
            task: Task description string.

        Returns:
            Category name string.
        """
        task_lower = task.lower()
        scores: dict[str, int] = {}
        for category, keywords in TASK_CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=lambda k: scores[k])
        return "research"  # default

    def summary(self) -> dict[str, Any]:
        return {
            "total_runs_learned": self._run_count,
            "patterns_discovered": len(self._patterns),
            "category_report": self.get_category_report(),
            "top_patterns": self.get_best_patterns(5),
        }
