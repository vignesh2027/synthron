"""Failure Analyzer — learn from failed subtasks to prevent repeat errors."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from synthron.agents.base_agent import CriticScore, SubTask, SubTaskResult
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class FailureAnalyzer:
    """Analyzes failed subtasks to extract patterns and prevent future failures.

    Tracks:
    - Which tool + task_type combinations fail most
    - Common error messages and their root causes
    - Suggested fixes for known failure patterns
    """

    def __init__(self) -> None:
        self._failures: list[dict[str, Any]] = []
        self._pattern_counts: dict[str, int] = defaultdict(int)
        self._tool_failure_rates: dict[str, dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0}
        )
        self._known_fixes: dict[str, str] = {
            "timeout": "Increase timeout or simplify the query.",
            "rate limit": "Add delay between requests or switch provider.",
            "no results": "Refine the search query with more specific terms.",
            "parse error": "Ensure the output format matches expectations.",
            "auth": "Verify API key is correct and has required permissions.",
            "connection": "Check network connectivity and retry.",
            "empty output": "Try a different tool or decompose the subtask further.",
        }

    async def record(
        self,
        subtask: SubTask,
        result: SubTaskResult,
        score: CriticScore,
    ) -> None:
        """Record a failed or low-scored subtask execution.

        Args:
            subtask: The subtask that failed/scored poorly.
            result: The execution result.
            score: The critic's evaluation.
        """
        failure_entry = {
            "ts": time.time(),
            "subtask_title": subtask.title,
            "subtask_description": subtask.description[:200],
            "tool_used": result.tool_used,
            "tool_hint": subtask.tool_hint,
            "score": score.score,
            "verdict": score.verdict,
            "error": result.error[:200] if result.error else "",
            "feedback": score.feedback,
            "tokens": result.tokens_used,
            "latency_ms": result.latency_ms,
        }

        self._failures.append(failure_entry)

        # Track tool failure rates
        if result.tool_used:
            self._tool_failure_rates[result.tool_used]["failure"] += 1

        # Extract and count failure patterns
        pattern = self._extract_pattern(failure_entry)
        if pattern:
            self._pattern_counts[pattern] += 1

        logger.debug(
            f"[failure_analyzer] Recorded failure: '{subtask.title}' "
            f"score={score.score:.2f} tool={result.tool_used}"
        )

    def record_success(self, tool_used: str) -> None:
        """Record a successful tool use to calculate failure rates."""
        if tool_used:
            self._tool_failure_rates[tool_used]["success"] += 1

    def get_fix_hint(self, error: str, tool: str = "") -> str:
        """Return a fix suggestion for a known error pattern.

        Args:
            error: Error message string.
            tool: Tool that was used.

        Returns:
            Actionable fix hint string.
        """
        error_lower = error.lower()
        for pattern, fix in self._known_fixes.items():
            if pattern in error_lower:
                return fix

        # Tool-specific hints
        tool_hints = {
            "web_search": "Try a more specific query or use the browser_tool instead.",
            "code_executor": "Simplify the code or break it into smaller pieces.",
            "api_caller": "Verify the URL and request format.",
            "browser_tool": "The page may be blocked. Try web_search instead.",
        }
        if tool in tool_hints:
            return tool_hints[tool]

        return "Retry with a different approach or tool."

    def get_top_patterns(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Return the most frequent failure patterns.

        Args:
            top_n: Number of patterns to return.

        Returns:
            List of {pattern, count, suggested_fix} dicts.
        """
        sorted_patterns = sorted(
            self._pattern_counts.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        return [
            {
                "pattern": p,
                "count": c,
                "suggested_fix": self.get_fix_hint(p),
            }
            for p, c in sorted_patterns
        ]

    def get_tool_failure_rates(self) -> dict[str, float]:
        """Return failure rate per tool (0.0 = perfect, 1.0 = always fails)."""
        rates = {}
        for tool, counts in self._tool_failure_rates.items():
            total = counts["success"] + counts["failure"]
            if total > 0:
                rates[tool] = round(counts["failure"] / total, 3)
        return rates

    def get_worst_tools(self, threshold: float = 0.3) -> list[str]:
        """Return tools with failure rate above threshold."""
        return [
            tool for tool, rate in self.get_tool_failure_rates().items()
            if rate >= threshold
        ]

    def generate_report(self) -> str:
        """Generate a human-readable failure analysis report."""
        if not self._failures:
            return "No failures recorded yet."

        lines = [
            f"## Failure Analysis Report",
            f"Total failures recorded: {len(self._failures)}",
            "",
            "### Top Failure Patterns",
        ]
        for p in self.get_top_patterns():
            lines.append(f"  - {p['pattern']} ({p['count']}x): {p['suggested_fix']}")

        tool_rates = self.get_tool_failure_rates()
        if tool_rates:
            lines.append("\n### Tool Failure Rates")
            for tool, rate in sorted(tool_rates.items(), key=lambda x: -x[1]):
                lines.append(f"  - {tool}: {rate*100:.1f}% failure rate")

        worst = self.get_worst_tools()
        if worst:
            lines.append(f"\n### Underperforming Tools: {', '.join(worst)}")
            lines.append("  Consider using alternative tools for these categories.")

        return "\n".join(lines)

    def _extract_pattern(self, entry: dict) -> str | None:
        """Extract a short pattern key from a failure entry."""
        error = entry.get("error", "").lower()
        tool = entry.get("tool_used", "")

        for keyword in ["timeout", "rate limit", "connection", "auth", "parse", "empty"]:
            if keyword in error:
                return f"{tool}:{keyword}" if tool else keyword

        if entry.get("verdict") == "FAIL" and not error:
            return f"{tool}:low_quality" if tool else "low_quality"

        return None

    def summary(self) -> dict[str, Any]:
        return {
            "total_failures": len(self._failures),
            "top_patterns": self.get_top_patterns(3),
            "tool_failure_rates": self.get_tool_failure_rates(),
        }
