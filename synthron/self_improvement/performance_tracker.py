"""Performance Tracker — track improvements over time across all runs."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """Tracks Synthron performance metrics across all task runs.

    Metrics tracked:
    - Success rate over time
    - Average tokens per task (efficiency)
    - Average retry count (quality)
    - Task completion time
    - Critic score trends
    """

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._runs: deque[dict[str, Any]] = deque(maxlen=window)
        self._total_runs = 0
        self._total_successes = 0
        self._total_tokens = 0
        self._total_time_s = 0.0
        self._total_retries = 0
        self._start_time = time.time()

    async def record(
        self,
        task: str,
        success: bool,
        tokens: int,
        time_s: float,
        retry_count: int = 0,
        scores: list[float] | None = None,
    ) -> None:
        """Record metrics for a completed task run.

        Args:
            task: Task description.
            success: Whether the task succeeded.
            tokens: Total tokens consumed.
            time_s: Total execution time in seconds.
            retry_count: Number of critic retries.
            scores: List of critic scores from the run.
        """
        avg_score = sum(scores) / len(scores) if scores else 0.0

        entry = {
            "ts": time.time(),
            "task": task[:80],
            "success": success,
            "tokens": tokens,
            "time_s": round(time_s, 2),
            "retries": retry_count,
            "avg_score": round(avg_score, 3),
        }
        self._runs.append(entry)

        # Cumulative stats
        self._total_runs += 1
        if success:
            self._total_successes += 1
        self._total_tokens += tokens
        self._total_time_s += time_s
        self._total_retries += retry_count

        logger.debug(
            f"[perf] Run #{self._total_runs}: success={success} "
            f"tokens={tokens:,} time={time_s:.1f}s retries={retry_count}"
        )

    def get_current_metrics(self) -> dict[str, Any]:
        """Return current rolling-window performance metrics."""
        window_runs = list(self._runs)
        if not window_runs:
            return {"message": "No runs recorded yet."}

        n = len(window_runs)
        successes = sum(1 for r in window_runs if r["success"])
        avg_tokens = sum(r["tokens"] for r in window_runs) / n
        avg_time = sum(r["time_s"] for r in window_runs) / n
        avg_retries = sum(r["retries"] for r in window_runs) / n
        avg_score = sum(r["avg_score"] for r in window_runs if r["avg_score"] > 0) / max(n, 1)

        return {
            "window_size": n,
            "success_rate": round(successes / n, 3),
            "avg_tokens": round(avg_tokens, 0),
            "avg_time_s": round(avg_time, 2),
            "avg_retry_count": round(avg_retries, 2),
            "avg_critic_score": round(avg_score, 3),
        }

    def get_cumulative_metrics(self) -> dict[str, Any]:
        """Return all-time cumulative metrics."""
        uptime = time.time() - self._start_time
        return {
            "total_runs": self._total_runs,
            "total_successes": self._total_successes,
            "success_rate": (
                round(self._total_successes / self._total_runs, 3)
                if self._total_runs else 0.0
            ),
            "total_tokens": self._total_tokens,
            "total_time_s": round(self._total_time_s, 1),
            "total_retries": self._total_retries,
            "uptime_s": round(uptime, 0),
            "tasks_per_hour": (
                round(self._total_runs / (uptime / 3600), 1) if uptime > 0 else 0
            ),
        }

    def get_trend(self, metric: str = "success_rate", window: int = 10) -> list[float]:
        """Return trend data for a metric over recent runs.

        Args:
            metric: One of: success_rate, tokens, time_s, retries, avg_score
            window: Number of recent runs to include.

        Returns:
            List of metric values over the window.
        """
        runs = list(self._runs)[-window:]
        if metric == "success_rate":
            return [1.0 if r["success"] else 0.0 for r in runs]
        elif metric in ("tokens", "time_s", "retries", "avg_score"):
            return [r.get(metric, 0.0) for r in runs]
        return []

    def is_improving(self) -> bool:
        """Return True if success rate is trending upward."""
        trend = self.get_trend("success_rate", window=20)
        if len(trend) < 10:
            return False
        first_half = sum(trend[:len(trend)//2]) / (len(trend)//2)
        second_half = sum(trend[len(trend)//2:]) / (len(trend) - len(trend)//2)
        return second_half > first_half

    def generate_report(self) -> str:
        """Generate a formatted performance report."""
        current = self.get_current_metrics()
        cumulative = self.get_cumulative_metrics()
        improving = self.is_improving()

        lines = [
            "## Synthron Performance Report",
            "",
            f"### Cumulative Stats (all time)",
            f"  Total runs:     {cumulative['total_runs']}",
            f"  Success rate:   {cumulative['success_rate']*100:.1f}%",
            f"  Total tokens:   {cumulative['total_tokens']:,}",
            f"  Avg time/task:  {round(cumulative['total_time_s'] / max(cumulative['total_runs'], 1), 1)}s",
            f"  Total retries:  {cumulative['total_retries']}",
            "",
            f"### Rolling Window (last {current.get('window_size', 0)} runs)",
            f"  Success rate:   {current.get('success_rate', 0)*100:.1f}%",
            f"  Avg tokens:     {current.get('avg_tokens', 0):,.0f}",
            f"  Avg time:       {current.get('avg_time_s', 0):.1f}s",
            f"  Avg retries:    {current.get('avg_retry_count', 0):.2f}",
            f"  Avg critic:     {current.get('avg_critic_score', 0):.3f}",
            "",
            f"### Trend: {'📈 Improving' if improving else '📊 Stable / needs attention'}",
        ]
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "current": self.get_current_metrics(),
            "cumulative": self.get_cumulative_metrics(),
            "improving": self.is_improving(),
        }
