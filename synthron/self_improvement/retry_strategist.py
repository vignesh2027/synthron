"""Retry Strategist — smart retry decisions combining critic, failure, and pattern data."""

from __future__ import annotations

from typing import Any

from synthron.agents.base_agent import CriticScore, SubTask, SubTaskResult
from synthron.utils.config import settings
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class RetryStrategy:
    """Decision object returned by the RetryStrategist."""

    def __init__(
        self,
        should_retry: bool,
        reason: str,
        new_tool: str = "",
        modified_prompt_suffix: str = "",
        delay_s: float = 0.0,
    ) -> None:
        self.should_retry = should_retry
        self.reason = reason
        self.new_tool = new_tool
        self.modified_prompt_suffix = modified_prompt_suffix
        self.delay_s = delay_s

    def __repr__(self) -> str:
        return (
            f"RetryStrategy(retry={self.should_retry}, reason={self.reason!r}, "
            f"new_tool={self.new_tool!r})"
        )


class RetryStrategist:
    """Decides whether and how to retry a failed subtask.

    Combines:
    - Critic score and feedback
    - Error type analysis
    - Tool failure history
    - Attempt count
    - Pattern learner recommendations
    """

    def decide(
        self,
        subtask: SubTask,
        result: SubTaskResult,
        score: CriticScore,
        attempt: int,
        max_retries: int | None = None,
        failure_history: dict[str, Any] | None = None,
    ) -> RetryStrategy:
        """Decide retry strategy for a failed subtask.

        Args:
            subtask: The failing subtask.
            result: Latest execution result.
            score: Latest critic score.
            attempt: Current attempt number (1-indexed).
            max_retries: Maximum allowed retries.
            failure_history: Dict of tool → failure_rate from FailureAnalyzer.

        Returns:
            RetryStrategy with decision and modification hints.
        """
        max_retries = max_retries or settings.agents.max_retries

        # Hard stop: exceeded max retries
        if attempt > max_retries:
            return RetryStrategy(
                should_retry=False,
                reason=f"Max retries ({max_retries}) exceeded",
            )

        # Hard pass: score is good enough
        if score.score >= settings.agents.critic_pass_threshold:
            return RetryStrategy(should_retry=False, reason="Score meets threshold")

        # Score is warn-level: pass with note (no retry)
        if score.score >= settings.agents.critic_warn_threshold and attempt > 1:
            return RetryStrategy(
                should_retry=False,
                reason=f"Score {score.score:.2f} acceptable after {attempt} attempts",
            )

        # Tool failure: try a different tool
        new_tool = self._suggest_alternative_tool(
            subtask.tool_hint, result.error, failure_history
        )

        # Build improvement suffix from critic feedback
        prompt_suffix = self._build_prompt_suffix(score, result)

        # Delay on rate limit or transient errors
        delay = self._compute_delay(result.error, attempt)

        reason = (
            f"Score {score.score:.2f} below threshold {settings.agents.critic_pass_threshold}. "
            f"{score.feedback}"
        )

        logger.debug(
            f"[retry_strategist] Attempt {attempt+1}: {reason}. "
            f"New tool: {new_tool or 'same'}. Delay: {delay:.1f}s"
        )

        return RetryStrategy(
            should_retry=True,
            reason=reason,
            new_tool=new_tool,
            modified_prompt_suffix=prompt_suffix,
            delay_s=delay,
        )

    def _suggest_alternative_tool(
        self,
        current_tool: str,
        error: str,
        failure_history: dict[str, Any] | None,
    ) -> str:
        """Suggest an alternative tool if the current one is failing."""
        if not current_tool:
            return ""

        error_lower = (error or "").lower()

        # Timeout → try faster tool
        if "timeout" in error_lower:
            alternatives = {
                "browser_tool": "web_search",
                "web_search": "api_caller",
            }
            return alternatives.get(current_tool, "")

        # Connection error → try offline tool
        if "connection" in error_lower:
            return "code_executor" if current_tool != "code_executor" else ""

        # Check failure history
        if failure_history and current_tool in failure_history:
            rate = failure_history.get(current_tool, 0)
            if rate > 0.5:
                fallbacks = {
                    "web_search": "browser_tool",
                    "browser_tool": "web_search",
                    "api_caller": "web_search",
                    "code_executor": "calculator",
                }
                return fallbacks.get(current_tool, "")

        return ""

    def _build_prompt_suffix(self, score: CriticScore, result: SubTaskResult) -> str:
        """Build a prompt improvement suffix based on critic feedback."""
        parts = []

        if score.improvement_hint:
            parts.append(f"IMPROVEMENT REQUIRED: {score.improvement_hint}")

        if score.feedback and score.verdict == "FAIL":
            parts.append(f"Previous attempt failed: {score.feedback}")

        if result.error:
            parts.append(f"Previous error: {result.error[:200]}")

        if result.output and len(result.output) < 100:
            parts.append("Previous output was too short. Provide a comprehensive response.")

        return "\n".join(parts) if parts else "Improve the completeness and accuracy."

    def _compute_delay(self, error: str, attempt: int) -> float:
        """Compute exponential backoff delay for retries."""
        error_lower = (error or "").lower()

        if "rate limit" in error_lower:
            return min(2.0 ** attempt, 30.0)  # exponential, max 30s
        elif "timeout" in error_lower:
            return 1.0
        elif "connection" in error_lower:
            return 2.0 * attempt

        return 0.0  # no delay for logic/quality failures
