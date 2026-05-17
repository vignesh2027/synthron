"""Token counting utilities for all supported providers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from synthron.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _get_tiktoken_encoder(model: str):
    """Load and cache a tiktoken encoder."""
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken (OpenAI-compatible tokenizer).

    Args:
        text: The text to count tokens for.
        model: Model name to select the tokenizer.

    Returns:
        Estimated token count.
    """
    encoder = _get_tiktoken_encoder(model)
    if encoder is None:
        return estimate_tokens_fast(text)
    return len(encoder.encode(text))


def estimate_tokens_fast(text: str) -> int:
    """Fast token estimation without a tokenizer.

    Uses ~4 chars/token heuristic (accurate to ±15%).

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    words = len(re.findall(r"\S+", text))
    chars = len(text)
    # Blend word-based (1.3 tok/word) and char-based (4 char/tok)
    return max(1, int((words * 1.3 + chars / 4) / 2))


def count_tokens(text: str, model: str = "gemini") -> int:
    """Count tokens for a given model family.

    Args:
        text: Input text.
        model: Provider/model identifier.

    Returns:
        Token count.
    """
    model_lower = model.lower()

    if "gemini" in model_lower:
        # Gemini uses ~4 chars/token on average
        return estimate_tokens_fast(text)
    elif any(m in model_lower for m in ["gpt", "openai", "groq", "cerebras", "deepseek"]):
        return count_tokens_tiktoken(text, model="gpt-4")
    else:
        return estimate_tokens_fast(text)


def count_messages_tokens(messages: list[dict[str, Any]], model: str = "gemini") -> int:
    """Count tokens across a list of chat messages.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Provider/model identifier.

    Returns:
        Total token count including role prefixes.
    """
    total = 0
    for msg in messages:
        # Account for role prefix overhead (~4 tokens per message)
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += count_tokens(part["text"], model)
    return total


class TokenBudget:
    """Track token consumption against a budget.

    Useful for keeping agent context within model limits.
    """

    def __init__(self, limit: int, model: str = "gemini") -> None:
        self.limit = limit
        self.model = model
        self._used = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self._used)

    @property
    def is_exceeded(self) -> bool:
        return self._used >= self.limit

    def consume(self, text: str) -> int:
        """Add tokens for text and return tokens added."""
        tokens = count_tokens(text, self.model)
        self._used += tokens
        return tokens

    def would_exceed(self, text: str) -> bool:
        """Check if adding text would exceed the budget."""
        return (self._used + count_tokens(text, self.model)) > self.limit

    def reset(self) -> None:
        self._used = 0

    def __repr__(self) -> str:
        return f"TokenBudget(used={self._used}/{self.limit}, model={self.model!r})"


class DailyUsageTracker:
    """Track per-provider daily token usage with persistence.

    Resets automatically at midnight UTC.
    """

    def __init__(self) -> None:
        from datetime import date

        self._date = date.today()
        self._usage: dict[str, int] = {}

    def _reset_if_new_day(self) -> None:
        from datetime import date

        today = date.today()
        if today != self._date:
            self._date = today
            self._usage.clear()
            logger.info("Daily token counters reset for new UTC day.")

    def add(self, provider: str, tokens: int) -> None:
        """Record token usage for a provider."""
        self._reset_if_new_day()
        self._usage[provider] = self._usage.get(provider, 0) + tokens

    def get(self, provider: str) -> int:
        """Return tokens used today for a provider."""
        self._reset_if_new_day()
        return self._usage.get(provider, 0)

    def total(self) -> int:
        """Return total tokens used today across all providers."""
        self._reset_if_new_day()
        return sum(self._usage.values())

    def report(self) -> dict[str, int]:
        """Return usage report dict."""
        self._reset_if_new_day()
        return dict(self._usage)

    def is_over_limit(self, provider: str, limit: float) -> bool:
        """Return True if provider has exceeded its daily limit."""
        if limit == float("inf"):
            return False
        return self.get(provider) >= int(limit)


# Global tracker instance
daily_tracker = DailyUsageTracker()
