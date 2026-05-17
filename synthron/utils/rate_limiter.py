"""Per-provider async rate limiting for Synthron."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from synthron.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProviderLimits:
    """Rate limit configuration for a single provider."""

    requests_per_minute: int = 60
    requests_per_day: int = 10_000
    tokens_per_minute: int = 100_000
    tokens_per_day: int = 1_000_000
    min_request_gap_ms: float = 0.0  # minimum ms between requests


# Default limits per provider
PROVIDER_LIMITS: dict[str, ProviderLimits] = {
    "gemini": ProviderLimits(
        requests_per_minute=60,
        requests_per_day=1_500,
        tokens_per_minute=1_000_000,
        tokens_per_day=33_000_000,
        min_request_gap_ms=100,
    ),
    "groq": ProviderLimits(
        requests_per_minute=30,
        requests_per_day=14_400,
        tokens_per_minute=15_000,
        tokens_per_day=1_000_000,
        min_request_gap_ms=200,
    ),
    "cerebras": ProviderLimits(
        requests_per_minute=30,
        requests_per_day=14_400,
        tokens_per_minute=60_000,
        tokens_per_day=1_000_000,
        min_request_gap_ms=100,
    ),
    "deepseek": ProviderLimits(
        requests_per_minute=60,
        requests_per_day=10_000,
        tokens_per_minute=100_000,
        tokens_per_day=1_000_000,
    ),
    "openrouter": ProviderLimits(
        requests_per_minute=20,
        requests_per_day=200,
        tokens_per_minute=50_000,
        tokens_per_day=1_000_000,
        min_request_gap_ms=500,
    ),
    "github": ProviderLimits(
        requests_per_minute=15,
        requests_per_day=150,
        tokens_per_minute=50_000,
        tokens_per_day=500_000,
        min_request_gap_ms=500,
    ),
    "ollama": ProviderLimits(
        requests_per_minute=1000,
        requests_per_day=1_000_000,
        tokens_per_minute=10_000_000,
        tokens_per_day=1_000_000_000,
    ),
}


class SlidingWindowCounter:
    """Thread-safe sliding window request counter."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = window_seconds
        self._timestamps: deque[float] = deque()

    def add(self) -> None:
        now = time.monotonic()
        self._timestamps.append(now)
        self._evict(now)

    def count(self) -> int:
        self._evict(time.monotonic())
        return len(self._timestamps)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


@dataclass
class ProviderRateLimiter:
    """Async rate limiter for a single provider."""

    provider: str
    limits: ProviderLimits
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _rpm_counter: SlidingWindowCounter = field(
        default_factory=lambda: SlidingWindowCounter(60)
    )
    _tpm_counter: SlidingWindowCounter = field(
        default_factory=lambda: SlidingWindowCounter(60)
    )
    _last_request_ts: float = field(default=0.0)
    _daily_requests: int = field(default=0)
    _daily_tokens: int = field(default=0)

    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Block until a request slot is available.

        Args:
            estimated_tokens: Estimated token cost of the request.

        Raises:
            asyncio.TimeoutError: If waiting too long (>5 min).
        """
        async with self._lock:
            deadline = time.monotonic() + 300  # 5-minute max wait

            while True:
                now = time.monotonic()

                if now > deadline:
                    raise asyncio.TimeoutError(
                        f"Rate limiter for '{self.provider}' timed out after 5 min"
                    )

                # Check daily limits
                if self._daily_requests >= self.limits.requests_per_day:
                    logger.warning(
                        f"[{self.provider}] Daily request limit reached "
                        f"({self._daily_requests}/{self.limits.requests_per_day})"
                    )
                    raise RuntimeError(
                        f"Provider '{self.provider}' daily request limit exhausted"
                    )

                # Check per-minute request rate
                rpm = self._rpm_counter.count()
                if rpm >= self.limits.requests_per_minute:
                    wait = 60.0 / self.limits.requests_per_minute
                    logger.debug(f"[{self.provider}] RPM limit hit, waiting {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue

                # Check minimum gap between requests
                if self.limits.min_request_gap_ms > 0:
                    elapsed_ms = (now - self._last_request_ts) * 1000
                    gap_needed = self.limits.min_request_gap_ms - elapsed_ms
                    if gap_needed > 0:
                        await asyncio.sleep(gap_needed / 1000)
                        continue

                # All checks passed — record and proceed
                self._rpm_counter.add()
                self._last_request_ts = time.monotonic()
                self._daily_requests += 1
                self._daily_tokens += estimated_tokens
                break

    def record_response(self, actual_tokens: int) -> None:
        """Update token counters with actual usage after a response."""
        self._tpm_counter.add()
        # Adjust daily tokens for actual vs estimate (clamped at 0)
        self._daily_tokens = max(0, self._daily_tokens + actual_tokens)

    @property
    def is_daily_limit_reached(self) -> bool:
        return (
            self._daily_requests >= self.limits.requests_per_day
            or self._daily_tokens >= self.limits.tokens_per_day
        )

    def reset_daily(self) -> None:
        """Reset daily counters (call at midnight)."""
        self._daily_requests = 0
        self._daily_tokens = 0

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "daily_requests": self._daily_requests,
            "daily_tokens": self._daily_tokens,
            "rpm": self._rpm_counter.count(),
            "requests_per_day_limit": self.limits.requests_per_day,
            "tokens_per_day_limit": self.limits.tokens_per_day,
        }


class RateLimiterRegistry:
    """Global registry of per-provider rate limiters."""

    def __init__(self) -> None:
        self._limiters: dict[str, ProviderRateLimiter] = {}

    def get(self, provider: str) -> ProviderRateLimiter:
        """Return or create the rate limiter for a provider."""
        if provider not in self._limiters:
            limits = PROVIDER_LIMITS.get(
                provider,
                ProviderLimits(),  # default generous limits for unknown providers
            )
            self._limiters[provider] = ProviderRateLimiter(provider=provider, limits=limits)
        return self._limiters[provider]

    async def acquire(self, provider: str, estimated_tokens: int = 1000) -> None:
        """Acquire a request slot for the given provider."""
        await self.get(provider).acquire(estimated_tokens)

    def is_exhausted(self, provider: str) -> bool:
        """Return True if the provider has reached its daily limit."""
        if provider not in self._limiters:
            return False
        return self._limiters[provider].is_daily_limit_reached

    def all_stats(self) -> dict[str, Any]:
        """Return stats for all tracked providers."""
        return {name: lim.stats() for name, lim in self._limiters.items()}


# Global registry instance
rate_registry = RateLimiterRegistry()
