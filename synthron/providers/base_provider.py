"""Abstract base class and data models for all LLM providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


# ─── Data Models ──────────────────────────────────────────────────────────────

class Message(BaseModel):
    """A single chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class GenerationRequest(BaseModel):
    """Unified request object for all providers."""

    messages: list[Message]
    model: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    stream: bool = False
    system_prompt: str = ""
    stop_sequences: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def with_system(self, system: str) -> "GenerationRequest":
        """Return a copy with the system prompt set."""
        return self.model_copy(update={"system_prompt": system})

    def add_message(self, role: str, content: str) -> "GenerationRequest":
        """Return a copy with an additional message appended."""
        new_msgs = list(self.messages) + [Message(role=role, content=content)]
        return self.model_copy(update={"messages": new_msgs})


class GenerationResponse(BaseModel):
    """Unified response from any provider."""

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate — $0 for free-tier providers."""
        return 0.0

    def __repr__(self) -> str:
        return (
            f"GenerationResponse(provider={self.provider!r}, model={self.model!r}, "
            f"tokens={self.total_tokens}, latency={self.latency_ms:.0f}ms)"
        )


class ProviderCapabilities(BaseModel):
    """Capabilities metadata for a provider."""

    max_context_tokens: int = 8192
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    is_offline: bool = False
    speed_tier: int = 2  # 1=fastest, 3=slowest
    quality_tier: int = 2  # 1=highest, 3=lowest


# ─── Abstract Base Provider ───────────────────────────────────────────────────

class BaseProvider(ABC):
    """Abstract interface all LLM providers must implement.

    Subclasses must implement `generate` and `generate_stream`.
    All other methods have default implementations.
    """

    name: str = "base"
    default_model: str = ""

    def __init__(self) -> None:
        self._request_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
        self._error_count = 0

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion for the given request.

        Args:
            request: The unified generation request.

        Returns:
            GenerationResponse with content and usage stats.

        Raises:
            ProviderError: On API failure.
            RateLimitError: When rate limited.
            TokenLimitError: When context exceeds model limit.
        """

    @abstractmethod
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Stream completion tokens as they are generated.

        Args:
            request: The unified generation request (stream=True is set automatically).

        Yields:
            String chunks as they arrive from the provider.
        """

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return this provider's capability metadata."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured and reachable."""

    # ── Optional overrides ────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector. Override in providers that support it."""
        raise NotImplementedError(f"Provider '{self.name}' does not support embeddings.")

    async def health_check(self) -> bool:
        """Ping the provider with a minimal request. Returns True if healthy."""
        try:
            req = GenerationRequest(
                messages=[Message(role="user", content="Say OK")],
                max_tokens=5,
            )
            resp = await self.generate(req)
            return bool(resp.content)
        except Exception:
            return False

    def get_model(self, model: str = "") -> str:
        """Return model name, falling back to provider default."""
        return model or self.default_model

    # ── Metrics helpers ────────────────────────────────────────────────────────

    def _record_success(self, response: GenerationResponse) -> None:
        self._request_count += 1
        self._total_tokens += response.total_tokens
        self._total_latency_ms += response.latency_ms

    def _record_error(self) -> None:
        self._error_count += 1

    @property
    def stats(self) -> dict[str, Any]:
        """Return cumulative provider statistics."""
        avg_latency = (
            self._total_latency_ms / self._request_count if self._request_count else 0
        )
        return {
            "provider": self.name,
            "requests": self._request_count,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": round(avg_latency, 1),
            "errors": self._error_count,
            "error_rate": (
                round(self._error_count / self._request_count, 3)
                if self._request_count
                else 0.0
            ),
        }

    # ── Context manager support ────────────────────────────────────────────────

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.default_model!r})"


# ─── Timer context manager ─────────────────────────────────────────────────────

class RequestTimer:
    """Measure elapsed time in milliseconds for a provider request."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "RequestTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
