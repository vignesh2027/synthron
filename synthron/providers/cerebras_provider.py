"""Cerebras provider — 2100 tokens/second, 1M free tokens/day."""

from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI
from openai import RateLimitError as OAIRateLimitError
from openai import AuthenticationError as OAIAuthError

from synthron.providers.base_provider import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
    RequestTimer,
)
from synthron.utils.config import settings
from synthron.utils.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from synthron.utils.logger import get_logger
from synthron.utils.rate_limiter import rate_registry
from synthron.utils.token_counter import count_tokens, daily_tracker

logger = get_logger(__name__)


class CerebrasProvider(BaseProvider):
    """Cerebras inference provider — world's fastest LLM inference at 2100+ tok/s.

    Uses the OpenAI-compatible API with a custom base URL.
    Default model: llama-3.3-70b (fastest 70B available).
    """

    name = "cerebras"
    default_model = "llama-3.3-70b"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        super().__init__()
        self._api_key = api_key or settings.providers.cerebras_api_key
        if not self._api_key:
            raise ProviderAuthError("cerebras")
        self.default_model = model or self.default_model
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=settings.providers.cerebras_base_url,
        )
        logger.debug(f"CerebrasProvider initialized with model '{self.default_model}'")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_context_tokens=8192,
            supports_streaming=True,
            supports_function_calling=False,
            supports_vision=False,
            supports_embeddings=False,
            speed_tier=1,  # fastest available
            quality_tier=2,
        )

    def _build_messages(self, request: GenerationRequest) -> list[dict]:
        msgs = []
        if request.system_prompt:
            msgs.append({"role": "system", "content": request.system_prompt})
        for m in request.messages:
            msgs.append(m.to_dict())
        return msgs

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion via Cerebras.

        Args:
            request: Unified generation request.

        Returns:
            GenerationResponse with ultra-fast completion.
        """
        estimated_tokens = sum(count_tokens(m.content) for m in request.messages)
        await rate_registry.acquire("cerebras", estimated_tokens)

        model_name = self.get_model(request.model)
        messages = self._build_messages(request)

        try:
            with RequestTimer() as timer:
                completion = await self._client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=min(request.max_tokens, 8192),
                )

            choice = completion.choices[0]
            text = choice.message.content or ""
            usage = completion.usage

            input_toks = usage.prompt_tokens if usage else estimated_tokens
            output_toks = usage.completion_tokens if usage else count_tokens(text)
            total_toks = input_toks + output_toks

            daily_tracker.add("cerebras", total_toks)
            rate_registry.get("cerebras").record_response(total_toks)

            resp = GenerationResponse(
                content=text,
                model=model_name,
                provider="cerebras",
                input_tokens=input_toks,
                output_tokens=output_toks,
                total_tokens=total_toks,
                latency_ms=timer.elapsed_ms,
                finish_reason=choice.finish_reason or "stop",
            )
            self._record_success(resp)
            logger.debug(
                f"[cerebras] {total_toks} tokens | {timer.elapsed_ms:.0f}ms | {model_name}"
            )
            return resp

        except OAIRateLimitError as exc:
            self._record_error()
            raise RateLimitError("cerebras") from exc
        except OAIAuthError as exc:
            self._record_error()
            raise ProviderAuthError("cerebras") from exc
        except Exception as exc:
            self._record_error()
            msg = str(exc).lower()
            if "503" in msg or "unavailable" in msg:
                raise ProviderUnavailableError("cerebras", str(exc)) from exc
            raise ProviderError(f"Cerebras error: {exc}", provider="cerebras") from exc

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream Cerebras completions.

        Args:
            request: Unified generation request.

        Yields:
            Token chunks as strings.
        """
        model_name = self.get_model(request.model)
        messages = self._build_messages(request)

        try:
            stream = await self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=min(request.max_tokens, 8192),
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as exc:
            self._record_error()
            raise ProviderError(f"Cerebras stream error: {exc}", provider="cerebras") from exc
