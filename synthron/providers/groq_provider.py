"""Groq provider — Llama 3.3 70B, 1M free tokens/day, ultra-fast."""

from __future__ import annotations

from typing import AsyncIterator

from groq import AsyncGroq
from groq import RateLimitError as GroqRateLimitError
from groq import AuthenticationError as GroqAuthError
from groq import BadRequestError as GroqBadRequestError

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
    TokenLimitError,
)
from synthron.utils.logger import get_logger
from synthron.utils.rate_limiter import rate_registry
from synthron.utils.token_counter import count_tokens, daily_tracker

logger = get_logger(__name__)

_GROQ_MODELS = {
    "llama-70b": "llama-3.3-70b-versatile",
    "llama-8b": "llama-3.1-8b-instant",
    "mixtral": "mixtral-8x7b-32768",
    "gemma": "gemma2-9b-it",
}

# Fallback chain — try newest/largest first, work down to reliably available models
_GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


class GroqProvider(BaseProvider):
    """Groq LPU provider using official groq-python SDK.

    Supports:
    - llama-3.3-70b-versatile (default)
    - llama-3.1-8b-instant (ultra-fast)
    - Streaming completions
    """

    name = "groq"
    default_model = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        super().__init__()
        self._api_key = api_key or settings.providers.groq_api_key
        if not self._api_key:
            raise ProviderAuthError("groq")
        self.default_model = model or self.default_model
        self._client = AsyncGroq(api_key=self._api_key)
        logger.debug(f"GroqProvider initialized with model '{self.default_model}'")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_context_tokens=128_000,
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=False,
            supports_embeddings=False,
            speed_tier=1,  # fastest
            quality_tier=2,
        )

    def _build_messages(self, request: GenerationRequest) -> list[dict]:
        """Convert request messages to Groq/OpenAI format, prepending system if set."""
        msgs = []
        if request.system_prompt:
            msgs.append({"role": "system", "content": request.system_prompt})
        for m in request.messages:
            msgs.append(m.to_dict())
        return msgs

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        estimated_tokens = sum(count_tokens(m.content, "groq") for m in request.messages)
        await rate_registry.acquire("groq", estimated_tokens)

        messages = self._build_messages(request)
        last_error: Exception | None = None

        # Try each model in the fallback chain until one works
        models_to_try = _GROQ_FALLBACK_MODELS[:]
        # If caller requested a specific model, try it first
        requested = self.get_model(request.model)
        if requested not in models_to_try:
            models_to_try.insert(0, requested)

        for model_name in models_to_try:
            kwargs: dict = {
                "model": model_name,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": min(request.max_tokens, 32_768),
            }
            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences

            try:
                with RequestTimer() as timer:
                    completion = await self._client.chat.completions.create(**kwargs)

                choice = completion.choices[0]
                text = choice.message.content or ""
                usage = completion.usage

                input_toks = usage.prompt_tokens if usage else estimated_tokens
                output_toks = usage.completion_tokens if usage else count_tokens(text, "groq")
                total_toks = input_toks + output_toks

                daily_tracker.add("groq", total_toks)
                rate_registry.get("groq").record_response(total_toks)

                resp = GenerationResponse(
                    content=text,
                    model=model_name,
                    provider="groq",
                    input_tokens=input_toks,
                    output_tokens=output_toks,
                    total_tokens=total_toks,
                    latency_ms=timer.elapsed_ms,
                    finish_reason=choice.finish_reason or "stop",
                )
                self._record_success(resp)
                logger.debug(f"[groq] {total_toks} tokens | {timer.elapsed_ms:.0f}ms | {model_name}")
                return resp

            except GroqRateLimitError as exc:
                self._record_error()
                raise RateLimitError("groq") from exc
            except GroqAuthError as exc:
                self._record_error()
                raise ProviderAuthError("groq") from exc
            except GroqBadRequestError as exc:
                msg = str(exc).lower()
                if "model" in msg or "not found" in msg or "deprecated" in msg or "does not exist" in msg:
                    logger.debug(f"[groq] model {model_name} unavailable: {exc}, trying next")
                    last_error = exc
                    continue
                if "context" in msg or "token" in msg:
                    self._record_error()
                    raise TokenLimitError("groq", estimated_tokens, 128_000) from exc
                self._record_error()
                raise ProviderError(f"Groq bad request: {exc}", provider="groq") from exc
            except Exception as exc:
                msg = str(exc).lower()
                if "503" in msg or "unavailable" in msg:
                    self._record_error()
                    raise ProviderUnavailableError("groq", str(exc)) from exc
                # Treat other errors as model-level failures and try next
                logger.debug(f"[groq] model {model_name} failed: {exc}, trying next")
                last_error = exc
                continue

        self._record_error()
        raise ProviderError(f"All Groq models failed: {last_error}", provider="groq") from last_error

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream Groq completions token by token.

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
                max_tokens=min(request.max_tokens, 32_768),
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except GroqRateLimitError as exc:
            self._record_error()
            raise RateLimitError("groq") from exc
        except Exception as exc:
            self._record_error()
            raise ProviderError(f"Groq stream error: {exc}", provider="groq") from exc
