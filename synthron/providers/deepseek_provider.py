"""DeepSeek V3.2 provider — best reasoning, generous free credits."""

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


class DeepSeekProvider(BaseProvider):
    """DeepSeek V3.2 provider — strongest reasoning model in the free tier.

    Uses the OpenAI-compatible DeepSeek API.
    Assigned to CriticAgent due to superior analytical ability.
    """

    name = "deepseek"
    default_model = "deepseek-chat"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        super().__init__()
        self._api_key = api_key or settings.providers.deepseek_api_key
        if not self._api_key:
            raise ProviderAuthError("deepseek")
        self.default_model = model or self.default_model
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=settings.providers.deepseek_base_url,
        )
        logger.debug(f"DeepSeekProvider initialized with model '{self.default_model}'")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_context_tokens=128_000,
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=False,
            supports_embeddings=False,
            speed_tier=2,
            quality_tier=1,  # best reasoning quality
        )

    def _build_messages(self, request: GenerationRequest) -> list[dict]:
        msgs = []
        if request.system_prompt:
            msgs.append({"role": "system", "content": request.system_prompt})
        for m in request.messages:
            msgs.append(m.to_dict())
        return msgs

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion via DeepSeek.

        Args:
            request: Unified generation request.

        Returns:
            GenerationResponse with high-quality reasoning output.
        """
        estimated_tokens = sum(count_tokens(m.content) for m in request.messages)
        await rate_registry.acquire("deepseek", estimated_tokens)

        model_name = self.get_model(request.model)
        messages = self._build_messages(request)

        try:
            with RequestTimer() as timer:
                completion = await self._client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stop=request.stop_sequences or None,
                )

            choice = completion.choices[0]
            text = choice.message.content or ""
            usage = completion.usage

            input_toks = usage.prompt_tokens if usage else estimated_tokens
            output_toks = usage.completion_tokens if usage else count_tokens(text)
            total_toks = input_toks + output_toks

            daily_tracker.add("deepseek", total_toks)
            rate_registry.get("deepseek").record_response(total_toks)

            resp = GenerationResponse(
                content=text,
                model=model_name,
                provider="deepseek",
                input_tokens=input_toks,
                output_tokens=output_toks,
                total_tokens=total_toks,
                latency_ms=timer.elapsed_ms,
                finish_reason=choice.finish_reason or "stop",
            )
            self._record_success(resp)
            logger.debug(
                f"[deepseek] {total_toks} tokens | {timer.elapsed_ms:.0f}ms | {model_name}"
            )
            return resp

        except OAIRateLimitError as exc:
            self._record_error()
            raise RateLimitError("deepseek") from exc
        except OAIAuthError as exc:
            self._record_error()
            raise ProviderAuthError("deepseek") from exc
        except Exception as exc:
            self._record_error()
            msg = str(exc).lower()
            if "503" in msg or "unavailable" in msg:
                raise ProviderUnavailableError("deepseek", str(exc)) from exc
            raise ProviderError(f"DeepSeek error: {exc}", provider="deepseek") from exc

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream DeepSeek completions.

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
                max_tokens=request.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as exc:
            self._record_error()
            raise ProviderError(f"DeepSeek stream error: {exc}", provider="deepseek") from exc
