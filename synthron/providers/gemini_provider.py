"""Google Gemini 2.5 Flash provider — 33M free tokens/day."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

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

_GEMINI_MODELS = {
    "flash": "gemini-2.5-flash",
    "pro": "gemini-2.0-pro",
    "flash-8b": "gemini-2.0-flash-8b",
    "embed": "models/embedding-001",
}


class GeminiProvider(BaseProvider):
    """Google Gemini provider using the google-generativeai SDK.

    Supports:
    - gemini-2.5-flash (default, 33M tokens/day free)
    - Text generation (sync + async streaming)
    - Text embeddings
    - Vision (multimodal inputs)
    """

    name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        super().__init__()
        self._api_key = api_key or settings.providers.gemini_api_key
        if not self._api_key:
            raise ProviderAuthError("gemini")
        genai.configure(api_key=self._api_key)
        self.default_model = model or self.default_model
        self._client = genai.GenerativeModel(self.default_model)
        logger.debug(f"GeminiProvider initialized with model '{self.default_model}'")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_context_tokens=1_000_000,
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            supports_embeddings=True,
            speed_tier=2,
            quality_tier=1,
        )

    def _build_gemini_history(
        self, messages: list, system_prompt: str
    ) -> tuple[list[dict], str | None]:
        """Convert GenerationRequest messages to Gemini chat history format."""
        history = []
        last_user_msg = ""

        # Gemini uses "user" and "model" roles
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            if msg == messages[-1] and role == "user":
                last_user_msg = msg.content
                continue
            history.append({"role": role, "parts": [msg.content]})

        return history, last_user_msg

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion using Gemini.

        Args:
            request: Generation request with messages and config.

        Returns:
            GenerationResponse with content and usage metadata.
        """
        estimated_tokens = sum(count_tokens(m.content, "gemini") for m in request.messages)
        await rate_registry.acquire("gemini", estimated_tokens)

        model_name = self.get_model(request.model)
        client = genai.GenerativeModel(
            model_name,
            system_instruction=request.system_prompt or None,
        )

        generation_config = genai.GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences or None,
        )

        history, last_user_msg = self._build_gemini_history(
            request.messages, request.system_prompt
        )

        try:
            with RequestTimer() as timer:
                if history:
                    chat = client.start_chat(history=history)
                    response: GenerateContentResponse = await asyncio.to_thread(
                        chat.send_message,
                        last_user_msg or request.messages[-1].content,
                        generation_config=generation_config,
                    )
                else:
                    content = request.messages[-1].content if request.messages else ""
                    response = await asyncio.to_thread(
                        client.generate_content,
                        content,
                        generation_config=generation_config,
                    )

            text = response.text or ""
            usage = response.usage_metadata

            input_toks = getattr(usage, "prompt_token_count", estimated_tokens)
            output_toks = getattr(usage, "candidates_token_count", count_tokens(text, "gemini"))
            total_toks = input_toks + output_toks

            daily_tracker.add("gemini", total_toks)
            rate_registry.get("gemini").record_response(total_toks)

            resp = GenerationResponse(
                content=text,
                model=model_name,
                provider="gemini",
                input_tokens=input_toks,
                output_tokens=output_toks,
                total_tokens=total_toks,
                latency_ms=timer.elapsed_ms,
                finish_reason=str(
                    response.candidates[0].finish_reason if response.candidates else "stop"
                ),
            )
            self._record_success(resp)
            logger.debug(
                f"[gemini] {total_toks} tokens | {timer.elapsed_ms:.0f}ms | {model_name}"
            )
            return resp

        except Exception as exc:
            self._record_error()
            self._handle_exception(exc)

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream tokens from Gemini in real time.

        Args:
            request: Generation request.

        Yields:
            Text chunks as strings.
        """
        model_name = self.get_model(request.model)
        client = genai.GenerativeModel(
            model_name,
            system_instruction=request.system_prompt or None,
        )
        generation_config = genai.GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )

        content = request.messages[-1].content if request.messages else ""

        try:
            stream = await asyncio.to_thread(
                client.generate_content,
                content,
                generation_config=generation_config,
                stream=True,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            self._record_error()
            self._handle_exception(exc)

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector using Gemini embedding model.

        Args:
            text: Input text to embed.

        Returns:
            List of floats (embedding vector).
        """
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=_GEMINI_MODELS["embed"],
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        except Exception as exc:
            self._handle_exception(exc)

    def _handle_exception(self, exc: Exception) -> None:
        """Map google-generativeai exceptions to Synthron exceptions."""
        msg = str(exc).lower()

        if "quota" in msg or "rate" in msg or "429" in msg:
            raise RateLimitError("gemini") from exc
        elif "api key" in msg or "auth" in msg or "403" in msg:
            raise ProviderAuthError("gemini") from exc
        elif "context" in msg or "tokens" in msg or "400" in msg:
            raise TokenLimitError("gemini", 0, 1_000_000) from exc
        elif "unavailable" in msg or "503" in msg or "500" in msg:
            raise ProviderUnavailableError("gemini", str(exc)) from exc
        else:
            raise ProviderError(f"Gemini error: {exc}", provider="gemini") from exc
