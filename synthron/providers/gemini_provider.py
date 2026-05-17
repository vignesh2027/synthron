"""Google Gemini provider — using new google-genai SDK (google.generativeai is deprecated)."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

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

# Model fallback chain — newest first
_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


def _build_prompt(request: GenerationRequest) -> str:
    """Build a single prompt string from request messages."""
    parts = []
    if request.system_prompt:
        parts.append(f"System: {request.system_prompt}")
    for m in request.messages:
        role = m.role.capitalize()
        parts.append(f"{role}: {m.content}")
    return "\n\n".join(parts)


class GeminiProvider(BaseProvider):
    """Google Gemini provider using the new google-genai SDK."""

    name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        super().__init__()
        self._api_key = api_key or settings.providers.gemini_api_key
        if not self._api_key:
            raise ProviderAuthError("gemini")
        self.default_model = model or self.default_model
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Try new SDK first, fall back to old SDK."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            self._sdk = "new"
            logger.debug("GeminiProvider using new google-genai SDK")
        except ImportError:
            try:
                import google.generativeai as genai_old
                genai_old.configure(api_key=self._api_key)
                self._genai_old = genai_old
                self._sdk = "old"
                logger.debug("GeminiProvider using legacy google-generativeai SDK")
            except ImportError:
                raise ProviderAuthError("gemini")

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

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        estimated_tokens = sum(count_tokens(m.content, "gemini") for m in request.messages)
        await rate_registry.acquire("gemini", estimated_tokens)

        prompt = _build_prompt(request)
        last_error = None

        for model_name in _GEMINI_MODELS:
            try:
                with RequestTimer() as timer:
                    text = await self._call(model_name, prompt, request)

                input_toks = estimated_tokens
                output_toks = count_tokens(text, "gemini")
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
                    finish_reason="stop",
                )
                self._record_success(resp)
                logger.debug(f"[gemini] {total_toks} tokens | {timer.elapsed_ms:.0f}ms | {model_name}")
                return resp

            except (RateLimitError, ProviderAuthError):
                raise
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                if "quota" in msg or "rate" in msg or "429" in msg:
                    raise RateLimitError("gemini") from exc
                if "api key" in msg or "auth" in msg or "403" in msg:
                    raise ProviderAuthError("gemini") from exc
                logger.debug(f"[gemini] model {model_name} failed: {exc}, trying next")
                continue

        self._record_error()
        raise ProviderError(f"All Gemini models failed: {last_error}", provider="gemini") from last_error

    async def _call(self, model_name: str, prompt: str, request: GenerationRequest) -> str:
        """Call Gemini with new or old SDK."""
        if self._sdk == "new":
            from google.genai import types as gtypes
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                ),
            )
            return response.text or ""
        else:
            client = self._genai_old.GenerativeModel(
                model_name,
                system_instruction=request.system_prompt or None,
            )
            cfg = self._genai_old.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )
            content = request.messages[-1].content if request.messages else prompt
            response = await asyncio.to_thread(client.generate_content, content, generation_config=cfg)
            return response.text or ""

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        prompt = _build_prompt(request)
        for model_name in _GEMINI_MODELS:
            try:
                if self._sdk == "new":
                    from google.genai import types as gtypes
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=model_name,
                        contents=prompt,
                        config=gtypes.GenerateContentConfig(
                            temperature=request.temperature,
                            max_output_tokens=request.max_tokens,
                        ),
                    )
                    if response.text:
                        yield response.text
                else:
                    client = self._genai_old.GenerativeModel(model_name)
                    cfg = self._genai_old.GenerationConfig(
                        temperature=request.temperature,
                        max_output_tokens=request.max_tokens,
                    )
                    content = request.messages[-1].content if request.messages else ""
                    stream = await asyncio.to_thread(
                        client.generate_content, content, generation_config=cfg, stream=True
                    )
                    for chunk in stream:
                        if chunk.text:
                            yield chunk.text
                return
            except Exception as exc:
                logger.debug(f"[gemini] stream model {model_name} failed: {exc}")
                continue

    async def embed(self, text: str) -> list[float]:
        try:
            if self._sdk == "new":
                result = await asyncio.to_thread(
                    self._client.models.embed_content,
                    model="text-embedding-004",
                    contents=text,
                )
                return result.embeddings[0].values
            else:
                result = await asyncio.to_thread(
                    self._genai_old.embed_content,
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document",
                )
                return result["embedding"]
        except Exception as exc:
            raise ProviderError(f"Gemini embed failed: {exc}", provider="gemini") from exc
