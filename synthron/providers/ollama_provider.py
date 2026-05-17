"""Ollama provider — unlimited offline local model inference."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import aiohttp

from synthron.providers.base_provider import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
    RequestTimer,
)
from synthron.utils.config import settings
from synthron.utils.exceptions import ProviderError, ProviderUnavailableError
from synthron.utils.logger import get_logger
from synthron.utils.token_counter import count_tokens, daily_tracker

logger = get_logger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama local provider — run any open model offline, unlimited tokens.

    Auto-detected if Ollama is running at localhost:11434.
    Acts as the final fallback in the smart router chain.
    Supports: llama3.2, mistral, codellama, phi3, gemma2, and more.
    """

    name = "ollama"
    default_model = "llama3.2"

    def __init__(self, base_url: str = "", model: str = "") -> None:
        super().__init__()
        self._base_url = (base_url or settings.providers.ollama_base_url).rstrip("/")
        self.default_model = model or settings.providers.ollama_default_model
        self._available: bool | None = None  # cached availability
        logger.debug(
            f"OllamaProvider initialized | base_url={self._base_url} | model={self.default_model}"
        )

    def is_available(self) -> bool:
        """Return cached availability (use async check_available for accuracy)."""
        if self._available is None:
            return True  # optimistic default; health_check will set real value
        return self._available

    async def check_available(self) -> bool:
        """Async check if Ollama is running and reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    self._available = resp.status == 200
                    return self._available
        except Exception:
            self._available = False
            return False

    async def list_models(self) -> list[str]:
        """Return list of locally installed Ollama models."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._base_url}/api/tags") as resp:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_context_tokens=32_768,
            supports_streaming=True,
            supports_function_calling=False,
            supports_vision=False,
            supports_embeddings=True,
            is_offline=True,
            speed_tier=3,  # depends on local hardware
            quality_tier=3,
        )

    def _build_prompt(self, request: GenerationRequest) -> str:
        """Build a plain-text prompt for Ollama's /api/generate endpoint."""
        parts = []
        if request.system_prompt:
            parts.append(f"System: {request.system_prompt}\n")
        for msg in request.messages:
            role = msg.role.capitalize()
            parts.append(f"{role}: {msg.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion using a local Ollama model.

        Args:
            request: Unified generation request.

        Returns:
            GenerationResponse from local inference.

        Raises:
            ProviderUnavailableError: If Ollama is not running.
        """
        model_name = self.get_model(request.model)
        prompt = self._build_prompt(request)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            with RequestTimer() as timer:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self._base_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=settings.agents.agent_timeout),
                    ) as resp:
                        if resp.status == 404:
                            raise ProviderUnavailableError(
                                "ollama", f"Model '{model_name}' not found. Run: ollama pull {model_name}"
                            )
                        resp.raise_for_status()
                        data = await resp.json()

            text = data.get("response", "")
            input_toks = count_tokens(prompt, "ollama")
            output_toks = count_tokens(text, "ollama")
            total_toks = input_toks + output_toks

            daily_tracker.add("ollama", total_toks)

            result = GenerationResponse(
                content=text,
                model=model_name,
                provider="ollama",
                input_tokens=input_toks,
                output_tokens=output_toks,
                total_tokens=total_toks,
                latency_ms=timer.elapsed_ms,
                finish_reason="stop" if data.get("done") else "length",
            )
            self._record_success(result)
            logger.debug(
                f"[ollama] {total_toks} tokens | {timer.elapsed_ms:.0f}ms | {model_name}"
            )
            return result

        except ProviderUnavailableError:
            raise
        except aiohttp.ClientConnectorError as exc:
            self._available = False
            raise ProviderUnavailableError(
                "ollama",
                "Ollama is not running. Start it with: ollama serve",
            ) from exc
        except Exception as exc:
            self._record_error()
            raise ProviderError(f"Ollama error: {exc}", provider="ollama") from exc

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream tokens from local Ollama model.

        Args:
            request: Unified generation request.

        Yields:
            Token chunks as strings.
        """
        model_name = self.get_model(request.model)
        prompt = self._build_prompt(request)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=settings.agents.agent_timeout),
                ) as resp:
                    async for line in resp.content:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except aiohttp.ClientConnectorError as exc:
            self._available = False
            raise ProviderUnavailableError("ollama", "Ollama not running") from exc
        except Exception as exc:
            self._record_error()
            raise ProviderError(f"Ollama stream error: {exc}", provider="ollama") from exc

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding using Ollama's embedding endpoint.

        Args:
            text: Input text.

        Returns:
            Embedding vector as list of floats.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self.default_model, "prompt": text},
                ) as resp:
                    data = await resp.json()
                    return data.get("embedding", [])
        except Exception as exc:
            raise ProviderError(f"Ollama embed error: {exc}", provider="ollama") from exc
