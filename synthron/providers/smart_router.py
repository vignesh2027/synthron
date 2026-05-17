"""Synthron Smart Router — 35M+ free tokens/day with auto-failover."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, TYPE_CHECKING

from synthron.providers.base_provider import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
)
from synthron.utils.config import settings
from synthron.utils.exceptions import (
    AllProvidersExhaustedError,
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from synthron.utils.logger import get_logger
from synthron.utils.rate_limiter import rate_registry
from synthron.utils.token_counter import daily_tracker

logger = get_logger(__name__)

# Lazy imports to avoid loading all SDKs at startup
def _load_provider(name: str, **kwargs) -> BaseProvider | None:
    """Dynamically load and instantiate a provider by name."""
    try:
        if name == "gemini":
            from synthron.providers.gemini_provider import GeminiProvider
            return GeminiProvider(**kwargs)
        elif name == "groq":
            from synthron.providers.groq_provider import GroqProvider
            return GroqProvider(**kwargs)
        elif name == "cerebras":
            from synthron.providers.cerebras_provider import CerebrasProvider
            return CerebrasProvider(**kwargs)
        elif name == "deepseek":
            from synthron.providers.deepseek_provider import DeepSeekProvider
            return DeepSeekProvider(**kwargs)
        elif name == "openrouter":
            from synthron.providers.openrouter_provider import OpenRouterProvider
            return OpenRouterProvider(**kwargs)
        elif name == "github":
            from synthron.providers.github_provider import GitHubModelsProvider
            return GitHubModelsProvider(**kwargs)
        elif name == "ollama":
            from synthron.providers.ollama_provider import OllamaProvider
            return OllamaProvider(**kwargs)
        else:
            logger.warning(f"Unknown provider name: '{name}'")
            return None
    except ProviderAuthError:
        logger.debug(f"Provider '{name}' skipped — no API key configured.")
        return None
    except Exception as exc:
        logger.warning(f"Provider '{name}' failed to load: {exc}")
        return None


class SynthronRouter:
    """Intelligent LLM router with automatic failover and daily budget tracking.

    Priority order (descending):
        gemini → cerebras → groq → deepseek → openrouter → github → ollama

    Routing policy per agent type:
        planner    → gemini, deepseek, groq, openrouter, ollama
        executor   → cerebras, groq, gemini, openrouter, ollama
        critic     → deepseek, gemini, groq, openrouter, ollama
        memory     → groq, gemini, openrouter, ollama
        researcher → gemini, openrouter, groq, ollama
        coder      → gemini, deepseek, groq, openrouter, ollama

    Failover triggers:
        - RateLimitError → skip to next provider
        - ProviderUnavailableError → skip to next provider
        - ProviderAuthError → skip (key not configured)
        - AllProvidersExhaustedError → raise final error
    """

    DAILY_LIMITS = settings.router.daily_limits
    AGENT_ROUTING = settings.router.agent_routing

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._failed_providers: set[str] = set()
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Load and validate all configured providers."""
        if self._initialized:
            return

        available = settings.providers.available_providers()
        logger.info(f"[router] Initializing providers: {available}")

        for name in available:
            provider = _load_provider(name)
            if provider and provider.is_available():
                self._providers[name] = provider
                logger.info(f"[router] ✅ Provider ready: {name}")
            else:
                logger.debug(f"[router] ⚠️  Provider unavailable: {name}")

        if not self._providers:
            logger.error(
                "[router] No providers configured! Set at least one API key in .env"
            )
        else:
            logger.info(
                f"[router] {len(self._providers)} provider(s) active: "
                f"{list(self._providers.keys())}"
            )

        self._initialized = True

    def _get_chain(self, agent_type: str = "default") -> list[str]:
        """Return ordered provider chain for the given agent type."""
        chain = self.AGENT_ROUTING.get(agent_type, self.AGENT_ROUTING["default"])
        # Filter to only providers that are loaded and not permanently failed
        return [
            p for p in chain
            if p in self._providers and p not in self._failed_providers
        ]

    def _is_budget_ok(self, provider: str) -> bool:
        """Return True if the provider has remaining daily budget."""
        limit = self.DAILY_LIMITS.get(provider, float("inf"))
        if limit == float("inf"):
            return True
        used = daily_tracker.get(provider)
        ok = used < limit
        if not ok:
            logger.warning(
                f"[router] '{provider}' daily budget exhausted "
                f"({used:,}/{int(limit):,} tokens)"
            )
        return ok

    async def get_provider(
        self, agent_type: str = "default", complexity: int = 5
    ) -> BaseProvider:
        """Return the best available provider for the agent type and complexity.

        Args:
            agent_type: Agent role (planner, executor, critic, etc.)
            complexity: Task complexity 1-10. High complexity prefers better models.

        Returns:
            The first available provider in the priority chain.

        Raises:
            AllProvidersExhaustedError: If no provider is available.
        """
        if not self._initialized:
            await self.initialize()

        chain = self._get_chain(agent_type)

        # High complexity → prefer quality-tier-1 providers
        if complexity >= 8:
            quality_first = ["gemini", "deepseek", "groq"]
            chain = sorted(chain, key=lambda p: (0 if p in quality_first else 1))

        for name in chain:
            if not self._is_budget_ok(name):
                continue
            if rate_registry.is_exhausted(name):
                continue
            logger.debug(f"[router] Selected provider '{name}' for agent_type='{agent_type}'")
            return self._providers[name]

        raise AllProvidersExhaustedError(chain or list(self._providers.keys()))

    async def generate(
        self,
        request: GenerationRequest,
        agent_type: str = "default",
        complexity: int = 5,
    ) -> GenerationResponse:
        """Generate with automatic failover across the provider chain.

        Args:
            request: Generation request to execute.
            agent_type: Agent role for routing hints.
            complexity: Task complexity for model selection.

        Returns:
            GenerationResponse from the first successful provider.

        Raises:
            AllProvidersExhaustedError: If every provider in the chain fails.
        """
        if not self._initialized:
            await self.initialize()

        chain = self._get_chain(agent_type)
        attempted: list[str] = []
        last_exc: Exception | None = None

        for provider_name in chain:
            if not self._is_budget_ok(provider_name):
                continue

            provider = self._providers[provider_name]
            attempted.append(provider_name)

            try:
                logger.debug(
                    f"[router] Trying '{provider_name}' for '{agent_type}' "
                    f"(complexity={complexity})"
                )
                response = await provider.generate(request)
                if attempted[0] != provider_name:
                    logger.info(
                        f"[router] Failover success: used '{provider_name}' "
                        f"(after: {attempted[:-1]})"
                    )
                return response

            except RateLimitError as exc:
                logger.warning(f"[router] '{provider_name}' rate limited → trying next")
                last_exc = exc

            except ProviderUnavailableError as exc:
                logger.warning(f"[router] '{provider_name}' unavailable → trying next")
                last_exc = exc

            except ProviderAuthError as exc:
                logger.error(f"[router] '{provider_name}' auth failed — removing from pool")
                self._failed_providers.add(provider_name)
                last_exc = exc

            except ProviderError as exc:
                logger.warning(f"[router] '{provider_name}' error: {exc} → trying next")
                last_exc = exc

            except Exception as exc:
                logger.warning(f"[router] '{provider_name}' unexpected error: {exc}")
                last_exc = exc

        raise AllProvidersExhaustedError(attempted) from last_exc

    async def generate_stream(
        self,
        request: GenerationRequest,
        agent_type: str = "default",
    ) -> AsyncIterator[str]:
        """Stream tokens with automatic provider failover.

        Args:
            request: Generation request.
            agent_type: Agent role for routing.

        Yields:
            Token chunks as strings.
        """
        if not self._initialized:
            await self.initialize()

        chain = self._get_chain(agent_type)
        attempted: list[str] = []

        for provider_name in chain:
            if not self._is_budget_ok(provider_name):
                continue

            provider = self._providers[provider_name]
            attempted.append(provider_name)

            try:
                async for chunk in provider.generate_stream(request):
                    yield chunk
                return  # stream completed successfully
            except (RateLimitError, ProviderUnavailableError) as exc:
                logger.warning(f"[router] Stream: '{provider_name}' failed → trying next")
                continue
            except Exception as exc:
                logger.warning(f"[router] Stream error from '{provider_name}': {exc}")
                continue

        raise AllProvidersExhaustedError(attempted)

    def status(self) -> dict:
        """Return router health status and usage summary."""
        return {
            "initialized": self._initialized,
            "active_providers": list(self._providers.keys()),
            "failed_providers": list(self._failed_providers),
            "daily_usage": daily_tracker.report(),
            "daily_limits": {
                k: v for k, v in self.DAILY_LIMITS.items() if k in self._providers
            },
            "rate_limiter_stats": rate_registry.all_stats(),
        }

    def provider_stats(self) -> list[dict]:
        """Return per-provider statistics."""
        return [p.stats for p in self._providers.values()]


# Global singleton router
router = SynthronRouter()
