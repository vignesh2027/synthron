"""Tests for Synthron providers."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synthron.providers.base_provider import GenerationRequest, GenerationResponse, Message
from synthron.providers.smart_router import SynthronRouter


class TestGenerationRequest:
    def test_defaults(self):
        req = GenerationRequest(
            messages=[Message(role="user", content="Hello")]
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 4096
        assert req.stream is False

    def test_add_message(self):
        req = GenerationRequest(messages=[Message(role="user", content="Hi")])
        new_req = req.add_message("assistant", "Hello!")
        assert len(new_req.messages) == 2
        assert new_req.messages[-1].content == "Hello!"

    def test_with_system(self):
        req = GenerationRequest(messages=[Message(role="user", content="Hi")])
        new_req = req.with_system("You are a helpful assistant.")
        assert new_req.system_prompt == "You are a helpful assistant."


class TestGenerationResponse:
    def test_repr(self):
        resp = GenerationResponse(
            content="Hello", model="test", provider="test",
            total_tokens=50, latency_ms=100.0,
        )
        assert "test" in repr(resp)
        assert "50" in repr(resp)


class TestSynthronRouter:
    async def test_initialize_with_no_providers(self):
        with patch("synthron.providers.smart_router._load_provider", return_value=None):
            router = SynthronRouter()
            await router.initialize()
            assert router._initialized is True

    async def test_get_chain_returns_list(self):
        router = SynthronRouter()
        router._initialized = True
        # With no providers, chain is empty
        chain = router._get_chain("planner")
        assert isinstance(chain, list)

    def test_budget_check_unlimited(self):
        router = SynthronRouter()
        router.DAILY_LIMITS["test_provider"] = float("inf")
        assert router._is_budget_ok("test_provider") is True

    def test_budget_check_exceeded(self):
        from synthron.utils.token_counter import daily_tracker
        router = SynthronRouter()
        router.DAILY_LIMITS["fake_provider"] = 100

        # Temporarily set usage over limit
        original = daily_tracker._usage.get("fake_provider", 0)
        daily_tracker._usage["fake_provider"] = 200
        result = router._is_budget_ok("fake_provider")
        daily_tracker._usage["fake_provider"] = original

        assert result is False
