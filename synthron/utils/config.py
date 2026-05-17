"""Centralized configuration for Synthron using pydantic-settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    """LLM provider API keys and endpoints."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── Free provider keys ────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    github_models_token: str = Field(default="", alias="GITHUB_MODELS_TOKEN")
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")

    # ─── Paid providers (optional) ─────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # ─── Ollama (local) ────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_default_model: str = Field(default="llama3.2", alias="OLLAMA_DEFAULT_MODEL")

    # ─── Provider API base URLs ────────────────────────────────────────────
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    github_models_base_url: str = "https://models.inference.ai.azure.com"

    def available_providers(self) -> list[str]:
        """Return list of providers that have API keys configured."""
        providers = []
        if self.gemini_api_key:
            providers.append("gemini")
        if self.cerebras_api_key:
            providers.append("cerebras")
        if self.groq_api_key:
            providers.append("groq")
        if self.deepseek_api_key:
            providers.append("deepseek")
        if self.openrouter_api_key:
            providers.append("openrouter")
        if self.github_models_token:
            providers.append("github")
        providers.append("ollama")  # always available as last resort
        return providers


class MemorySettings(BaseSettings):
    """Memory subsystem configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    sqlite_url: str = Field(
        default="sqlite+aiosqlite:///./data/synthron.db", alias="SQLITE_URL"
    )
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="synthron", alias="PINECONE_INDEX_NAME")

    short_term_max_messages: int = 20
    long_term_top_k: int = 5
    embedding_model: str = "models/embedding-001"  # Gemini embedding


class DashboardSettings(BaseSettings):
    """Dashboard server configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dashboard_host: str = Field(default="0.0.0.0", alias="DASHBOARD_HOST")
    dashboard_port: int = Field(default=8080, alias="DASHBOARD_PORT")
    dashboard_secret: str = Field(default="synthron_secret", alias="DASHBOARD_SECRET")
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]


class AgentSettings(BaseSettings):
    """Agent behavior configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Critic thresholds
    critic_pass_threshold: float = 0.8
    critic_warn_threshold: float = 0.5
    max_retries: int = 3

    # Timeouts (seconds)
    agent_timeout: float = 120.0
    tool_timeout: float = 30.0
    provider_timeout: float = 60.0

    # Planning
    max_subtasks: int = 8
    min_subtasks: int = 2

    # Concurrency
    max_parallel_executors: int = 3

    # Models per agent type
    planner_model: str = "gemini-2.5-flash"
    executor_model: str = "llama-3.3-70b-versatile"
    critic_model: str = "deepseek-chat"
    memory_model: str = "llama-3.3-70b-versatile"
    researcher_model: str = "gemini-2.5-flash"
    coder_model: str = "gemini-2.5-flash"


class RouterSettings(BaseSettings):
    """Smart router daily token budget configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    daily_limits: dict[str, float] = {
        "gemini": 33_000_000,
        "cerebras": 1_000_000,
        "groq": 1_000_000,
        "deepseek": 1_000_000,
        "openrouter": 1_000_000,
        "github": 500_000,
        "ollama": float("inf"),
    }

    # Routing preferences per agent type (ordered by priority)
    agent_routing: dict[str, list[str]] = {
        "planner": ["gemini", "deepseek", "groq", "openrouter", "ollama"],
        "executor": ["cerebras", "groq", "gemini", "openrouter", "ollama"],
        "critic": ["deepseek", "gemini", "groq", "openrouter", "ollama"],
        "memory": ["groq", "gemini", "openrouter", "ollama"],
        "researcher": ["gemini", "openrouter", "groq", "ollama"],
        "coder": ["gemini", "deepseek", "groq", "openrouter", "ollama"],
        "coordinator": ["gemini", "deepseek", "groq", "openrouter", "ollama"],
        "default": ["gemini", "cerebras", "groq", "deepseek", "openrouter", "github", "ollama"],
    }


class SynthronSettings(BaseSettings):
    """Master settings object aggregating all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="SYNTHRON_LOG_LEVEL"
    )
    env: Literal["development", "production", "testing"] = Field(
        default="development", alias="SYNTHRON_ENV"
    )

    # Sub-configs (instantiated lazily)
    _providers: ProviderSettings | None = None
    _memory: MemorySettings | None = None
    _dashboard: DashboardSettings | None = None
    _agents: AgentSettings | None = None
    _router: RouterSettings | None = None

    @property
    def providers(self) -> ProviderSettings:
        if self._providers is None:
            self._providers = ProviderSettings()
        return self._providers

    @property
    def memory(self) -> MemorySettings:
        if self._memory is None:
            self._memory = MemorySettings()
        return self._memory

    @property
    def dashboard(self) -> DashboardSettings:
        if self._dashboard is None:
            self._dashboard = DashboardSettings()
        return self._dashboard

    @property
    def agents(self) -> AgentSettings:
        if self._agents is None:
            self._agents = AgentSettings()
        return self._agents

    @property
    def router(self) -> RouterSettings:
        if self._router is None:
            self._router = RouterSettings()
        return self._router

    def ensure_data_dirs(self) -> None:
        """Create required data directories."""
        dirs = [
            Path("./data/chroma"),
            Path("./data/episodic"),
            Path("./logs"),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict:
        """Return a sanitized config summary (no secrets)."""
        available = self.providers.available_providers()
        return {
            "env": self.env,
            "log_level": self.log_level,
            "available_providers": available,
            "provider_count": len(available),
            "redis_enabled": bool(self.memory.redis_url),
            "chroma_dir": self.memory.chroma_persist_dir,
        }


@lru_cache(maxsize=1)
def get_settings() -> SynthronSettings:
    """Return cached global settings instance."""
    settings = SynthronSettings()
    settings.ensure_data_dirs()
    return settings


# Convenient module-level access
settings = get_settings()
