"""Custom exception hierarchy for Synthron."""

from __future__ import annotations


class SynthronError(Exception):
    """Base exception for all Synthron errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


# ─── Provider Errors ──────────────────────────────────────────────────────────

class ProviderError(SynthronError):
    """Base error for LLM provider failures."""

    def __init__(self, message: str, provider: str, details: dict | None = None) -> None:
        super().__init__(message, details)
        self.provider = provider


class RateLimitError(ProviderError):
    """Provider rate limit exceeded."""

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        super().__init__(
            f"Rate limit exceeded for provider '{provider}'",
            provider=provider,
            details={"retry_after": retry_after},
        )
        self.retry_after = retry_after


class ProviderAuthError(ProviderError):
    """API key missing or invalid."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Authentication failed for provider '{provider}'. Check your API key.",
            provider=provider,
        )


class ProviderUnavailableError(ProviderError):
    """Provider is down or unreachable."""

    def __init__(self, provider: str, reason: str = "") -> None:
        super().__init__(
            f"Provider '{provider}' is unavailable. {reason}".strip(),
            provider=provider,
            details={"reason": reason},
        )


class AllProvidersExhaustedError(SynthronError):
    """All providers in the router failover chain have failed."""

    def __init__(self, attempted: list[str]) -> None:
        super().__init__(
            f"All providers exhausted: {', '.join(attempted)}",
            details={"attempted": attempted},
        )
        self.attempted = attempted


class TokenLimitError(ProviderError):
    """Request exceeds the provider's context window."""

    def __init__(self, provider: str, tokens: int, limit: int) -> None:
        super().__init__(
            f"Token limit exceeded for '{provider}': {tokens} > {limit}",
            provider=provider,
            details={"tokens": tokens, "limit": limit},
        )
        self.tokens = tokens
        self.limit = limit


# ─── Agent Errors ─────────────────────────────────────────────────────────────

class AgentError(SynthronError):
    """Base error for agent-level failures."""

    def __init__(self, message: str, agent_name: str, details: dict | None = None) -> None:
        super().__init__(message, details)
        self.agent_name = agent_name


class AgentTimeoutError(AgentError):
    """Agent exceeded its allowed execution time."""

    def __init__(self, agent_name: str, timeout: float) -> None:
        super().__init__(
            f"Agent '{agent_name}' timed out after {timeout}s",
            agent_name=agent_name,
            details={"timeout": timeout},
        )


class MaxRetriesExceededError(AgentError):
    """Critic rejected the result too many times."""

    def __init__(self, agent_name: str, retries: int, last_score: float) -> None:
        super().__init__(
            f"Agent '{agent_name}' exceeded max retries ({retries}) with score {last_score:.2f}",
            agent_name=agent_name,
            details={"retries": retries, "last_score": last_score},
        )
        self.retries = retries
        self.last_score = last_score


class PlanningError(AgentError):
    """Planner agent failed to decompose the task."""

    def __init__(self, agent_name: str, task: str, reason: str = "") -> None:
        super().__init__(
            f"Planning failed for task: {task[:80]}{'...' if len(task) > 80 else ''}. {reason}".strip(),
            agent_name=agent_name,
            details={"task": task, "reason": reason},
        )


class ExecutionError(AgentError):
    """Executor agent failed to complete a subtask."""

    def __init__(self, agent_name: str, subtask: str, reason: str = "") -> None:
        super().__init__(
            f"Execution failed for subtask '{subtask}'. {reason}".strip(),
            agent_name=agent_name,
            details={"subtask": subtask, "reason": reason},
        )


# ─── Tool Errors ──────────────────────────────────────────────────────────────

class ToolError(SynthronError):
    """Base error for tool failures."""

    def __init__(self, message: str, tool_name: str, details: dict | None = None) -> None:
        super().__init__(message, details)
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Requested tool does not exist in the registry."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"Tool '{tool_name}' not found in registry.",
            tool_name=tool_name,
        )


class ToolExecutionError(ToolError):
    """Tool raised an error during execution."""

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(
            f"Tool '{tool_name}' failed: {reason}",
            tool_name=tool_name,
            details={"reason": reason},
        )


class ToolTimeoutError(ToolError):
    """Tool execution exceeded its timeout."""

    def __init__(self, tool_name: str, timeout: float) -> None:
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout}s",
            tool_name=tool_name,
            details={"timeout": timeout},
        )


class CodeExecutionError(ToolError):
    """Code executor encountered a runtime error."""

    def __init__(self, code: str, error: str) -> None:
        super().__init__(
            f"Code execution failed: {error}",
            tool_name="code_executor",
            details={"code": code[:500], "error": error},
        )


# ─── Memory Errors ────────────────────────────────────────────────────────────

class MemoryError(SynthronError):  # noqa: A001
    """Base error for memory subsystem failures."""


class MemoryWriteError(MemoryError):
    """Failed to persist data to memory store."""

    def __init__(self, store: str, reason: str) -> None:
        super().__init__(
            f"Memory write failed [{store}]: {reason}",
            details={"store": store, "reason": reason},
        )


class MemoryReadError(MemoryError):
    """Failed to retrieve data from memory store."""

    def __init__(self, store: str, reason: str) -> None:
        super().__init__(
            f"Memory read failed [{store}]: {reason}",
            details={"store": store, "reason": reason},
        )


# ─── Orchestrator Errors ──────────────────────────────────────────────────────

class OrchestratorError(SynthronError):
    """Base error for orchestrator failures."""


class WorkflowError(OrchestratorError):
    """DAG workflow encountered an irrecoverable error."""

    def __init__(self, workflow_id: str, reason: str) -> None:
        super().__init__(
            f"Workflow '{workflow_id}' failed: {reason}",
            details={"workflow_id": workflow_id, "reason": reason},
        )


class SessionError(OrchestratorError):
    """Session management failure."""


# ─── Configuration Errors ─────────────────────────────────────────────────────

class ConfigError(SynthronError):
    """Invalid or missing configuration."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(
            f"Configuration error for '{field}': {reason}",
            details={"field": field, "reason": reason},
        )
