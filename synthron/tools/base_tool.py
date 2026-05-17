"""Base tool interface and global tool registry."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from synthron.utils.exceptions import ToolNotFoundError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class ToolResult(BaseModel):
    """Standardized result from any tool execution."""

    tool_name: str
    success: bool = True
    output: Any = None
    error: str = ""
    metadata: dict[str, Any] = {}

    def __str__(self) -> str:
        if not self.success:
            return f"[{self.tool_name} ERROR]: {self.error}"
        return str(self.output) if self.output is not None else ""


class BaseTool(ABC):
    """Abstract base for all Synthron tools.

    Tools are async by default and can be registered in the global registry.
    Agents call tools via tool.run(input_text, context=ctx).
    """

    name: str = "base_tool"
    description: str = "A Synthron tool"
    category: str = "general"
    requires_network: bool = False
    is_destructive: bool = False  # marks tools that write/delete data

    @abstractmethod
    async def run(self, input_text: str, context: Any = None) -> str:
        """Execute the tool with the given input.

        Args:
            input_text: Primary input string (query, code, path, etc.)
            context: Optional context dict or value from the calling agent.

        Returns:
            String output from the tool.

        Raises:
            ToolExecutionError: On non-recoverable tool failure.
            ToolTimeoutError: If execution exceeds allowed time.
        """

    async def validate_input(self, input_text: str) -> bool:
        """Validate input before execution. Override for input-specific validation."""
        return bool(input_text and isinstance(input_text, str))

    async def safe_run(
        self, input_text: str, context: Any = None, timeout: float = 30.0
    ) -> ToolResult:
        """Execute the tool with error handling and timeout.

        Args:
            input_text: Tool input.
            context: Optional context.
            timeout: Execution timeout in seconds.

        Returns:
            ToolResult with output or error info.
        """
        try:
            if not await self.validate_input(input_text):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Invalid input for tool '{self.name}'",
                )

            output = await asyncio.wait_for(
                self.run(input_text, context=context),
                timeout=timeout,
            )
            return ToolResult(tool_name=self.name, success=True, output=output)

        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Tool '{self.name}' timed out after {timeout}s",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
            )

    def schema(self) -> dict[str, Any]:
        """Return tool schema for agent/LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "requires_network": self.requires_network,
            "is_destructive": self.is_destructive,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class ToolRegistry:
    """Global registry for all Synthron tools.

    Tools are registered by name and retrieved by agents on demand.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        logger.debug(f"[registry] Registered tool: {tool.name}")

    def register_all(self, tools: list[BaseTool]) -> None:
        """Register a list of tool instances."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> BaseTool:
        """Retrieve a tool by name.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def get_optional(self, name: str) -> BaseTool | None:
        """Retrieve a tool by name, returning None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """Return schemas for all tools (for LLM consumption)."""
        return [t.schema() for t in self._tools.values()]

    def by_category(self, category: str) -> list[BaseTool]:
        """Return tools filtered by category."""
        return [t for t in self._tools.values() if t.category == category]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Global tool registry
tool_registry = ToolRegistry()
