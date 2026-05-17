"""Synthron tools package — 10 built-in tools."""

from synthron.tools.api_caller import ApiCallerTool
from synthron.tools.base_tool import BaseTool, ToolRegistry, ToolResult, tool_registry
from synthron.tools.browser_tool import BrowserTool
from synthron.tools.calculator import CalculatorTool
from synthron.tools.code_executor import CodeExecutorTool
from synthron.tools.data_analyzer import DataAnalyzerTool
from synthron.tools.email_tool import EmailTool
from synthron.tools.file_tool import FileTool
from synthron.tools.image_tool import ImageTool
from synthron.tools.terminal_tool import TerminalTool
from synthron.tools.web_search import WebSearchTool

# Default tool set (no API keys required)
DEFAULT_TOOLS: list[BaseTool] = [
    WebSearchTool(),
    CodeExecutorTool(),
    FileTool(),
    ApiCallerTool(),
    CalculatorTool(),
    DataAnalyzerTool(),
    BrowserTool(),
    ImageTool(),
    TerminalTool(),
    EmailTool(),
]


def get_default_tools() -> list[BaseTool]:
    """Return the default tool set and register them globally."""
    tool_registry.register_all(DEFAULT_TOOLS)
    return DEFAULT_TOOLS


def get_tools_by_names(names: list[str]) -> list[BaseTool]:
    """Return only the tools matching the given names."""
    all_tools = {t.name: t for t in DEFAULT_TOOLS}
    return [all_tools[n] for n in names if n in all_tools]


__all__ = [
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "tool_registry",
    "WebSearchTool",
    "CodeExecutorTool",
    "FileTool",
    "ApiCallerTool",
    "CalculatorTool",
    "DataAnalyzerTool",
    "BrowserTool",
    "ImageTool",
    "TerminalTool",
    "EmailTool",
    "DEFAULT_TOOLS",
    "get_default_tools",
    "get_tools_by_names",
]
