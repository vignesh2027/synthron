"""Rich-powered structured logger for Synthron."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback

# Install rich tracebacks globally
install_rich_traceback(show_locals=False, max_frames=10)

# ─── Color theme ──────────────────────────────────────────────────────────────

SYNTHRON_THEME = Theme(
    {
        "agent.planner": "bold cyan",
        "agent.executor": "bold green",
        "agent.critic": "bold yellow",
        "agent.memory": "bold magenta",
        "agent.researcher": "bold blue",
        "agent.coder": "bold white",
        "agent.coordinator": "bold red",
        "provider.gemini": "cyan",
        "provider.groq": "green",
        "provider.cerebras": "blue",
        "provider.deepseek": "magenta",
        "provider.openrouter": "yellow",
        "provider.ollama": "white",
        "score.pass": "bold green",
        "score.warn": "bold yellow",
        "score.fail": "bold red",
        "token": "dim cyan",
        "subtask": "italic",
        "thought": "dim white",
        "tool": "bold blue",
        "synthron": "bold magenta",
    }
)

console = Console(theme=SYNTHRON_THEME, stderr=True)


class SynthronHighlighter(RegexHighlighter):
    """Highlight Synthron-specific patterns in log output."""

    base_style = "synthron."
    highlights = [
        r"(?P<score_pass>score[:\s]+[0-9.]+✅)",
        r"(?P<score_warn>score[:\s]+[0-9.]+⚠️)",
        r"(?P<score_fail>score[:\s]+[0-9.]+❌)",
        r"(?P<token>\d+\s*tok(?:en)?s?)",
    ]


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger with rich output.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        highlighter=SynthronHighlighter(),
        markup=True,
        log_time_format="[%H:%M:%S]",
    )

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler.setLevel(numeric_level)
    logger.setLevel(numeric_level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


class AgentLogger:
    """Structured logger with agent-aware formatting for streaming to dashboard."""

    AGENT_COLORS: dict[str, str] = {
        "planner": "cyan",
        "executor": "green",
        "critic": "yellow",
        "memory": "magenta",
        "researcher": "blue",
        "coder": "white",
        "coordinator": "red",
    }

    def __init__(self, agent_name: str, agent_type: str) -> None:
        self.agent_name = agent_name
        self.agent_type = agent_type.lower()
        self._logger = get_logger(f"synthron.{agent_type}.{agent_name}")
        self._color = self.AGENT_COLORS.get(self.agent_type, "white")
        self._subscribers: list[Any] = []  # event bus subscribers

    def _prefix(self) -> str:
        return f"[bold {self._color}][{self.agent_name.upper()}][/bold {self._color}]"

    def thought(self, message: str) -> None:
        """Log an agent thought (reasoning step)."""
        console.print(f"{self._prefix()} [dim]💭 {message}[/dim]")
        self._emit("thought", message)

    def action(self, tool: str, args: str = "") -> None:
        """Log a tool action."""
        console.print(f"{self._prefix()} [bold blue]🔧 {tool}[/bold blue] {args}")
        self._emit("action", f"{tool}: {args}")

    def result(self, message: str) -> None:
        """Log a result."""
        console.print(f"{self._prefix()} [green]✅ {message}[/green]")
        self._emit("result", message)

    def warning(self, message: str) -> None:
        """Log a warning."""
        console.print(f"{self._prefix()} [yellow]⚠️  {message}[/yellow]")
        self._emit("warning", message)

    def error(self, message: str) -> None:
        """Log an error."""
        console.print(f"{self._prefix()} [bold red]❌ {message}[/bold red]")
        self._emit("error", message)

    def score(self, value: float, threshold: float = 0.8) -> None:
        """Log a critic score with visual indicator."""
        if value >= threshold:
            icon = "✅"
            style = "bold green"
        elif value >= 0.5:
            icon = "⚠️"
            style = "bold yellow"
        else:
            icon = "❌"
            style = "bold red"
        console.print(f"{self._prefix()} [{style}]Score: {value:.2f} {icon}[/{style}]")
        self._emit("score", str(value))

    def _emit(self, event_type: str, message: str) -> None:
        """Emit event to dashboard subscribers."""
        event = {
            "ts": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "agent_type": self.agent_type,
            "event_type": event_type,
            "message": message,
        }
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                pass

    def subscribe(self, callback: Any) -> None:
        """Register a callback for log events (used by dashboard)."""
        self._subscribers.append(callback)


def print_banner() -> None:
    """Print the Synthron startup banner."""
    console.print(
        """
[bold magenta]
╔══════════════════════════════════════════════════════════╗
║  ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗██████╗   ║
║  ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██╔══██╗  ║
║  ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║██████╔╝  ║
║  ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██╔══██╗  ║
║  ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║██║  ██║  ║
║  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝  ║
║                                                          ║
║       The Neural Fabric for Autonomous AI Agents         ║
╚══════════════════════════════════════════════════════════╝
[/bold magenta]"""
    )


# Module-level default logger
logger = get_logger("synthron")
