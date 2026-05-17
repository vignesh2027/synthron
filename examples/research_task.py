"""
Synthron Research Task Example — deep multi-source research with live streaming.

Run:
    python examples/research_task.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from synthron import Synthron
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    console.print("\n[bold magenta]Synthron Research Agent[/bold magenta]\n")

    agent = Synthron(
        tools=["web_search", "browser_tool", "data_analyzer", "calculator", "file_tool"],
    )

    # Subscribe to live events
    def on_event(event):
        agent_type = event.get("agent_type", "")
        content = event.get("content", "")
        event_type = event.get("event_type", event.get("type", ""))

        icons = {
            "thought": "💭",
            "action": "🔧",
            "result": "✅",
            "score": "📊",
            "plan_created": "📋",
            "executing": "⚡",
        }
        icon = icons.get(event_type, "·")
        color = "cyan" if "planner" in agent_type else "green" if "executor" in agent_type else "yellow"
        console.print(f"  [{color}]{icon} {content[:100]}[/{color}]")

    agent.subscribe(on_event)

    task = (
        "Research the top 5 AI companies by market cap in 2025-2026, "
        "include their main products, revenue estimates, and key AI breakthroughs."
    )

    console.print(f"[bold]Task:[/bold] {task}\n")

    with console.status("[bold green]Running research agent..."):
        result = await agent.run(task)

    console.print(Panel(
        result.output,
        title="[bold green]Research Report[/bold green]",
        border_style="green",
    ))

    console.print(
        f"\n[dim]Tokens used: {result.total_tokens:,} | "
        f"Time: {result.total_time_s:.1f}s | "
        f"Retries: {result.retry_count}[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(main())
