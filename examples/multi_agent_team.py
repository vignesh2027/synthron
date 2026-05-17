"""
Synthron Multi-Agent Team Example — coordinator + researcher + coder + critic.

Run:
    python examples/multi_agent_team.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from synthron import Orchestrator
    from synthron.tools import get_default_tools
    from rich.console import Console

    console = Console()
    console.print("\n[bold magenta]Synthron Multi-Agent Team[/bold magenta]\n")

    tools = get_default_tools()

    # Full orchestrator with all agents
    orch = Orchestrator(
        tools=tools,
        max_executors=3,
        critic_threshold=0.75,
    )
    await orch.initialize()

    # Live event streaming
    def on_event(event):
        console.print(
            f"  [dim]{event.get('agent', 'system')}[/dim] → "
            f"{event.get('content', '')[:80]}"
        )

    orch.subscribe(on_event)

    # Complex task that requires multiple agent types
    task = (
        "Create a comprehensive report on Python vs JavaScript for AI development in 2026: "
        "include ecosystem comparison, popular libraries, performance benchmarks, "
        "job market data, and a code example in each language."
    )

    console.print(f"[bold]Task:[/bold] {task[:100]}...\n")
    console.print("[dim]Agents: Planner → Executor (×3 parallel) → Critic → Memory[/dim]\n")

    result = await orch.run(task)

    console.print("\n" + "=" * 60)
    console.print("[bold green]FINAL REPORT[/bold green]")
    console.print("=" * 60)
    console.print(result.output)
    console.print(f"\n[dim]Total tokens: {result.total_tokens:,} | Time: {result.total_time_s:.1f}s[/dim]")

    # Show orchestrator stats
    status = orch.status()
    console.print(f"\n[dim]Router status: {list(status['router']['active_providers'])}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
