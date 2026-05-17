"""
Synthron Finance Agent Example — market analysis, stock data, financial reasoning.

Run:
    python examples/finance_agent.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from synthron import Synthron
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print("\n[bold green]Synthron Finance Agent[/bold green]\n")
    console.print("[dim]Tools: web_search + data_analyzer + calculator + browser_tool[/dim]\n")

    agent = Synthron(
        tools=["web_search", "data_analyzer", "calculator", "browser_tool"],
    )

    events_seen = []

    def on_event(event):
        event_type = event.get("event_type", event.get("type", ""))
        content = event.get("content", "")
        agent_type = event.get("agent_type", "system")

        color_map = {
            "thought": "cyan",
            "action": "yellow",
            "result": "green",
            "plan_created": "magenta",
            "executing": "blue",
        }
        color = color_map.get(event_type, "white")
        events_seen.append(event_type)
        console.print(f"  [{color}]{agent_type}[/{color}] [dim]{event_type}[/dim] → {content[:90]}")

    agent.subscribe(on_event)

    tasks = [
        {
            "title": "Tech Stock Comparison",
            "task": (
                "Compare the investment potential of NVIDIA, AMD, and Intel in 2026. "
                "Include revenue growth trends, AI chip market share, and P/E ratios. "
                "Give a clear recommendation with reasoning."
            ),
        },
        {
            "title": "Crypto Market Analysis",
            "task": (
                "Analyze the current state of the top 3 cryptocurrencies by market cap in 2026. "
                "Include institutional adoption trends, regulatory developments, and price forecasts. "
                "Calculate the annualized return if someone invested $10,000 in Bitcoin in 2022."
            ),
        },
        {
            "title": "Portfolio Risk Calculator",
            "task": (
                "I have a portfolio: 40% S&P 500 index, 30% bonds, 20% tech stocks, 10% gold. "
                "Calculate the expected annual return and risk (standard deviation) assuming: "
                "S&P 500: 10% avg return, 15% std; bonds: 4% return, 5% std; "
                "tech: 18% return, 30% std; gold: 6% return, 12% std. "
                "Correlation between assets is 0.3 on average. Use the Markowitz formula."
            ),
        },
    ]

    results_table = Table(title="Finance Analysis Results", show_header=True, header_style="bold green")
    results_table.add_column("Task", style="cyan", width=25)
    results_table.add_column("Status", width=10)
    results_table.add_column("Tokens", width=10)
    results_table.add_column("Time", width=8)

    for item in tasks:
        console.print(f"\n[bold yellow]► {item['title']}[/bold yellow]")
        console.print(f"[dim]{item['task'][:120]}...[/dim]\n")

        with console.status(f"[bold green]Analyzing {item['title']}..."):
            result = await agent.run(item["task"])

        if result.success:
            console.print(
                Panel(
                    result.output[:1500] + ("..." if len(result.output) > 1500 else ""),
                    title=f"[bold green]{item['title']}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            status = "[green]✅ Pass[/green]"
        else:
            console.print(f"[red]Failed: {result.error}[/red]")
            status = "[red]❌ Fail[/red]"

        results_table.add_row(
            item["title"],
            status,
            f"{result.total_tokens:,}",
            f"{result.total_time_s:.1f}s",
        )

    console.print("\n")
    console.print(results_table)

    console.print(
        f"\n[dim]Events received: {len(events_seen)} | "
        f"Agent status: {agent.status()['router']['active_providers']}[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(main())
