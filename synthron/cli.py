"""Synthron CLI — run tasks, start dashboard, run benchmarks."""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="synthron",
    help="Synthron — The Neural Fabric for Autonomous AI Agents",
    no_args_is_help=True,
)
console = Console()


@app.command("run")
def run_task(
    task: str = typer.Argument(..., help="Task to execute"),
    session_id: str = typer.Option("", "--session", "-s", help="Session ID for context"),
    stream: bool = typer.Option(False, "--stream", help="Stream output in real time"),
    tools: str = typer.Option("", "--tools", help="Comma-separated tools to enable"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run a task through Synthron and print the result."""
    from synthron.utils.logger import print_banner
    print_banner()

    async def _run():
        from synthron import Synthron

        tool_list = [t.strip() for t in tools.split(",") if t.strip()] if tools else None
        agent = Synthron(tools=tool_list)

        console.print(f"\n[bold cyan]Task:[/bold cyan] {task}\n")

        if stream:
            async for chunk in agent._orchestrator.stream(task, session_id=session_id):
                console.print(chunk, end="", markup=False)
            console.print()
        else:
            with console.status("[bold green]Running Synthron..."):
                result = await agent.run(task, session_id=session_id)

            if result.success:
                console.print(Panel(
                    result.output,
                    title="[bold green]Result[/bold green]",
                    border_style="green",
                ))
                if verbose:
                    console.print(
                        f"\n[dim]Tokens: {result.total_tokens:,} | "
                        f"Time: {result.total_time_s:.1f}s | "
                        f"Retries: {result.retry_count}[/dim]"
                    )
            else:
                console.print(f"[bold red]Task failed:[/bold red] {result.error}")
                raise typer.Exit(1)

    asyncio.run(_run())


@app.command("serve")
def serve_api(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
) -> None:
    """Start the Synthron API server and dashboard."""
    import uvicorn
    from synthron.utils.logger import print_banner
    print_banner()
    console.print(f"\n[bold cyan]Starting Synthron API on http://{host}:{port}[/bold cyan]")
    console.print(f"[dim]Dashboard: http://{host}:{port}/dashboard[/dim]")
    console.print(f"[dim]API docs:  http://{host}:{port}/docs[/dim]\n")

    uvicorn.run(
        "synthron.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command("benchmark")
def run_benchmark(
    tasks: int = typer.Option(5, "--tasks", "-n", help="Number of benchmark tasks to run"),
) -> None:
    """Run the Synthron benchmark suite."""
    from synthron.utils.logger import print_banner
    print_banner()

    async def _bench():
        from synthron.evals.benchmark import BenchmarkRunner
        runner = BenchmarkRunner()
        console.print(f"\n[bold]Running {tasks} benchmark tasks...[/bold]\n")
        summary = await runner.run_benchmark(max_tasks=tasks)
        runner.print_report(summary)

    asyncio.run(_bench())


@app.command("status")
def show_status() -> None:
    """Show Synthron configuration and provider status."""
    from synthron.utils.config import settings

    console.print("\n[bold cyan]Synthron Status[/bold cyan]\n")
    config = settings.summary()

    for key, value in config.items():
        console.print(f"  [dim]{key}:[/dim] {value}")

    providers = settings.providers.available_providers()
    console.print(f"\n  [bold]Available providers ({len(providers)}):[/bold]")
    for p in providers:
        console.print(f"    ✅ {p}")


@app.command("chat")
def interactive_chat() -> None:
    """Interactive chat mode — multi-turn conversation with Synthron."""
    from synthron.utils.logger import print_banner
    print_banner()
    console.print("\n[bold cyan]Synthron Interactive Mode[/bold cyan]")
    console.print("[dim]Type 'exit' to quit, 'clear' to reset session[/dim]\n")

    async def _chat():
        from synthron import Synthron
        agent = Synthron()
        session_id = ""

        while True:
            try:
                task = typer.prompt("You")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if task.strip().lower() in ("exit", "quit", "q"):
                break
            if task.strip().lower() == "clear":
                session_id = ""
                console.print("[dim]Session cleared.[/dim]")
                continue

            try:
                with console.status("Thinking..."):
                    result = await agent.run(task, session_id=session_id)
                session_id = session_id or "interactive"
                console.print(Panel(
                    result.output or result.error,
                    title="[bold green]Synthron[/bold green]",
                    border_style="cyan",
                ))
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")

    asyncio.run(_chat())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
