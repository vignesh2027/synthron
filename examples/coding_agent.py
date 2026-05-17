"""
Synthron Coding Agent Example — write, execute, and debug code automatically.

Run:
    python examples/coding_agent.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from synthron import Synthron
    from rich.console import Console
    from rich.syntax import Syntax

    console = Console()
    console.print("\n[bold white]Synthron Coding Agent[/bold white]\n")

    agent = Synthron(tools=["code_executor", "file_tool", "calculator"])

    coding_tasks = [
        "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes",
        "Create a Python class for a simple stack data structure with push, pop, peek, and is_empty methods",
        "Write a Python script that generates a Fibonacci sequence up to the 20th term and prints it",
    ]

    for task in coding_tasks:
        console.print(f"[bold cyan]Task:[/bold cyan] {task}\n")
        result = await agent.run(task)

        if result.success:
            # Try to extract and display code nicely
            import re
            code_match = re.search(r"```python\n(.*?)```", result.output, re.DOTALL)
            if code_match:
                syntax = Syntax(code_match.group(1), "python", theme="monokai")
                console.print(syntax)
            else:
                console.print(result.output[:500])
        else:
            console.print(f"[red]Failed: {result.error}[/red]")

        console.print()


if __name__ == "__main__":
    asyncio.run(main())
