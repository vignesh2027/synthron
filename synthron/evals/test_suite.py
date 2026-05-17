"""Extended test suite for SYNTHRON — unit and integration style eval tasks."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


@dataclass
class EvalTask:
    id: str
    category: str
    task: str
    expected_keywords: list[str] = field(default_factory=list)
    max_score: float = 1.0
    timeout_s: float = 120.0


@dataclass
class EvalResult:
    task_id: str
    category: str
    score: float
    passed: bool
    duration_s: float
    tokens: int
    output_snippet: str
    error: str = ""


EVAL_TASKS = [
    # ── MATH & CALCULATION ──
    EvalTask(
        id="math_001",
        category="math",
        task="Calculate the compound interest on $10,000 invested at 7% annually for 15 years.",
        expected_keywords=["27590", "27,590", "17590", "17,590"],
    ),
    EvalTask(
        id="math_002",
        category="math",
        task="What is the derivative of f(x) = x³ + 5x² - 3x + 7?",
        expected_keywords=["3x²", "3x^2", "10x", "-3"],
    ),
    EvalTask(
        id="math_003",
        category="math",
        task="A triangle has sides of length 3, 4, and 5. What is its area?",
        expected_keywords=["6", "right triangle"],
    ),

    # ── REASONING ──
    EvalTask(
        id="reason_001",
        category="reasoning",
        task="If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops definitely Lazzles?",
        expected_keywords=["yes", "true", "definitely", "are lazzles"],
    ),
    EvalTask(
        id="reason_002",
        category="reasoning",
        task=(
            "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost?"
        ),
        expected_keywords=["5 cents", "$0.05", "0.05"],
    ),
    EvalTask(
        id="reason_003",
        category="reasoning",
        task="What comes next in the sequence: 2, 6, 12, 20, 30, ?",
        expected_keywords=["42"],
    ),

    # ── FACTUAL KNOWLEDGE ──
    EvalTask(
        id="fact_001",
        category="factual",
        task="What is the time complexity of binary search?",
        expected_keywords=["O(log n)", "logarithmic", "log n"],
    ),
    EvalTask(
        id="fact_002",
        category="factual",
        task="What does the CAP theorem state in distributed systems?",
        expected_keywords=["consistency", "availability", "partition"],
    ),
    EvalTask(
        id="fact_003",
        category="factual",
        task="Name the SOLID principles in software engineering.",
        expected_keywords=["single", "open", "liskov", "interface", "dependency"],
    ),

    # ── CODING ──
    EvalTask(
        id="code_001",
        category="coding",
        task="Write a Python function to check if a string is a palindrome.",
        expected_keywords=["def", "return", "[::-1]", "reversed"],
    ),
    EvalTask(
        id="code_002",
        category="coding",
        task="Write a Python function to flatten a nested list.",
        expected_keywords=["def", "flatten", "isinstance", "list"],
    ),
    EvalTask(
        id="code_003",
        category="coding",
        task="Write a SQL query to find the second highest salary from an employees table.",
        expected_keywords=["SELECT", "MAX", "salary", "WHERE", "NOT IN"],
    ),

    # ── WRITING / SUMMARIZATION ──
    EvalTask(
        id="write_001",
        category="writing",
        task=(
            "Summarize the key benefits of microservices architecture in 3 bullet points."
        ),
        expected_keywords=["scalab", "independent", "deploy"],
    ),
    EvalTask(
        id="write_002",
        category="writing",
        task="What are the pros and cons of remote work? Give 2 of each.",
        expected_keywords=["pro", "con", "flexib", "isolat"],
    ),

    # ── DATA ANALYSIS ──
    EvalTask(
        id="data_001",
        category="data",
        task=(
            "Given sales data: Jan=120, Feb=95, Mar=145, Apr=110, May=160. "
            "What is the average, and which month had the highest sales?"
        ),
        expected_keywords=["126", "may", "160"],
    ),
    EvalTask(
        id="data_002",
        category="data",
        task="If a dataset has mean=50 and std=10, what percentage of values fall between 40 and 60 using the empirical rule?",
        expected_keywords=["68%", "68 percent", "one standard deviation"],
    ),
]


class TestSuiteRunner:
    def __init__(self, agent=None):
        self._agent = agent
        self._results: list[EvalResult] = []

    async def _run_single(self, task: EvalTask) -> EvalResult:
        start = time.time()
        try:
            result = await asyncio.wait_for(
                self._agent.run(task.task),
                timeout=task.timeout_s,
            )
            duration = time.time() - start
            output = result.output.lower() if result.output else ""

            # Score: fraction of expected keywords found
            if task.expected_keywords:
                hits = sum(1 for kw in task.expected_keywords if kw.lower() in output)
                score = min(hits / max(len(task.expected_keywords), 1), 1.0)
                # Give partial credit: if at least 1 keyword matches, count as pass
                passed = hits >= max(1, len(task.expected_keywords) // 2)
            else:
                score = 1.0 if result.success else 0.0
                passed = result.success

            return EvalResult(
                task_id=task.id,
                category=task.category,
                score=round(score, 3),
                passed=passed,
                duration_s=round(duration, 2),
                tokens=result.total_tokens,
                output_snippet=result.output[:120] if result.output else "",
            )
        except asyncio.TimeoutError:
            return EvalResult(
                task_id=task.id,
                category=task.category,
                score=0.0,
                passed=False,
                duration_s=task.timeout_s,
                tokens=0,
                output_snippet="",
                error=f"Timeout after {task.timeout_s}s",
            )
        except Exception as e:
            return EvalResult(
                task_id=task.id,
                category=task.category,
                score=0.0,
                passed=False,
                duration_s=time.time() - start,
                tokens=0,
                output_snippet="",
                error=str(e)[:200],
            )

    async def run(self, tasks: list[EvalTask] | None = None, concurrency: int = 1) -> list[EvalResult]:
        tasks = tasks or EVAL_TASKS
        self._results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            eval_task = progress.add_task("[green]Running eval suite...", total=len(tasks))

            sem = asyncio.Semaphore(concurrency)

            async def run_with_sem(t):
                async with sem:
                    result = await self._run_single(t)
                    self._results.append(result)
                    status = "✅" if result.passed else "❌"
                    progress.advance(eval_task)
                    progress.console.print(
                        f"  {status} [{t.category}] {t.id}: score={result.score:.2f} ({result.duration_s:.1f}s)"
                    )

            await asyncio.gather(*[run_with_sem(t) for t in tasks])

        return self._results

    def print_report(self):
        results = self._results
        if not results:
            console.print("[yellow]No results to display.[/yellow]")
            return

        # Category breakdown
        categories: dict[str, list[EvalResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        table = Table(title="Synthron Eval Suite Results", show_header=True, header_style="bold cyan")
        table.add_column("Category", style="cyan")
        table.add_column("Tasks", width=6)
        table.add_column("Passed", width=7)
        table.add_column("Avg Score", width=10)
        table.add_column("Avg Time", width=9)

        total_passed = 0
        total_tasks = len(results)

        for cat, cat_results in sorted(categories.items()):
            passed = sum(1 for r in cat_results if r.passed)
            total_passed += passed
            avg_score = sum(r.score for r in cat_results) / len(cat_results)
            avg_time = sum(r.duration_s for r in cat_results) / len(cat_results)
            table.add_row(
                cat,
                str(len(cat_results)),
                f"{passed}/{len(cat_results)}",
                f"{avg_score:.1%}",
                f"{avg_time:.1f}s",
            )

        console.print(table)

        overall_score = sum(r.score for r in results) / len(results)
        console.print(f"\n[bold]Overall: {total_passed}/{total_tasks} passed | Avg score: {overall_score:.1%}[/bold]")

        failed = [r for r in results if not r.passed]
        if failed:
            console.print("\n[bold red]Failed tasks:[/bold red]")
            for r in failed:
                console.print(f"  ❌ {r.task_id}: {r.error or 'score too low'} (score={r.score:.2f})")


async def run_test_suite(agent=None) -> list[EvalResult]:
    if agent is None:
        from synthron import Synthron
        agent = Synthron()

    runner = TestSuiteRunner(agent=agent)
    results = await runner.run()
    runner.print_report()
    return results


if __name__ == "__main__":
    asyncio.run(run_test_suite())
