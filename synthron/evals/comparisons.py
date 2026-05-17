"""Framework comparison benchmarks — Synthron vs baselines."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class ComparisonTask:
    id: str
    description: str
    prompt: str
    evaluator: Callable[[str], float]  # returns 0.0–1.0


@dataclass
class FrameworkResult:
    framework: str
    task_id: str
    score: float
    duration_s: float
    tokens: int
    output: str = ""
    error: str = ""


def _keyword_evaluator(keywords: list[str], min_hits: int = 1) -> Callable[[str], float]:
    def evaluate(output: str) -> float:
        hits = sum(1 for kw in keywords if kw.lower() in output.lower())
        if hits == 0:
            return 0.0
        return min(hits / len(keywords), 1.0) if hits >= min_hits else 0.0
    return evaluate


def _length_evaluator(min_words: int = 50) -> Callable[[str], float]:
    def evaluate(output: str) -> float:
        words = len(output.split())
        if words >= min_words:
            return 1.0
        return words / min_words
    return evaluate


def _combined_evaluator(*evaluators: Callable[[str], float]) -> Callable[[str], float]:
    def evaluate(output: str) -> float:
        if not evaluators:
            return 0.0
        return sum(e(output) for e in evaluators) / len(evaluators)
    return evaluate


COMPARISON_TASKS = [
    ComparisonTask(
        id="research_basic",
        description="Basic research — Python AI libraries",
        prompt="What are the top 5 Python libraries for machine learning? List them with a one-sentence description each.",
        evaluator=_combined_evaluator(
            _keyword_evaluator(["tensorflow", "pytorch", "scikit", "keras", "numpy", "pandas", "xgboost"], min_hits=3),
            _length_evaluator(min_words=50),
        ),
    ),
    ComparisonTask(
        id="coding_function",
        description="Coding — write a working function",
        prompt="Write a Python function called `binary_search(arr, target)` that implements binary search and returns the index or -1.",
        evaluator=_combined_evaluator(
            _keyword_evaluator(["def binary_search", "mid", "low", "high", "return"], min_hits=4),
            _length_evaluator(min_words=20),
        ),
    ),
    ComparisonTask(
        id="analysis_comparison",
        description="Analysis — compare two options",
        prompt="Compare REST and GraphQL APIs. Give 3 advantages of each with examples.",
        evaluator=_combined_evaluator(
            _keyword_evaluator(["rest", "graphql", "over-fetching", "endpoint", "query", "flexible"], min_hits=4),
            _length_evaluator(min_words=100),
        ),
    ),
    ComparisonTask(
        id="math_reasoning",
        description="Math reasoning — word problem",
        prompt="A train travels 120 km in 2 hours then 180 km in 3 hours. What is the average speed for the entire journey?",
        evaluator=_keyword_evaluator(["60 km/h", "60km/h", "60 kilometers per hour", "300", "5 hours"], min_hits=2),
    ),
    ComparisonTask(
        id="structured_output",
        description="Structured output — JSON generation",
        prompt='Generate a JSON object representing a user profile with: name, email, age (25), skills (list of 3), and is_active (true).',
        evaluator=_combined_evaluator(
            _keyword_evaluator(['"name"', '"email"', '"age"', '"skills"', '"is_active"', "true"], min_hits=5),
            _length_evaluator(min_words=10),
        ),
    ),
]


class FrameworkBenchmark:
    """
    Compares Synthron against simple LLM-only baselines.

    Baselines:
    - "direct_llm" — single LLM call with no agent pipeline
    - "synthron"   — full Synthron agent with tools + pipeline
    """

    def __init__(self):
        self._results: list[FrameworkResult] = []

    async def _run_synthron(self, task: ComparisonTask) -> FrameworkResult:
        from synthron import Synthron

        agent = Synthron(tools=["web_search", "calculator", "code_executor"])
        start = time.time()
        try:
            result = await asyncio.wait_for(agent.run(task.prompt), timeout=90.0)
            duration = time.time() - start
            score = task.evaluator(result.output or "")
            return FrameworkResult(
                framework="synthron",
                task_id=task.id,
                score=round(score, 3),
                duration_s=round(duration, 2),
                tokens=result.total_tokens,
                output=result.output[:200] if result.output else "",
            )
        except Exception as e:
            return FrameworkResult(
                framework="synthron",
                task_id=task.id,
                score=0.0,
                duration_s=time.time() - start,
                tokens=0,
                error=str(e)[:150],
            )

    async def _run_direct_llm(self, task: ComparisonTask) -> FrameworkResult:
        """Single LLM call — no tools, no pipeline."""
        from synthron.providers.smart_router import router
        from synthron.providers.base_provider import GenerationRequest, Message

        start = time.time()
        try:
            request = GenerationRequest(
                messages=[Message(role="user", content=task.prompt)],
                max_tokens=1024,
                temperature=0.7,
            )
            response = await asyncio.wait_for(
                router.generate(request, agent_type="default"),
                timeout=30.0,
            )
            duration = time.time() - start
            output = response.content or ""
            score = task.evaluator(output)
            return FrameworkResult(
                framework="direct_llm",
                task_id=task.id,
                score=round(score, 3),
                duration_s=round(duration, 2),
                tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                output=output[:200],
            )
        except Exception as e:
            return FrameworkResult(
                framework="direct_llm",
                task_id=task.id,
                score=0.0,
                duration_s=time.time() - start,
                tokens=0,
                error=str(e)[:150],
            )

    async def run(self, tasks: list[ComparisonTask] | None = None) -> list[FrameworkResult]:
        tasks = tasks or COMPARISON_TASKS
        self._results = []

        for task in tasks:
            console.print(f"\n[bold cyan]► {task.description}[/bold cyan]")
            console.print(f"  [dim]{task.prompt[:80]}...[/dim]")

            # Run frameworks in parallel
            synthron_result, direct_result = await asyncio.gather(
                self._run_synthron(task),
                self._run_direct_llm(task),
                return_exceptions=False,
            )

            self._results.extend([synthron_result, direct_result])

            winner = "🏆 Synthron" if synthron_result.score >= direct_result.score else "   Direct LLM"
            console.print(
                f"  Synthron: score={synthron_result.score:.2f} | {synthron_result.duration_s:.1f}s"
            )
            console.print(
                f"  Direct:   score={direct_result.score:.2f} | {direct_result.duration_s:.1f}s"
            )
            console.print(f"  {winner}")

        return self._results

    def print_report(self):
        results = self._results
        if not results:
            console.print("[yellow]No results.[/yellow]")
            return

        # Group by framework
        frameworks: dict[str, list[FrameworkResult]] = {}
        for r in results:
            frameworks.setdefault(r.framework, []).append(r)

        table = Table(title="Framework Comparison Results", header_style="bold magenta")
        table.add_column("Framework", style="bold")
        table.add_column("Avg Score", width=10)
        table.add_column("Avg Time", width=9)
        table.add_column("Total Tokens", width=13)
        table.add_column("Tasks Won", width=10)

        task_ids = list({r.task_id for r in results})
        fw_names = list(frameworks.keys())

        wins: dict[str, int] = {fw: 0 for fw in fw_names}
        for task_id in task_ids:
            task_results = {r.framework: r for r in results if r.task_id == task_id}
            if len(task_results) >= 2:
                best_fw = max(task_results, key=lambda fw: task_results[fw].score)
                wins[best_fw] += 1

        for fw, fw_results in frameworks.items():
            avg_score = sum(r.score for r in fw_results) / len(fw_results)
            avg_time = sum(r.duration_s for r in fw_results) / len(fw_results)
            total_tokens = sum(r.tokens for r in fw_results)
            style = "bold green" if fw == "synthron" else ""
            table.add_row(
                f"[{style}]{fw}[/{style}]" if style else fw,
                f"{avg_score:.1%}",
                f"{avg_time:.1f}s",
                f"{total_tokens:,}",
                f"{wins[fw]}/{len(task_ids)}",
            )

        console.print(table)

        synthron_avg = sum(r.score for r in frameworks.get("synthron", [])) / max(len(frameworks.get("synthron", [])), 1)
        direct_avg = sum(r.score for r in frameworks.get("direct_llm", [])) / max(len(frameworks.get("direct_llm", [])), 1)

        if synthron_avg > direct_avg:
            improvement = ((synthron_avg - direct_avg) / max(direct_avg, 0.001)) * 100
            console.print(f"\n[bold green]✅ Synthron outperforms direct LLM by {improvement:.1f}%[/bold green]")
        else:
            console.print("\n[yellow]Note: Results vary by task type and provider availability.[/yellow]")


async def run_comparison(tasks: list[ComparisonTask] | None = None) -> list[FrameworkResult]:
    benchmark = FrameworkBenchmark()
    results = await benchmark.run(tasks)
    benchmark.print_report()
    return results


if __name__ == "__main__":
    asyncio.run(run_comparison())
