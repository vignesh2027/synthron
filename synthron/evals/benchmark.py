"""GAIA-style benchmark runner for Synthron."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from synthron.utils.logger import get_logger

logger = get_logger(__name__)

# Sample GAIA-style benchmark tasks (subset)
BENCHMARK_TASKS = [
    {
        "id": "gaia_001",
        "task": "What is 2 to the power of 10?",
        "expected": "1024",
        "category": "math",
        "difficulty": 1,
    },
    {
        "id": "gaia_002",
        "task": "What is the capital of France?",
        "expected": "Paris",
        "category": "knowledge",
        "difficulty": 1,
    },
    {
        "id": "gaia_003",
        "task": "Write a Python function that returns the fibonacci sequence up to n terms.",
        "expected": "fibonacci",
        "category": "coding",
        "difficulty": 2,
    },
    {
        "id": "gaia_004",
        "task": "What is the square root of 144 divided by 3?",
        "expected": "4",
        "category": "math",
        "difficulty": 2,
    },
    {
        "id": "gaia_005",
        "task": "Explain the difference between supervised and unsupervised learning in 2 sentences.",
        "expected": "labeled|training|supervised",
        "category": "knowledge",
        "difficulty": 2,
    },
]


class BenchmarkRunner:
    """Run Synthron against benchmark tasks and report accuracy."""

    def __init__(self, orchestrator: Any = None) -> None:
        self._orchestrator = orchestrator
        self._results: list[dict] = []

    async def run_benchmark(
        self, tasks: list[dict] | None = None, max_tasks: int = 10
    ) -> dict[str, Any]:
        """Run the benchmark suite.

        Args:
            tasks: List of task dicts with id, task, expected, category, difficulty.
            max_tasks: Maximum tasks to run.

        Returns:
            Benchmark results summary dict.
        """
        benchmark_tasks = (tasks or BENCHMARK_TASKS)[:max_tasks]
        logger.info(f"[benchmark] Running {len(benchmark_tasks)} benchmark tasks...")

        if not self._orchestrator:
            from synthron.orchestrator.orchestrator import Orchestrator
            self._orchestrator = Orchestrator()
            await self._orchestrator.initialize()

        results = []
        start_time = time.perf_counter()

        for i, task_def in enumerate(benchmark_tasks, 1):
            logger.info(f"[benchmark] Task {i}/{len(benchmark_tasks)}: {task_def['task'][:60]}")
            task_start = time.perf_counter()

            try:
                result = await asyncio.wait_for(
                    self._orchestrator.run(task_def["task"]),
                    timeout=120,
                )
                correct = self._check_answer(result.output, task_def["expected"])
                elapsed = time.perf_counter() - task_start

                task_result = {
                    "id": task_def["id"],
                    "task": task_def["task"][:80],
                    "category": task_def.get("category", ""),
                    "difficulty": task_def.get("difficulty", 1),
                    "correct": correct,
                    "output": result.output[:200],
                    "tokens": result.total_tokens,
                    "time_s": round(elapsed, 2),
                    "success": result.success,
                }
                logger.info(
                    f"[benchmark] {'✅' if correct else '❌'} {task_def['id']} "
                    f"({elapsed:.1f}s, {result.total_tokens} tokens)"
                )

            except asyncio.TimeoutError:
                task_result = {
                    "id": task_def["id"],
                    "task": task_def["task"][:80],
                    "correct": False,
                    "error": "Timeout",
                    "time_s": 120.0,
                }
                logger.warning(f"[benchmark] Timeout: {task_def['id']}")

            except Exception as exc:
                task_result = {
                    "id": task_def["id"],
                    "task": task_def["task"][:80],
                    "correct": False,
                    "error": str(exc),
                }
                logger.error(f"[benchmark] Error on {task_def['id']}: {exc}")

            results.append(task_result)

        total_time = time.perf_counter() - start_time
        self._results = results

        return self._compute_summary(results, total_time)

    def _check_answer(self, output: str, expected: str) -> bool:
        """Check if the output contains the expected answer."""
        output_lower = output.lower()
        # Support pipe-separated alternatives
        candidates = [e.strip().lower() for e in expected.split("|")]
        return any(c in output_lower for c in candidates)

    def _compute_summary(self, results: list[dict], total_time: float) -> dict[str, Any]:
        """Compute accuracy and summary statistics."""
        total = len(results)
        correct = sum(1 for r in results if r.get("correct", False))
        accuracy = correct / total if total else 0

        by_category: dict[str, dict] = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0}
            by_category[cat]["total"] += 1
            if r.get("correct"):
                by_category[cat]["correct"] += 1

        summary = {
            "total_tasks": total,
            "correct": correct,
            "accuracy": round(accuracy, 3),
            "accuracy_pct": f"{accuracy * 100:.1f}%",
            "total_time_s": round(total_time, 1),
            "avg_time_s": round(total_time / total, 2) if total else 0,
            "by_category": {
                cat: {
                    "accuracy": round(v["correct"] / v["total"], 3),
                    **v,
                }
                for cat, v in by_category.items()
            },
            "results": results,
        }

        logger.info(
            f"[benchmark] ✅ Complete: {correct}/{total} = {accuracy*100:.1f}% accuracy "
            f"in {total_time:.1f}s"
        )
        return summary

    def print_report(self, summary: dict) -> None:
        """Print a formatted benchmark report."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Synthron Benchmark Results")
        table.add_column("Task ID")
        table.add_column("Category")
        table.add_column("Result")
        table.add_column("Time (s)")

        for r in summary.get("results", []):
            table.add_row(
                r["id"],
                r.get("category", ""),
                "✅ PASS" if r.get("correct") else "❌ FAIL",
                str(r.get("time_s", "")),
            )

        console.print(table)
        console.print(
            f"\n[bold]Overall Accuracy: {summary['accuracy_pct']}[/bold] "
            f"({summary['correct']}/{summary['total_tasks']})"
        )
