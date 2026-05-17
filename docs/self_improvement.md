# Self-Improvement System

SYNTHRON learns from every run and improves automatically.

## Overview

The self-improvement system has 5 components that work together:

```
Each Task Run
    │
    ├── CriticAgent scores result (0.0–1.0)
    │
    ├── FailureAnalyzer records what went wrong
    ├── PromptOptimizer A/B tests prompt variants
    ├── PatternLearner categorizes the task + updates recommendations
    ├── PerformanceTracker records metrics to rolling window
    └── RetryStrategist decides how to recover from failures
```

---

## FailureAnalyzer

Tracks every failure to identify patterns and fixes.

```python
from synthron.self_improvement.failure_analyzer import FailureAnalyzer

analyzer = FailureAnalyzer()

# Record a failure
analyzer.record_failure(
    task="Research quantum computing",
    error="RateLimitError: gemini daily limit exceeded",
    tool_used="web_search",
    agent_type="researcher"
)

# Get suggested fix
fix = analyzer.get_fix("RateLimitError")
# "Switch to backup provider; current provider quota exceeded"

# Generate report
report = analyzer.generate_report()
print(report)
# ## Failure Analysis Report
# Total failures: 3
# Top error types:
#   - RateLimitError: 2 occurrences
#   - TimeoutError: 1 occurrence
# ...
```

**Tracked metrics:**
- Failure count per error type
- Failure count per tool
- Failure count per agent type
- Time of last failure
- Known fixes (pattern → suggested action)

---

## PromptOptimizer

Uses epsilon-greedy A/B testing to find the best prompt for each task type.

```python
from synthron.self_improvement.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer(router)

# Get best prompt for a task type
prompt = await optimizer.get_best_prompt(
    task_type="coding",
    base_prompt="You are a Python expert. Write clean, well-tested code."
)

# After seeing the result, update the variant's score
optimizer.record_result(task_type="coding", variant_id="v_abc123", score=0.92)

# Manually generate an improved variant via LLM
new_variant = await optimizer.generate_improved_prompt(
    task_type="coding",
    base_prompt="...",
    failure_context="Previous attempts had poor error handling"
)
```

**Epsilon-greedy strategy:**
- 80% of the time: use the best-known prompt variant (exploit)
- 20% of the time: try a random variant (explore)
- After enough data: the best variant wins and 80% traffic goes to it

---

## PatternLearner

Classifies tasks and learns which tools/approaches work best per category.

```python
from synthron.self_improvement.pattern_learner import PatternLearner

learner = PatternLearner()

# Classify a task
category = learner._classify_task("Write unit tests for my FastAPI endpoints")
# "coding"

# Get tool recommendations for this category
tools = learner.get_tool_recommendation("coding")
# ["code_executor", "file_tool", "terminal_tool"]

# Get prompt hints
hint = learner.get_prompt_hint("coding")
# "Use code_executor to test your implementation. Prefer working code over explanations."

# Learn from a completed task
await learner.learn(
    task="Write a binary search implementation",
    task_type="coding",
    tools_used=["code_executor"],
    score=0.95,
    duration_s=12.3
)
```

**Task categories:**
- `research` — keywords: research, find, search, investigate, analyze, study
- `coding` — keywords: code, write, implement, debug, fix, refactor, test
- `math` — keywords: calculate, compute, solve, equation, formula, statistics
- `writing` — keywords: write, draft, compose, summarize, translate, explain
- `data` — keywords: data, csv, json, database, sql, chart, visualize
- `general` — everything else

---

## PerformanceTracker

Rolling window metrics with trend detection.

```python
from synthron.self_improvement.performance_tracker import PerformanceTracker

tracker = PerformanceTracker(window_size=100)

# Record a completed task
tracker.record(
    task_id="task_abc",
    score=0.87,
    duration_s=15.2,
    tokens_used=2400,
    success=True
)

# Get current metrics
metrics = tracker.get_current_metrics()
# {
#   "avg_score": 0.84,
#   "p50_duration_s": 12.1,
#   "p95_duration_s": 38.5,
#   "success_rate": 0.91,
#   "avg_tokens": 2150,
#   "total_tasks": 47
# }

# Detect trend
trend = tracker.get_trend()
# "improving" | "declining" | "stable" | "insufficient_data"

is_better = tracker.is_improving()
# True — last 10 runs scored higher than prior 10

report = tracker.generate_report()
# Full markdown performance report
```

---

## RetryStrategist

Decides whether and how to retry failed tasks.

```python
from synthron.self_improvement.retry_strategist import RetryStrategist

strategist = RetryStrategist(
    max_retries=3,
    min_score_threshold=0.5,
)

# Decide what to do after a failure
strategy = await strategist.decide(
    task="Scrape product data from Amazon",
    current_score=0.3,
    attempt_number=1,
    error="TimeoutError: browser_tool took too long",
    tools_used=["browser_tool"],
    failure_analyzer=analyzer,
)

print(strategy.action)        # "RETRY" | "SKIP"
print(strategy.retry_delay_s) # 4.0 (exponential: 2^attempt)
print(strategy.new_prompt)    # improved prompt with context
print(strategy.suggested_tools) # ["web_search"] instead of ["browser_tool"]
```

**Retry decision logic:**
1. If `attempt_number >= max_retries` → SKIP
2. If `current_score >= min_score_threshold` → SKIP (good enough)
3. If error is a known-unfixable type (auth, quota) → SKIP
4. Otherwise → RETRY with:
   - Exponential backoff: `2^attempt` seconds
   - Improved prompt (adds failure context)
   - Alternative tool suggestions (from FailureAnalyzer)

---

## Integration with Orchestrator

The orchestrator uses all 5 components automatically:

```python
# In orchestrator.py — simplified
async def _execute_with_improvement(self, subtask, attempt=0):
    result = await executor.execute(subtask)
    score = await critic.score(result)

    # Track performance
    perf_tracker.record(subtask.id, score, result.duration, result.tokens)

    # Learn from result
    await pattern_learner.learn(subtask.task, task_type, result.tools, score)

    if score < 0.5:
        # Analyze failure
        failure_analyzer.record_failure(subtask.task, result.error, ...)

        # Decide retry strategy
        strategy = await retry_strategist.decide(subtask.task, score, attempt, ...)

        if strategy.action == "RETRY" and attempt < max_retries:
            await asyncio.sleep(strategy.retry_delay_s)
            return await self._execute_with_improvement(subtask, attempt + 1)

    return result
```

---

## Accessing Improvement Data

```python
from synthron import Orchestrator
import asyncio

async def main():
    orch = Orchestrator()
    await orch.initialize()

    # Run some tasks...
    await orch.run("Task 1")
    await orch.run("Task 2")

    # Access improvement insights
    status = orch.status()
    print(status["performance"])
    # {"avg_score": 0.87, "success_rate": 0.92, "trend": "improving"}

    print(status["patterns"])
    # {"top_task_types": ["coding", "research"], "best_tools": {...}}

asyncio.run(main())
```
