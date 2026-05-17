# Architecture

## Overview

SYNTHRON is built on four pillars:

1. **Smart Routing** — automatic LLM provider selection with failover
2. **Agent Pipeline** — Plan → Execute → Critique → Retry loop
3. **4-Tier Memory** — from ephemeral working memory to persistent vector storage
4. **Self-Improvement** — learns from every run to get better over time

---

## Core Pipeline

```
User Task
    │
    ▼
PlannerAgent          → creates DAG of subtasks using Gemini Pro
    │
    ▼
WorkflowEngine        → executes DAG in topological order
    │
    ├── ExecutorAgent (parallel × N)
    │       ├── select_tool()     → LLM picks best tool for subtask
    │       ├── run tool          → execute with safety sandbox
    │       └── LLM fallback      → if tool fails, LLM answers directly
    │
    ▼
CriticAgent           → scores result 0.0–1.0
    │
    ├── score ≥ 0.8   → PASS, store in memory
    ├── score ≥ 0.5   → WARN, continue with note
    └── score < 0.5   → FAIL, RetryStrategist decides retry vs skip
         │
         └── retry → back to ExecutorAgent with improved prompt
    │
    ▼
MemoryAgent           → stores task, result, patterns in all memory tiers
    │
    ▼
SelfImprovementEngine → FailureAnalyzer + PromptOptimizer + PatternLearner
    │
    ▼
Final Result
```

---

## Smart Router

The `SynthronRouter` manages all 7 LLM providers with daily token budgets:

```python
DAILY_LIMITS = {
    "gemini":     1_000_000,   # Google AI Studio free tier
    "groq":         500_000,   # Groq free tier (~14,400 req/day)
    "cerebras":   1_000_000,   # Cerebras free tier
    "deepseek":   5_000_000,   # DeepSeek $5 credit
    "openrouter":   300_000,   # OpenRouter $1 credit
    "github":       100_000,   # GitHub Models rate-limited
    "ollama":   999_999_999,   # Local, unlimited
}
```

**Agent routing** maps each agent type to its preferred provider chain:

```python
AGENT_ROUTING = {
    "planner":     ["gemini", "deepseek", "groq"],
    "executor":    ["groq", "cerebras", "gemini"],
    "critic":      ["gemini", "deepseek", "groq"],
    "researcher":  ["gemini", "groq", "openrouter"],
    "coder":       ["deepseek", "groq", "cerebras"],
    "coordinator": ["gemini", "deepseek", "groq"],
    "memory":      ["groq", "cerebras", "gemini"],
    "default":     ["gemini", "groq", "cerebras"],
}
```

**Failover flow**: If provider A fails (rate limit, API error, timeout), router automatically tries provider B, then C, then falls back to Ollama. Each attempt is logged with `ProviderError` details.

---

## 4-Tier Memory System

### Tier 1: Working Memory (ephemeral)
- Stored in Python dict, lost after task completes
- Holds: current context, tool results, agent outputs, shared data
- Access: `working_memory.build_context_for_executor()`

### Tier 2: Short-term (Redis)
- TTL: 24 hours by default
- Falls back to in-memory deque if Redis unavailable
- Stores: recent conversation turns, intermediate results
- Capacity: 100 entries rolling window

### Tier 3: Long-term (ChromaDB)
- Persistent vector storage with cosine similarity search
- Stores: task→result pairs, compressed summaries
- Query: semantic similarity with metadata filtering
- Also supports: Pinecone (cloud alternative)

### Tier 4: Episodic (SQLite)
- Structured timestamped log of all episodes
- Tables: `episodes` (task, result, score, duration) + `patterns` (task_type, tools, avg_score)
- Enables: trend analysis, pattern extraction, self-improvement

```python
# MemoryManager unified API
context = await memory.get_context_for_task(task, limit=5)
await memory.remember_task_result(task, result, score=0.9)
stats = await memory.memory_stats()
```

---

## DAG Workflow Engine

`WorkflowEngine` executes subtasks in parallel when dependencies allow:

```
subtask_1 ──┐
subtask_2 ──┤── subtask_4 ──┐
subtask_3 ──┘               ├── subtask_6 (final)
            subtask_5 ──────┘
```

**Algorithm:**
1. Find all subtasks with no pending dependencies (`ready`)
2. Launch all ready subtasks in parallel via `asyncio.gather`
3. As each completes, unlock dependents
4. Detect deadlock: if no tasks are running and some remain with unsatisfied deps → skip with error
5. Repeat until all tasks done

---

## Self-Improvement System

### FailureAnalyzer
- Tracks failures per tool, per error type, per agent
- Maintains `_known_fixes` dict: error pattern → suggested action
- Generates markdown reports: `failure_analyzer.generate_report()`

### PromptOptimizer
- Epsilon-greedy A/B testing: 80% exploit best variant, 20% explore new
- Uses LLM to generate improved prompt variants
- Tracks win/loss per variant with `PromptVariant.update_score()`

### PatternLearner
- Classifies tasks into categories: research, coding, math, writing, data, general
- Per-category tool recommendations from learned history
- Per-category prompt hints ("Focus on accuracy for math tasks")

### PerformanceTracker
- Rolling window of 100 entries: score, duration, tokens, success
- Computes: mean, P50/P95, trend (improving/declining/stable)
- `is_improving()`: compares last 10 vs previous 10 runs

### RetryStrategist
- Decides: RETRY vs SKIP based on score, max_retries, error type
- Suggests alternative tools when primary tool fails repeatedly
- Computes exponential backoff delay: `base_delay * 2^attempt`

---

## API Layer

### REST Endpoints

```
POST   /api/v1/tasks/run              → sync task execution
POST   /api/v1/tasks/async            → background task
GET    /api/v1/tasks/{task_id}        → poll task status
GET    /api/v1/tasks/stream/{sid}     → SSE event stream
GET    /api/v1/agents/status          → all agent states
GET    /api/v1/agents/providers       → provider token usage
GET    /api/v1/memory/recall          → semantic memory search
GET    /api/v1/memory/stats           → memory tier statistics
GET    /health                        → health check
GET    /docs                          → Swagger UI
```

### WebSocket

```
WS /ws/{session_id}      → per-session live events
WS /ws/events/all        → all events (dashboard use)
```

---

## EventBus

All inter-agent communication flows through the global `EventBus`:

```python
event_bus.publish(AgentEvent(
    agent_type="executor",
    event_type="action",
    content="Running web_search tool",
    metadata={"tool": "web_search", "query": "..."}
))
```

Subscribers can filter by event type or agent type. The API WebSocket handler subscribes to all events and pushes them to connected clients in real-time.

---

## Security Model

| Layer | Protection |
|-------|-----------|
| CodeExecutor | subprocess isolation, /tmp only, timeout |
| FileTool | workspace directory restriction |
| ApiCallerTool | blocks localhost, 169.254.x.x, 10.x.x.x |
| TerminalTool | SAFE_COMMANDS whitelist, BLOCKED_COMMANDS blacklist |
| Input validation | Pydantic v2 on all API inputs |
| Secrets | .env only, never in source code |
