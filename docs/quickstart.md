# Quick Start

## Installation

```bash
pip install synthron
```

Or from source:

```bash
git clone https://github.com/yourusername/synthron
cd synthron
pip install -e .
```

## Configuration

Copy the example env file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required: at least one of these
GEMINI_API_KEY=your_google_ai_studio_key
GROQ_API_KEY=your_groq_key

# Optional: more providers = more free tokens + better routing
CEREBRAS_API_KEY=your_cerebras_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENROUTER_API_KEY=your_openrouter_key
GITHUB_TOKEN=your_github_pat
```

**Getting free API keys:**
- Gemini: [aistudio.google.com](https://aistudio.google.com) → Get API key
- Groq: [console.groq.com](https://console.groq.com) → Create API key
- Both are free, no credit card required.

## 1-Line Usage

```python
import asyncio
from synthron import run

result = asyncio.run(run("What are the best Python libraries for data science in 2026?"))
print(result)
```

## Developer API

```python
import asyncio
from synthron import Synthron

async def main():
    # Create agent with specific tools
    agent = Synthron(tools=["web_search", "code_executor", "file_tool"])

    # Subscribe to live events
    def on_event(event):
        print(f"[{event['agent_type']}] {event['content'][:100]}")
    agent.subscribe(on_event)

    # Run a task
    result = await agent.run("Build a Python CLI tool for managing TODO lists")

    print(result.output)
    print(f"Success: {result.success}")
    print(f"Tokens: {result.total_tokens:,}")
    print(f"Time: {result.total_time_s:.1f}s")

asyncio.run(main())
```

## Full Orchestrator

```python
import asyncio
from synthron import Orchestrator

async def main():
    orch = Orchestrator(
        max_executors=3,       # parallel executors
        critic_threshold=0.75, # minimum score to accept result
    )
    await orch.initialize()

    result = await orch.run(
        "Research quantum computing breakthroughs in 2025-2026, "
        "summarize key papers, and write a 500-word technical overview"
    )

    print(result.output)

asyncio.run(main())
```

## CLI Usage

```bash
# Run a single task
synthron run "Analyze the sentiment of Apple's latest earnings call"

# Start REST API server
synthron serve --port 8080

# Interactive chat mode
synthron chat

# Check provider status
synthron status

# Run evaluation benchmark
synthron benchmark
```

## REST API

```bash
# Start server
synthron serve

# POST a task
curl -X POST http://localhost:8080/api/v1/tasks/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Compare Python and Rust for systems programming",
    "session_id": "my-session-123"
  }'

# Stream events via SSE
curl -N http://localhost:8080/api/v1/tasks/stream/my-session-123

# Check health
curl http://localhost:8080/health
```

## Docker

```bash
git clone https://github.com/yourusername/synthron
cd synthron
cp .env.example .env
# Edit .env

docker compose up -d
# API: http://localhost:8080/docs
# Dashboard: http://localhost:3000
```

## Context Managers

```python
async with Synthron(tools=["web_search"]) as agent:
    result = await agent.run("Latest news in AI")
    print(result.output)
# Automatically cleaned up
```

## Checking Status

```python
from synthron import Synthron
import asyncio

async def main():
    agent = Synthron()
    status = agent.status()
    print(status)
    # {
    #   "router": {"active_providers": ["gemini", "groq"], ...},
    #   "memory": {"short_term": {...}, "long_term": {...}},
    #   ...
    # }

asyncio.run(main())
```

## Next Steps

- [Providers Guide](providers.md) — configure all 7 providers
- [Tools Reference](tools.md) — all 10 built-in tools
- [Architecture](architecture.md) — how SYNTHRON works internally
- [Self-Improvement](self_improvement.md) — the learning system
- [Examples](../examples/) — 5 working examples
