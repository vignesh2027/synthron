# Providers Guide

SYNTHRON supports 7 LLM providers with automatic failover routing.

## Provider Comparison

| Provider | Free Tier | Daily Limit | Speed | Best For |
|----------|-----------|-------------|-------|----------|
| Gemini | ✅ 1M tokens | 1,000,000 | Fast | Planning, vision |
| Groq | ✅ Free | 14,400 req | Fastest | Execution |
| Cerebras | ✅ 1M tokens | 1,000,000 | Ultra-fast | Code |
| DeepSeek | $5 credit | 5,000,000 | Medium | Reasoning |
| OpenRouter | $1 credit | ~300,000 | Varies | Diversity |
| GitHub Models | ✅ Rate-limited | ~100,000 | Medium | Dev testing |
| Ollama | ✅ Local | Unlimited | Depends | Privacy |

## Setup

### Gemini (Google AI Studio)

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click "Get API key" → "Create API key"
3. Add to `.env`: `GEMINI_API_KEY=your_key`

Default model: `gemini-1.5-flash` (fast, free)
Premium model: `gemini-1.5-pro` (set `GEMINI_MODEL=gemini-1.5-pro`)

```bash
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-1.5-flash  # optional
```

### Groq

1. Go to [console.groq.com](https://console.groq.com)
2. Create account → API Keys → Create key
3. Add to `.env`: `GROQ_API_KEY=your_key`

Default model: `llama-3.3-70b-versatile`

```bash
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile  # optional
```

Available free models:
- `llama-3.3-70b-versatile` (default)
- `llama-3.1-8b-instant` (faster)
- `mixtral-8x7b-32768` (long context)
- `gemma2-9b-it`

### Cerebras

1. Go to [cloud.cerebras.ai](https://cloud.cerebras.ai)
2. Sign up → Get API key
3. Add to `.env`: `CEREBRAS_API_KEY=your_key`

Default model: `llama3.1-70b`

```bash
CEREBRAS_API_KEY=csk_...
```

Cerebras runs on custom silicon — up to 2100 tokens/second.

### DeepSeek

1. Go to [platform.deepseek.com](https://platform.deepseek.com)
2. Register → API Keys → Create
3. New accounts get $5 free credit
4. Add to `.env`: `DEEPSEEK_API_KEY=your_key`

```bash
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat  # or deepseek-reasoner
```

DeepSeek-R1 is excellent for complex reasoning tasks.

### OpenRouter

1. Go to [openrouter.ai](https://openrouter.ai)
2. Create account → Keys → Create key
3. New accounts get $1 free credit
4. Add to `.env`: `OPENROUTER_API_KEY=your_key`

```bash
OPENROUTER_API_KEY=sk-or-...
```

OpenRouter gives access to 100+ models. SYNTHRON defaults to free models:
- `meta-llama/llama-3.1-8b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`

### GitHub Models

1. Go to [github.com/marketplace/models](https://github.com/marketplace/models)
2. Use your existing GitHub PAT or create one with no special scopes
3. Add to `.env`: `GITHUB_TOKEN=your_pat`

```bash
GITHUB_TOKEN=ghp_...
GITHUB_MODEL=gpt-4o-mini  # optional
```

Available models: GPT-4o, GPT-4o-mini, Llama 3.1, Mistral, Phi-3

### Ollama (Local)

1. Install: [ollama.ai](https://ollama.ai)
2. Start: `ollama serve`
3. Pull a model: `ollama pull llama3.1`

```bash
OLLAMA_BASE_URL=http://localhost:11434  # optional, this is default
OLLAMA_MODEL=llama3.1  # optional
```

No API key needed. Runs 100% locally. Unlimited tokens.

Recommended models:
- `llama3.1` (general)
- `codellama` (coding)
- `mistral` (fast)

## Custom Routing

Override the default routing per agent type:

```python
from synthron import Orchestrator

orch = Orchestrator(
    # Uses default AGENT_ROUTING from smart_router.py
)

# Or configure provider preferences via env:
# PROVIDER_ORDER=gemini,deepseek,groq,cerebras
```

## Checking Provider Status

```bash
synthron status
```

```
Provider Status
─────────────────────────────────────
gemini      ✅ active  | used: 12,450 / 1,000,000 tokens today
groq        ✅ active  | used: 2,100 / 500,000 tokens today
cerebras    ✅ active  | used: 8,900 / 1,000,000 tokens today
deepseek    ⚠️ no key  | set DEEPSEEK_API_KEY to enable
openrouter  ⚠️ no key  | set OPENROUTER_API_KEY to enable
github      ⚠️ no key  | set GITHUB_TOKEN to enable
ollama      ⚠️ offline | start with: ollama serve
```

## Failover Behavior

When a provider fails:
1. Request is retried with `RetryStrategist.decide()` (respects max retries)
2. Next provider in agent's priority chain is tried
3. Each failure is logged and recorded in `FailureAnalyzer`
4. If all providers in chain fail, Ollama is tried as final fallback
5. If Ollama is unavailable, `MaxRetriesExceededError` is raised

Errors that trigger immediate failover (no retry on same provider):
- `RateLimitError` — daily/minute limit exceeded
- `ProviderError` with 401/403 — bad API key

Errors that retry with backoff:
- Timeouts
- 500/503 server errors
- Network errors

## Token Budget Tracking

```python
from synthron.utils.token_counter import daily_tracker

# Check current usage
report = daily_tracker.get_report()
print(report)
# {
#   "gemini": {"used": 12450, "limit": 1000000, "remaining": 987550},
#   "groq": {"used": 2100, "limit": 500000, "remaining": 497900},
#   ...
# }
```

Budgets reset at midnight UTC.
