# Contributing to SYNTHRON

Thank you for contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/yourusername/synthron
cd synthron
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
# Add at least GEMINI_API_KEY and GROQ_API_KEY
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=synthron --cov-report=term-missing
```

## Code Style

```bash
ruff check synthron/ tests/
black synthron/ tests/
mypy synthron/
```

These run automatically in CI on every PR.

## Adding a New Tool

1. Create `synthron/tools/my_tool.py` inheriting from `BaseTool`
2. Add to `DEFAULT_TOOLS` in `synthron/tools/__init__.py`
3. Add tests in `tests/test_tools.py`
4. Document in `docs/tools.md`

```python
from synthron.tools.base_tool import BaseTool, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does X given Y input"

    async def execute(self, input_data: str) -> ToolResult:
        output = do_something(input_data)
        return ToolResult(success=True, output=str(output))
```

## Adding a New Provider

1. Create `synthron/providers/my_provider.py` inheriting from `BaseProvider`
2. Add to `_load_provider()` in `smart_router.py`
3. Add `DAILY_LIMITS` entry
4. Add to `AGENT_ROUTING` where appropriate
5. Add env var to `.env.example`

## Submitting a PR

1. Fork the repo
2. Create a branch: `git checkout -b feat/my-feature`
3. Write tests for your changes
4. Ensure CI passes: `pytest && ruff check && black --check`
5. Open a PR with a clear description

## Reporting Issues

Use [GitHub Issues](https://github.com/yourusername/synthron/issues). Include:
- Python version
- Error traceback
- Minimal reproduction case
- Which provider(s) you're using

## License

By contributing, you agree your contributions will be licensed under MIT.
