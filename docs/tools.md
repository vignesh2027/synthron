# Tools Reference

SYNTHRON includes 10 built-in tools. All tools implement `BaseTool` and are available via `ToolRegistry`.

## Using Tools

```python
from synthron import Synthron

# Use all default tools
agent = Synthron()

# Use specific tools by name
agent = Synthron(tools=["web_search", "code_executor", "calculator"])

# Get tool instances
from synthron.tools import get_default_tools, get_tools_by_names
tools = get_tools_by_names(["web_search", "file_tool"])
```

## Tool Reference

### web_search

Search the web using DuckDuckGo (no API key required).

**Input:** Search query string
**Output:** Top results with title, URL, and snippet

```python
# Used automatically by agents, or directly:
from synthron.tools import tool_registry
tool = tool_registry.get("web_search")
result = await tool.execute("Python async best practices 2026")
print(result.output)
```

Also available: `SerperSearchTool` (set `SERPER_API_KEY` for Google-quality results)

---

### browser_tool

Fetch and extract text content from any webpage.

**Input:** URL string
**Output:** Extracted text content (HTML stripped)

Uses Playwright when available (handles JS-rendered pages), falls back to aiohttp.

```bash
pip install playwright
playwright install chromium
```

---

### code_executor

Execute code safely in a subprocess sandbox.

**Input:** Code string (auto-detects language) or `{"language": "python", "code": "..."}`
**Output:** stdout/stderr, execution time, exit code

**Supported languages:** Python, JavaScript (Node.js), Bash, Ruby

**Safety:**
- Runs in `/tmp` temp directory
- 30-second timeout
- Output truncated at 10,000 chars
- `_is_safe()` checks for dangerous imports before execution

```python
result = await tool.execute("""
import math
primes = [n for n in range(2, 100) if all(n % i != 0 for i in range(2, n))]
print(primes)
""")
```

---

### file_tool

Read, write, and manage files within the workspace directory.

**Input:** JSON with `action` and parameters
**Output:** File contents or operation status

**Actions:**
- `read` — read file contents
- `write` — write/overwrite file
- `append` — append to file
- `list` — list directory contents
- `search` — search text in files (recursive grep)
- `exists` — check if file/dir exists
- `delete` — delete file

**Security:** All writes are restricted to the workspace directory. Path traversal (`../`) is blocked.

```json
{"action": "write", "path": "output/report.md", "content": "# My Report\n..."}
{"action": "read", "path": "data/input.csv"}
{"action": "search", "path": "src/", "query": "TODO"}
```

---

### calculator

Safe mathematical expression evaluator.

**Input:** Math expression string
**Output:** Numeric result

Uses AST parsing — no `eval()`, no code injection possible.

**Supported operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Functions: `sqrt`, `abs`, `floor`, `ceil`, `round`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `exp`
- Constants: `pi`, `e`, `inf`
- Comparison: `<`, `>`, `<=`, `>=`, `==`

```
Input:  "sqrt(2) * pi + log(100, 10)"
Output: "6.585..."

Input:  "sum([1, 2, 3, 4, 5])"
Output: "15"
```

---

### data_analyzer

Analyze CSV and JSON data files.

**Input:** File path or JSON data string
**Output:** Statistical summary, column info, sample rows

For CSV files with pandas available:
- Shape (rows × columns)
- Column types and null counts
- Numeric stats (mean, std, min, max, quartiles)
- Sample rows

For JSON: structure analysis, key counts, nested depth.

```json
{"file": "data/sales.csv"}
{"data": "[{\"name\": \"Alice\", \"score\": 95}, ...]"}
```

---

### api_caller

Make HTTP requests to external APIs.

**Input:** `"METHOD URL"` string or JSON with method/url/headers/body
**Output:** Response status, headers, body (truncated at 50KB)

**Security blocking:**
- Localhost (127.x.x.x, ::1)
- Link-local (169.254.x.x) — blocks cloud metadata endpoints
- Private networks (10.x.x.x, 192.168.x.x, 172.16-31.x.x)

```
Input:  "GET https://api.github.com/repos/python/cpython"
Output: {"status": 200, "body": {"stargazers_count": 62000, ...}}
```

---

### image_tool

Analyze images using Gemini Vision.

**Input:** Image URL or local file path, optional action
**Output:** Description, OCR text, or analysis

**Actions:** `describe`, `ocr`, `analyze`

Requires `GEMINI_API_KEY`. Supports JPEG, PNG, GIF, WebP.

```json
{"url": "https://example.com/chart.png", "action": "analyze"}
{"file": "screenshot.png", "action": "ocr"}
```

---

### terminal_tool

Run whitelisted shell commands.

**Input:** Shell command string
**Output:** stdout/stderr, exit code

**Safe commands whitelist:**
`ls`, `pwd`, `echo`, `cat`, `head`, `tail`, `grep`, `find`, `wc`, `sort`, `uniq`, `diff`, `date`, `env`, `which`, `python`, `pip`, `git`, `curl`, `wget`

**Blocked commands:**
`rm -rf`, `sudo`, `chmod 777`, `mkfs`, `dd`, `shutdown`, `reboot`, `passwd`, `su`

Set `strict_mode=True` to only allow exact whitelist matches.

---

### email_tool

Send and receive emails via SMTP/IMAP.

**Configuration:**
```bash
EMAIL_ADDRESS=your@email.com
EMAIL_PASSWORD=your_app_password
SMTP_HOST=smtp.gmail.com      # optional
SMTP_PORT=587                 # optional
IMAP_HOST=imap.gmail.com      # optional
```

**Actions:**
- `send` — send email with subject and body
- `read` — read recent emails from inbox

```json
{
  "action": "send",
  "to": "recipient@example.com",
  "subject": "Report Ready",
  "body": "Your analysis report is attached."
}
```

For Gmail: enable 2FA and use an App Password.

---

## Custom Tools

Implement `BaseTool` to add your own:

```python
from synthron.tools.base_tool import BaseTool, ToolResult, tool_registry

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    async def execute(self, input_data: str) -> ToolResult:
        # Your logic here
        result = f"Processed: {input_data}"
        return ToolResult(success=True, output=result)

# Register it
tool_registry.register(MyCustomTool())

# Use in agent
agent = Synthron(tools=["web_search", "my_tool"])
```

## Tool Registry

```python
from synthron.tools import tool_registry

# List all tools
tools = tool_registry.list_tools()
print(tools)  # ["web_search", "browser_tool", "code_executor", ...]

# Get tool schemas (for LLM function calling)
schemas = tool_registry.get_schemas()

# Get specific tool
tool = tool_registry.get("calculator")
result = await tool.execute("2 ** 32")
```
