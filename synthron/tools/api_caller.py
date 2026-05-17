"""API caller tool — make HTTP requests to any REST API."""

from __future__ import annotations

import json
from typing import Any

import aiohttp

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class ApiCallerTool(BaseTool):
    """Make HTTP requests to any REST API.

    Input format (JSON string):
    {
        "url": "https://api.example.com/endpoint",
        "method": "GET",            # GET, POST, PUT, PATCH, DELETE
        "headers": {},              # optional
        "params": {},               # query params
        "body": {}                  # request body for POST/PUT
    }

    Or simple string format: "GET https://api.example.com/endpoint"
    """

    name = "api_caller"
    description = "Make HTTP requests to REST APIs. Supports GET, POST, PUT, DELETE."
    category = "network"
    requires_network = True
    is_destructive = False

    BLOCKED_HOSTS = {
        "localhost", "127.0.0.1", "0.0.0.0", "::1",
        "169.254.169.254",  # AWS metadata
        "metadata.google.internal",
    }

    def __init__(self, timeout: float = 20.0, max_response_bytes: int = 500_000) -> None:
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    async def run(self, input_text: str, context: Any = None) -> str:
        """Execute an HTTP request.

        Args:
            input_text: JSON config or 'METHOD URL' shorthand.
            context: Unused.

        Returns:
            Response body as string (JSON pretty-printed if applicable).
        """
        config = self._parse_input(input_text)

        url = config.get("url", "")
        method = config.get("method", "GET").upper()
        headers = config.get("headers", {})
        params = config.get("params", {})
        body = config.get("body", None)

        if not url:
            return "Missing URL in API call config."

        self._check_url_safety(url)

        logger.debug(f"[api_caller] {method} {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=body if isinstance(body, dict) else None,
                    data=body if isinstance(body, str) else None,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ssl=True,
                ) as resp:
                    status = resp.status
                    content_type = resp.content_type or ""

                    raw = await resp.read()
                    if len(raw) > self.max_response_bytes:
                        raw = raw[: self.max_response_bytes]
                        truncated = True
                    else:
                        truncated = False

                    text = raw.decode("utf-8", errors="replace")

                    # Pretty-print JSON responses
                    if "json" in content_type:
                        try:
                            parsed = json.loads(text)
                            text = json.dumps(parsed, indent=2)
                        except Exception:
                            pass

                    result = f"HTTP {status} {method} {url}\n\n{text}"
                    if truncated:
                        result += f"\n\n[Response truncated at {self.max_response_bytes:,} bytes]"
                    return result

        except aiohttp.ClientConnectorError as exc:
            raise ToolExecutionError("api_caller", f"Connection error: {exc}") from exc
        except aiohttp.ClientResponseError as exc:
            raise ToolExecutionError("api_caller", f"HTTP error {exc.status}: {exc.message}") from exc
        except Exception as exc:
            raise ToolExecutionError("api_caller", str(exc)) from exc

    def _parse_input(self, text: str) -> dict:
        """Parse API call config from text.

        Supports:
        - JSON object: {"url": "...", "method": "GET"}
        - Simple: "GET https://..."
        - URL only: "https://..."
        """
        text = text.strip()

        # Try JSON
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Simple method + URL
        parts = text.split(None, 1)
        if len(parts) == 2 and parts[0].upper() in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            return {"method": parts[0].upper(), "url": parts[1]}

        # URL only
        if text.startswith(("http://", "https://")):
            return {"method": "GET", "url": text}

        return {"url": text, "method": "GET"}

    def _check_url_safety(self, url: str) -> None:
        """Block requests to localhost and cloud metadata endpoints."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or ""

        if host in self.BLOCKED_HOSTS:
            raise ToolExecutionError(
                "api_caller",
                f"Blocked request to restricted host: {host}",
            )
