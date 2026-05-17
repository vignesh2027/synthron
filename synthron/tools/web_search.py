"""Web search tool using DuckDuckGo (free, no API key required)."""

from __future__ import annotations

import asyncio
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """DuckDuckGo web search tool — no API key required.

    Returns the top N search results as formatted text including
    title, URL, and snippet for each result.
    """

    name = "web_search"
    description = "Search the web for current information using DuckDuckGo. Returns titles, URLs, and snippets."
    category = "research"
    requires_network = True

    def __init__(self, max_results: int = 8) -> None:
        self.max_results = max_results

    async def run(self, input_text: str, context: Any = None) -> str:
        """Execute a web search.

        Args:
            input_text: Search query string.
            context: Unused.

        Returns:
            Formatted string of search results.
        """
        query = input_text.strip()
        if not query:
            return "Empty search query."

        logger.debug(f"[web_search] Searching: '{query}'")

        try:
            results = await asyncio.to_thread(self._search_sync, query)
            if not results:
                return f"No results found for: '{query}'"
            return self._format_results(results, query)
        except ImportError:
            raise ToolExecutionError(
                "web_search",
                "duckduckgo-search is not installed. Run: pip install duckduckgo-search",
            )
        except Exception as exc:
            raise ToolExecutionError("web_search", str(exc)) from exc

    def _search_sync(self, query: str) -> list[dict]:
        """Run DuckDuckGo search synchronously (called via asyncio.to_thread)."""
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=self.max_results))

    def _format_results(self, results: list[dict], query: str) -> str:
        """Format raw search results into readable text.

        Args:
            results: List of result dicts with 'title', 'href', 'body' keys.
            query: Original search query.

        Returns:
            Formatted string.
        """
        lines = [f"Search results for: '{query}'\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("href", "")
            snippet = result.get("body", "")
            lines.append(f"{i}. **{title}**")
            if url:
                lines.append(f"   URL: {url}")
            if snippet:
                lines.append(f"   {snippet[:300]}")
            lines.append("")
        return "\n".join(lines)


class SerperSearchTool(BaseTool):
    """Serper.dev search tool — higher quality results (requires free API key)."""

    name = "serper_search"
    description = "High-quality Google search results via Serper.dev API."
    category = "research"
    requires_network = True

    def __init__(self, api_key: str = "") -> None:
        import os
        self._api_key = api_key or os.getenv("SERPER_API_KEY", "")

    async def run(self, input_text: str, context: Any = None) -> str:
        """Execute a Serper.dev search.

        Args:
            input_text: Search query.
            context: Unused.

        Returns:
            Formatted search results string.
        """
        if not self._api_key:
            # Fall back to DuckDuckGo if no key
            ddg = WebSearchTool()
            return await ddg.run(input_text, context)

        import aiohttp

        payload = {"q": input_text, "num": 8}
        headers = {"X-API-KEY": self._api_key, "Content-Type": "application/json"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/search",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()

            results = data.get("organic", [])
            lines = [f"Serper results for: '{input_text}'\n"]
            for i, r in enumerate(results[:8], 1):
                lines.append(f"{i}. **{r.get('title', '')}**")
                lines.append(f"   URL: {r.get('link', '')}")
                lines.append(f"   {r.get('snippet', '')[:300]}")
                lines.append("")
            return "\n".join(lines)

        except Exception as exc:
            raise ToolExecutionError("serper_search", str(exc)) from exc
