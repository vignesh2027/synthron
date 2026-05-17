"""Browser tool — headless web browsing for content extraction."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class BrowserTool(BaseTool):
    """Headless browser for fetching and extracting web page content.

    Tries playwright first (full JS rendering), falls back to aiohttp
    for simple HTML fetching if playwright is not installed.
    """

    name = "browser_tool"
    description = (
        "Fetch and read the content of a web page. "
        "Renders JavaScript. Input: URL string."
    )
    category = "research"
    requires_network = True

    def __init__(self, max_content_chars: int = 8000, timeout: float = 20.0) -> None:
        self.max_content_chars = max_content_chars
        self.timeout = timeout
        self._playwright_available: bool | None = None

    async def run(self, input_text: str, context: Any = None) -> str:
        """Fetch and extract text content from a URL.

        Args:
            input_text: URL to fetch.
            context: Unused.

        Returns:
            Extracted text content from the page.
        """
        url = input_text.strip()
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        logger.debug(f"[browser] Fetching: {url}")

        # Try playwright first
        if await self._check_playwright():
            try:
                return await self._fetch_playwright(url)
            except Exception as exc:
                logger.debug(f"[browser] Playwright failed: {exc}, trying aiohttp")

        # Fallback to aiohttp
        return await self._fetch_aiohttp(url)

    async def _check_playwright(self) -> bool:
        """Check if playwright is installed and available."""
        if self._playwright_available is None:
            try:
                import playwright
                self._playwright_available = True
            except ImportError:
                self._playwright_available = False
        return self._playwright_available

    async def _fetch_playwright(self, url: str) -> str:
        """Fetch page using playwright for JS rendering."""
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=int(self.timeout * 1000), wait_until="domcontentloaded")
                await asyncio.sleep(1)  # wait for JS to settle

                # Extract text content
                text = await page.evaluate("""() => {
                    const remove = document.querySelectorAll('nav, footer, header, script, style, aside');
                    remove.forEach(el => el.remove());
                    return document.body ? document.body.innerText : '';
                }""")

                title = await page.title()
                clean_text = self._clean_text(text)

                return f"Page: {title}\nURL: {url}\n\n{clean_text}"
            finally:
                await browser.close()

    async def _fetch_aiohttp(self, url: str) -> str:
        """Fetch page using aiohttp (no JS rendering)."""
        import aiohttp

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ssl=True,
                    allow_redirects=True,
                ) as resp:
                    if resp.status >= 400:
                        return f"HTTP {resp.status} error for {url}"

                    content_type = resp.content_type or ""
                    if "html" not in content_type and "text" not in content_type:
                        return f"Non-text content type: {content_type}"

                    html = await resp.text(errors="replace")

            text = self._extract_text_from_html(html)
            return f"URL: {url}\n\n{text}"

        except aiohttp.ClientConnectorError as exc:
            raise ToolExecutionError("browser_tool", f"Connection failed: {exc}") from exc
        except asyncio.TimeoutError:
            raise ToolExecutionError("browser_tool", f"Timeout fetching: {url}")
        except Exception as exc:
            raise ToolExecutionError("browser_tool", str(exc)) from exc

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML without Beautiful Soup."""
        # Remove scripts, styles, and HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode HTML entities
        text = (
            text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
        )

        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Clean and truncate extracted text."""
        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

        if len(text) > self.max_content_chars:
            text = text[: self.max_content_chars] + f"\n\n[... truncated at {self.max_content_chars:,} chars]"

        return text.strip()
