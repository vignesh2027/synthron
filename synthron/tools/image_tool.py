"""Image tool — describe and analyze images using vision models."""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class ImageTool(BaseTool):
    """Read and describe images using Gemini's vision capability.

    Supports:
    - Local image files (PNG, JPG, WEBP, GIF)
    - Remote image URLs
    - Image description and analysis
    - OCR-style text extraction from images
    """

    name = "image_tool"
    description = (
        "Read, describe, and analyze images. Supports local paths and URLs. "
        "Input: image path or URL, optionally prefixed with 'describe:', 'ocr:', or 'analyze:'"
    )
    category = "vision"
    requires_network = True

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def __init__(self) -> None:
        self._gemini_client = None

    async def run(self, input_text: str, context: Any = None) -> str:
        """Describe or analyze an image.

        Args:
            input_text: Image path/URL, optionally with 'describe:', 'ocr:', or 'analyze:' prefix.
            context: Optional dict with 'question' key for specific queries.

        Returns:
            Text description or analysis of the image.
        """
        text = input_text.strip()
        mode = "describe"
        question = ""

        # Parse mode prefix
        if text.startswith("ocr:"):
            mode = "ocr"
            text = text[4:].strip()
        elif text.startswith("analyze:"):
            mode = "analyze"
            text = text[8:].strip()
        elif text.startswith("describe:"):
            mode = "describe"
            text = text[9:].strip()

        if isinstance(context, dict):
            question = context.get("question", "")

        # Determine if URL or local file
        if text.startswith(("http://", "https://")):
            return await self._process_url(text, mode, question)
        else:
            return await self._process_file(text, mode, question)

    async def _process_file(self, path: str, mode: str, question: str) -> str:
        """Process a local image file."""
        if not os.path.exists(path):
            return f"Image file not found: {path}"

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            return f"Unsupported format: {ext}. Supported: {', '.join(self.SUPPORTED_FORMATS)}"

        file_size = os.path.getsize(path)
        if file_size > 10_000_000:  # 10MB limit
            return f"Image too large ({file_size:,} bytes). Max 10MB."

        try:
            with open(path, "rb") as f:
                image_data = f.read()
            return await self._describe_with_gemini(image_data, ext, mode, question)
        except Exception as exc:
            raise ToolExecutionError("image_tool", f"File read error: {exc}") from exc

    async def _process_url(self, url: str, mode: str, question: str) -> str:
        """Fetch and process an image from a URL."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        return f"Failed to fetch image: HTTP {resp.status}"
                    content_type = resp.content_type or ""
                    image_data = await resp.read()

            ext = self._ext_from_content_type(content_type)
            return await self._describe_with_gemini(image_data, ext, mode, question)
        except Exception as exc:
            raise ToolExecutionError("image_tool", f"URL fetch error: {exc}") from exc

    async def _describe_with_gemini(
        self, image_data: bytes, ext: str, mode: str, question: str
    ) -> str:
        """Send image to Gemini Vision for analysis."""
        try:
            import google.generativeai as genai
            from synthron.utils.config import settings

            if not settings.providers.gemini_api_key:
                return self._fallback_description(image_data, ext)

            genai.configure(api_key=settings.providers.gemini_api_key)

            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }
            mime = mime_map.get(ext, "image/jpeg")

            prompts = {
                "describe": "Describe this image in detail. What do you see?",
                "ocr": (
                    "Extract ALL text visible in this image. "
                    "Preserve formatting. Output only the extracted text."
                ),
                "analyze": (
                    "Analyze this image thoroughly: describe content, identify objects/people, "
                    "note any text, assess quality, and provide insights."
                ),
            }
            prompt = prompts.get(mode, prompts["describe"])
            if question:
                prompt = f"{prompt}\n\nSpecific question: {question}"

            model = genai.GenerativeModel("gemini-2.0-flash")
            image_part = {
                "mime_type": mime,
                "data": base64.b64encode(image_data).decode(),
            }

            response = await asyncio.to_thread(
                model.generate_content,
                [prompt, image_part],
            )
            return response.text or "Could not extract description from image."

        except ImportError:
            return self._fallback_description(image_data, ext)
        except Exception as exc:
            return f"Vision analysis failed: {exc}"

    def _fallback_description(self, image_data: bytes, ext: str) -> str:
        """Provide basic image info when vision model unavailable."""
        size_kb = len(image_data) / 1024
        return (
            f"Image info: {ext.upper()} format, {size_kb:.1f} KB. "
            f"Vision analysis requires a GEMINI_API_KEY."
        )

    def _ext_from_content_type(self, content_type: str) -> str:
        """Map MIME type to extension."""
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        for mime, ext in mapping.items():
            if mime in content_type:
                return ext
        return ".jpg"
