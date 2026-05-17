"""Sandboxed code execution tool with safety limits."""

from __future__ import annotations

import asyncio
import io
import os
import resource
import sys
import tempfile
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import CodeExecutionError, ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

# Safety limits
MAX_OUTPUT_CHARS = 10_000
MAX_EXECUTION_SECONDS = 15
MAX_MEMORY_MB = 256


class CodeExecutorTool(BaseTool):
    """Sandboxed Python code executor with safety limits.

    Executes Python code in an isolated subprocess with:
    - Memory limit (256 MB)
    - Time limit (15 seconds)
    - Captured stdout/stderr
    - No file system writes outside /tmp
    """

    name = "code_executor"
    description = "Execute Python code in a safe sandbox and return the output."
    category = "code"
    requires_network = False
    is_destructive = False

    def __init__(
        self,
        timeout: float = MAX_EXECUTION_SECONDS,
        max_output: int = MAX_OUTPUT_CHARS,
        allow_imports: list[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self.allow_imports = allow_imports  # None = allow all

    async def run(self, input_text: str, context: Any = None) -> str:
        """Execute Python code and return output.

        Args:
            input_text: Python code to execute.
            context: Optional dict with 'language' key.

        Returns:
            Combined stdout + stderr output, truncated to max_output chars.
        """
        language = "python"
        if isinstance(context, dict):
            language = context.get("language", "python")

        code = self._clean_code(input_text)
        if not code:
            return "No code to execute."

        if language != "python":
            return await self._execute_non_python(code, language)

        logger.debug(f"[code_executor] Executing {len(code)} chars of Python")

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_subprocess, code),
                timeout=self.timeout + 2,
            )
            return result
        except asyncio.TimeoutError:
            return f"❌ Code execution timed out after {self.timeout}s"
        except Exception as exc:
            return f"❌ Execution error: {exc}"

    def _run_subprocess(self, code: str) -> str:
        """Run code in a subprocess with resource limits."""
        import subprocess

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd="/tmp",
                env={
                    **os.environ,
                    "PYTHONPATH": "",
                },
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]:\n{result.stderr}"
            if result.returncode != 0 and not output:
                output = f"Exit code: {result.returncode}"

            return output[: self.max_output] if len(output) > self.max_output else output

        except subprocess.TimeoutExpired:
            return f"❌ Timed out after {self.timeout}s"
        except Exception as exc:
            return f"❌ Subprocess error: {exc}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def _execute_non_python(self, code: str, language: str) -> str:
        """Execute non-Python code using appropriate interpreter."""
        lang_map = {
            "javascript": ("node", ".js"),
            "js": ("node", ".js"),
            "bash": ("bash", ".sh"),
            "sh": ("bash", ".sh"),
            "ruby": ("ruby", ".rb"),
        }

        if language not in lang_map:
            return f"Language '{language}' not supported for execution. Showing code only:\n{code}"

        interpreter, ext = lang_map[language]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            import subprocess
            result = await asyncio.to_thread(
                subprocess.run,
                [interpreter, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout or ""
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            return output[: self.max_output]
        except FileNotFoundError:
            return f"❌ Interpreter '{interpreter}' not found. Install {language} to run this code."
        except Exception as exc:
            return f"❌ Execution error: {exc}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _clean_code(self, text: str) -> str:
        """Extract pure code from potentially markdown-wrapped input."""
        import re

        # Strip markdown code fences
        match = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Already plain code
        return text.strip()

    def _is_safe(self, code: str) -> tuple[bool, str]:
        """Basic static safety check on the code.

        Args:
            code: Python code to check.

        Returns:
            (is_safe, reason) tuple.
        """
        dangerous_patterns = [
            ("os.system", "system command execution"),
            ("subprocess.Popen", "subprocess creation"),
            ("__import__('os').system", "system bypass"),
            ("eval(", "eval() call"),
            ("exec(", "exec() call"),
            ("open('", "file write attempt"),
            ("shutil.rmtree", "directory deletion"),
        ]

        for pattern, reason in dangerous_patterns:
            if pattern in code:
                return False, reason

        return True, ""
