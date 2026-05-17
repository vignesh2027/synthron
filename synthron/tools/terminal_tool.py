"""Terminal tool — run shell commands with safety constraints."""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

# Blocked commands that could cause harm
BLOCKED_COMMANDS = frozenset({
    "rm", "rmdir", "mkfs", "dd", "fdisk", "format",
    "shutdown", "reboot", "halt", "poweroff",
    "iptables", "ufw", "firewall-cmd",
    "passwd", "useradd", "userdel",
    "sudo", "su", "chmod 777",
    "curl | bash", "wget | bash",
})

# Only allow these safe commands by default
SAFE_COMMANDS = frozenset({
    "ls", "echo", "cat", "head", "tail", "wc", "grep", "find",
    "pwd", "date", "whoami", "uname", "df", "du", "free",
    "ps", "top", "which", "type", "env", "printenv",
    "python3", "python", "node", "pip", "pip3",
    "git", "curl", "wget", "ping",
    "mkdir", "touch", "cp", "mv",
    "sort", "uniq", "cut", "awk", "sed", "tr",
    "zip", "unzip", "tar",
    "jq", "yq",
})


class TerminalTool(BaseTool):
    """Execute shell commands with safety guardrails.

    By default, only allows a whitelist of safe commands.
    Destructive operations (rm, sudo, etc.) are blocked.
    All commands run with a timeout.
    """

    name = "terminal_tool"
    description = (
        "Run shell commands safely. Blocked: rm, sudo, and other destructive ops. "
        "Input: shell command string."
    )
    category = "system"
    requires_network = False
    is_destructive = True

    def __init__(
        self,
        timeout: float = 15.0,
        max_output: int = 10_000,
        strict_mode: bool = True,
        working_dir: str = "/tmp",
    ) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self.strict_mode = strict_mode
        self.working_dir = working_dir

    async def run(self, input_text: str, context: Any = None) -> str:
        """Execute a shell command.

        Args:
            input_text: Shell command to execute.
            context: Optional dict with 'cwd' key for working directory.

        Returns:
            Command stdout + stderr as string.
        """
        command = input_text.strip()
        if not command:
            return "Empty command."

        # Safety check
        is_safe, reason = self._is_safe(command)
        if not is_safe:
            return f"❌ Blocked: {reason}\nCommand: {command}"

        cwd = self.working_dir
        if isinstance(context, dict) and "cwd" in context:
            cwd = context["cwd"]

        logger.debug(f"[terminal] Running: {command}")

        try:
            result = await asyncio.wait_for(
                self._run_command(command, cwd),
                timeout=self.timeout + 2,
            )
            return result
        except asyncio.TimeoutError:
            return f"❌ Command timed out after {self.timeout}s: {command}"

    async def _run_command(self, command: str, cwd: str) -> str:
        """Run command as subprocess."""
        import subprocess

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env={**os.environ, "HOME": os.environ.get("HOME", "/tmp")},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return f"❌ Timed out after {self.timeout}s"

            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                output += f"\n[stderr]: {stderr.decode('utf-8', errors='replace')}"

            if proc.returncode != 0 and not output.strip():
                output = f"Exit code: {proc.returncode}"

            return output[: self.max_output]

        except Exception as exc:
            raise ToolExecutionError("terminal_tool", str(exc)) from exc

    def _is_safe(self, command: str) -> tuple[bool, str]:
        """Check if a command is safe to execute.

        Args:
            command: Shell command string.

        Returns:
            (is_safe, reason) tuple.
        """
        cmd_lower = command.lower().strip()

        # Check for blocked commands
        for blocked in BLOCKED_COMMANDS:
            if re.search(rf"\b{re.escape(blocked)}\b", cmd_lower):
                return False, f"Blocked command: '{blocked}'"

        # Strict mode: only allow whitelist
        if self.strict_mode:
            first_token = shlex.split(command)[0] if command else ""
            base = os.path.basename(first_token)
            if base not in SAFE_COMMANDS:
                return False, (
                    f"Command '{base}' not in safe list. "
                    f"Allowed: {', '.join(sorted(SAFE_COMMANDS))}"
                )

        # Block path traversal
        if "../" in command and "/tmp" not in command:
            return False, "Path traversal detected"

        return True, ""


import re  # noqa: E402 (needed after class for _is_safe)
