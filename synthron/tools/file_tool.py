"""File system tool — read, write, list, and search files."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

SAFE_BASE_DIRS = ["/tmp", os.path.expanduser("~/synthron_workspace")]


class FileTool(BaseTool):
    """File system operations: read, write, list, search files.

    For safety, write operations are restricted to allowed base directories.
    Read operations work on any path the process can access.
    """

    name = "file_tool"
    description = (
        "Read, write, list, and search files. "
        "Input format: 'ACTION:path[:content]'. "
        "Actions: read, write, list, search, exists."
    )
    category = "filesystem"
    requires_network = False
    is_destructive = True  # write/delete can modify data

    def __init__(self, workspace_dir: str = "") -> None:
        self.workspace_dir = workspace_dir or os.path.expanduser("~/synthron_workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)

    async def run(self, input_text: str, context: Any = None) -> str:
        """Execute a file operation.

        Input format: 'ACTION:path' or 'ACTION:path:content'
        Actions: read, write, append, list, search, exists, delete

        Args:
            input_text: Formatted action string.
            context: Unused.

        Returns:
            Operation result as string.
        """
        parts = input_text.split(":", 2)
        action = parts[0].strip().lower() if parts else ""
        path = parts[1].strip() if len(parts) > 1 else ""
        content = parts[2] if len(parts) > 2 else ""

        # If no ACTION: prefix, assume 'read'
        if not action or action not in ("read", "write", "append", "list", "search", "exists", "delete"):
            if os.path.exists(input_text.strip()):
                action, path = "read", input_text.strip()
            else:
                return f"Invalid file command: '{input_text}'. Use format: ACTION:path[:content]"

        actions = {
            "read": self._read,
            "write": self._write,
            "append": self._append,
            "list": self._list,
            "search": self._search,
            "exists": self._exists,
            "delete": self._delete,
        }

        handler = actions.get(action)
        if not handler:
            return f"Unknown action: '{action}'"

        try:
            return await asyncio.to_thread(handler, path, content)
        except Exception as exc:
            raise ToolExecutionError("file_tool", str(exc)) from exc

    def _read(self, path: str, _: str = "") -> str:
        """Read file contents."""
        full_path = self._resolve(path)
        if not os.path.exists(full_path):
            return f"File not found: {full_path}"
        if not os.path.isfile(full_path):
            return f"Not a file: {full_path}"

        size = os.path.getsize(full_path)
        if size > 1_000_000:  # 1MB limit
            return f"File too large ({size:,} bytes). Read a specific section."

        try:
            with open(full_path, encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            return f"File is binary (not text-readable): {full_path}"

    def _write(self, path: str, content: str = "") -> str:
        """Write content to file (restricted to workspace)."""
        full_path = self._resolve_workspace(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {full_path}"

    def _append(self, path: str, content: str = "") -> str:
        """Append content to file."""
        full_path = self._resolve_workspace(path)
        with open(full_path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} chars to {full_path}"

    def _list(self, path: str, _: str = "") -> str:
        """List directory contents."""
        full_path = self._resolve(path) if path else self.workspace_dir
        if not os.path.exists(full_path):
            return f"Path not found: {full_path}"
        if os.path.isfile(full_path):
            stat = os.stat(full_path)
            return f"{full_path} ({stat.st_size:,} bytes)"

        items = []
        try:
            for entry in sorted(os.scandir(full_path), key=lambda e: e.name):
                size_str = f" ({entry.stat().st_size:,}B)" if entry.is_file() else "/"
                items.append(f"  {entry.name}{size_str}")
            return f"Contents of {full_path}:\n" + "\n".join(items[:100])
        except PermissionError:
            return f"Permission denied: {full_path}"

    def _search(self, path: str, pattern: str = "") -> str:
        """Search for pattern in files within a directory."""
        if not pattern:
            return "Provide a search pattern: search:path:pattern"
        full_path = self._resolve(path) if path else self.workspace_dir
        matches = []
        try:
            for root, _, files in os.walk(full_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, encoding="utf-8", errors="ignore") as f:
                            for lineno, line in enumerate(f, 1):
                                if pattern.lower() in line.lower():
                                    matches.append(f"{fpath}:{lineno}: {line.strip()}")
                                    if len(matches) >= 50:
                                        break
                    except Exception:
                        continue
                if len(matches) >= 50:
                    break
        except Exception as exc:
            return f"Search error: {exc}"

        if not matches:
            return f"No matches for '{pattern}' in {full_path}"
        return f"Found {len(matches)} match(es):\n" + "\n".join(matches)

    def _exists(self, path: str, _: str = "") -> str:
        """Check if a path exists."""
        full_path = self._resolve(path)
        if os.path.exists(full_path):
            kind = "file" if os.path.isfile(full_path) else "directory"
            return f"EXISTS ({kind}): {full_path}"
        return f"NOT FOUND: {full_path}"

    def _delete(self, path: str, _: str = "") -> str:
        """Delete a file (workspace only)."""
        full_path = self._resolve_workspace(path)
        if not os.path.exists(full_path):
            return f"Not found: {full_path}"
        if os.path.isdir(full_path):
            return "Use rmdir to delete directories (not supported for safety)."
        os.unlink(full_path)
        return f"Deleted: {full_path}"

    def _resolve(self, path: str) -> str:
        """Resolve a path (may be absolute or relative to workspace)."""
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.workspace_dir, path))

    def _resolve_workspace(self, path: str) -> str:
        """Resolve path and ensure it's within the workspace directory."""
        resolved = self._resolve(path)
        if not resolved.startswith(os.path.normpath(self.workspace_dir)):
            # Also allow /tmp
            if not resolved.startswith("/tmp"):
                raise ToolExecutionError(
                    "file_tool",
                    f"Write access denied outside workspace: {resolved}",
                )
        return resolved
