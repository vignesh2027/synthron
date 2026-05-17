"""Short-term memory — in-process ring buffer with optional Redis backing."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any

from synthron.utils.config import settings
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class InMemoryBuffer:
    """Fast in-process ring buffer for recent messages."""

    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def add(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to the buffer."""
        async with self._lock:
            self._buffer.append({
                "role": role,
                "content": content,
                "ts": time.time(),
                "metadata": metadata or {},
            })

    async def get_all(self) -> list[dict[str, Any]]:
        """Return all messages in chronological order."""
        async with self._lock:
            return list(self._buffer)

    async def get_last_n(self, n: int) -> list[dict[str, Any]]:
        """Return last N messages."""
        async with self._lock:
            items = list(self._buffer)
            return items[-n:] if len(items) >= n else items

    async def clear(self) -> None:
        """Clear all messages from buffer."""
        async with self._lock:
            self._buffer.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._buffer)

    def to_messages(self) -> list[dict[str, str]]:
        """Return messages in LLM-ready format (role + content only)."""
        return [{"role": m["role"], "content": m["content"]} for m in self._buffer]


class RedisShortTermMemory:
    """Redis-backed short-term memory with TTL expiry.

    Falls back to InMemoryBuffer if Redis is unavailable.
    """

    def __init__(self, session_id: str, ttl_seconds: int = 3600, max_size: int = 20) -> None:
        self.session_id = session_id
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._key = f"synthron:stm:{session_id}"
        self._redis = None
        self._fallback = InMemoryBuffer(max_size)
        self._redis_available = False

    async def connect(self) -> bool:
        """Attempt to connect to Redis."""
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(
                settings.memory.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=2,
            )
            await self._redis.ping()
            self._redis_available = True
            logger.debug(f"[short_term] Redis connected for session '{self.session_id}'")
            return True
        except Exception as exc:
            logger.debug(f"[short_term] Redis unavailable ({exc}), using in-memory fallback")
            self._redis_available = False
            return False

    async def add(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to short-term memory."""
        entry = json.dumps({
            "role": role,
            "content": content,
            "ts": time.time(),
            "metadata": metadata or {},
        })

        if self._redis_available and self._redis:
            try:
                async with self._redis.pipeline() as pipe:
                    pipe.rpush(self._key, entry)
                    pipe.ltrim(self._key, -self.max_size, -1)
                    pipe.expire(self._key, self.ttl)
                    await pipe.execute()
                return
            except Exception as exc:
                logger.debug(f"[short_term] Redis write failed: {exc}")

        await self._fallback.add(role, content, metadata)

    async def get_all(self) -> list[dict[str, Any]]:
        """Retrieve all messages."""
        if self._redis_available and self._redis:
            try:
                raw = await self._redis.lrange(self._key, 0, -1)
                return [json.loads(r) for r in raw]
            except Exception:
                pass

        return await self._fallback.get_all()

    async def get_last_n(self, n: int) -> list[dict[str, Any]]:
        """Retrieve last N messages."""
        if self._redis_available and self._redis:
            try:
                raw = await self._redis.lrange(self._key, -n, -1)
                return [json.loads(r) for r in raw]
            except Exception:
                pass

        return await self._fallback.get_last_n(n)

    async def clear(self) -> None:
        """Clear all messages for this session."""
        if self._redis_available and self._redis:
            try:
                await self._redis.delete(self._key)
            except Exception:
                pass
        await self._fallback.clear()

    def to_llm_messages(self, last_n: int = 10) -> list[dict[str, str]]:
        """Return recent messages in LLM format (synchronous snapshot)."""
        buffer = list(self._fallback._buffer)[-last_n:]
        return [{"role": m["role"], "content": m["content"]} for m in buffer]
