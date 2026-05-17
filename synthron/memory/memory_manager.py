"""MemoryManager — unified interface to all memory subsystems."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from synthron.memory.episodic import EpisodicMemory
from synthron.memory.long_term import ChromaLongTermMemory, PineconeLongTermMemory
from synthron.memory.short_term import RedisShortTermMemory
from synthron.memory.working_memory import WorkingMemory
from synthron.utils.config import settings
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Unified API across all four memory stores.

    Memory hierarchy:
        working    → current task context (in-process, ephemeral)
        short_term → recent messages (Redis/in-memory, session-scoped, TTL 1h)
        long_term  → semantic embeddings (ChromaDB/Pinecone, persistent)
        episodic   → full run history (SQLite, persistent, for self-improvement)

    Usage:
        mm = MemoryManager()
        await mm.initialize()
        await mm.add_message("user", "Research AI companies")
        await mm.store_long_term("topic_ai_2026", "Key findings: ...")
        results = await mm.recall("AI funding 2026")
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._short_term = RedisShortTermMemory(self.session_id)
        self._long_term = ChromaLongTermMemory()
        self._long_term_cloud: PineconeLongTermMemory | None = None
        self._episodic = EpisodicMemory()
        self._working: WorkingMemory | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all memory subsystems."""
        if self._initialized:
            return

        results = await asyncio.gather(
            self._short_term.connect(),
            self._long_term.initialize(),
            self._episodic.initialize(),
            return_exceptions=True,
        )

        redis_ok, chroma_ok, sqlite_ok = results

        # Try Pinecone if API key set
        if settings.memory.pinecone_api_key:
            self._long_term_cloud = PineconeLongTermMemory()
            await self._long_term_cloud.initialize()

        logger.info(
            f"[memory] Initialized — "
            f"Redis: {'✅' if redis_ok else '⚠️'} | "
            f"Chroma: {'✅' if chroma_ok else '⚠️'} | "
            f"SQLite: {'✅' if sqlite_ok else '⚠️'}"
        )
        self._initialized = True

    def start_working(self, task: str) -> WorkingMemory:
        """Create a new working memory for the current task.

        Args:
            task: Current task description.

        Returns:
            Fresh WorkingMemory instance.
        """
        self._working = WorkingMemory(task=task, session_id=self.session_id)
        return self._working

    @property
    def working(self) -> WorkingMemory | None:
        """Access current working memory."""
        return self._working

    # ── Short-term operations ──────────────────────────────────────────────────

    async def add_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to short-term memory."""
        await self._short_term.add(role, content, metadata)

    async def get_short_term(self, last_n: int = 20) -> list[dict]:
        """Retrieve recent messages from short-term memory."""
        return await self._short_term.get_last_n(last_n)

    # ── Long-term operations ───────────────────────────────────────────────────

    async def store_long_term(
        self, key: str, content: str, metadata: dict | None = None
    ) -> None:
        """Store content in long-term vector memory.

        Writes to both local (ChromaDB) and cloud (Pinecone) if configured.

        Args:
            key: Unique identifier for this memory entry.
            content: Text content to store.
            metadata: Optional metadata tags.
        """
        await self._long_term.store(key, content, metadata)

        if self._long_term_cloud:
            try:
                await self._long_term_cloud.store(key, content, metadata)
            except Exception as exc:
                logger.debug(f"[memory] Pinecone store failed: {exc}")

    async def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search across long-term memory.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.

        Returns:
            List of relevant memory entries.
        """
        # Try local first
        results = await self._long_term.recall(query, top_k)

        # Supplement with cloud if available and local has few results
        if self._long_term_cloud and len(results) < top_k:
            try:
                cloud_results = await self._long_term_cloud.recall(
                    query, top_k - len(results)
                )
                results.extend(cloud_results)
            except Exception:
                pass

        return results[:top_k]

    # ── Episodic operations ────────────────────────────────────────────────────

    async def store_episode(
        self,
        task: str,
        plan: dict,
        results: list[dict],
        success: bool,
        tokens: int,
        time_s: float = 0.0,
        tags: list[str] | None = None,
    ) -> str:
        """Store a complete task run in episodic memory."""
        return await self._episodic.store_episode(
            task, plan, results, success, tokens, time_s, tags
        )

    async def get_similar_episodes(self, task: str, limit: int = 5) -> list[dict]:
        """Find similar past task runs."""
        return await self._episodic.search_similar(task, limit)

    async def get_performance_stats(self) -> dict[str, Any]:
        """Return overall performance statistics from episode history."""
        return await self._episodic.get_stats()

    async def record_pattern(
        self, pattern_key: str, description: str, success: bool, metadata: dict | None = None
    ) -> None:
        """Record a behavioral pattern observation for self-improvement."""
        await self._episodic.store_pattern(pattern_key, description, success, metadata)

    async def get_patterns(self) -> list[dict]:
        """Return all learned behavioral patterns."""
        return await self._episodic.get_patterns()

    # ── Convenience helpers ────────────────────────────────────────────────────

    async def remember_task_result(
        self, task: str, result: str, success: bool
    ) -> None:
        """Store a task result in both short-term and long-term memory.

        Args:
            task: Task description (used as memory key/query anchor).
            result: Task output to remember.
            success: Whether the task succeeded.
        """
        await self.add_message("assistant", f"Task: {task}\nResult: {result[:200]}")
        await self.store_long_term(
            key=f"task_{hash(task) % 10**8}",
            content=result,
            metadata={"task": task[:200], "success": success},
        )

    async def get_context_for_task(self, task: str) -> str:
        """Build a context string from all memory stores for a new task.

        Args:
            task: Incoming task description.

        Returns:
            Formatted context string for injection into prompts.
        """
        parts: list[str] = []

        # Short-term: recent conversation
        recent = await self.get_short_term(5)
        if recent:
            msgs = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')[:150]}"
                for m in recent
            )
            parts.append(f"[Recent context]\n{msgs}")

        # Long-term: semantic recall
        recalled = await self.recall(task, top_k=3)
        if recalled:
            memory_text = "\n".join(r["content"][:200] for r in recalled)
            parts.append(f"[Relevant memories]\n{memory_text}")

        # Similar episodes
        episodes = await self.get_similar_episodes(task, limit=2)
        if episodes:
            ep_text = "\n".join(
                f"- Past task: {ep.get('task', '')[:80]} ({'✅' if ep.get('success') else '❌'})"
                for ep in episodes
            )
            parts.append(f"[Similar past tasks]\n{ep_text}")

        return "\n\n".join(parts) if parts else ""

    async def memory_stats(self) -> dict[str, Any]:
        """Return comprehensive memory system statistics."""
        chroma_count = await self._long_term.count()
        episodic_stats = await self._episodic.get_stats()

        return {
            "session_id": self.session_id,
            "chroma_entries": chroma_count,
            "redis_available": self._short_term._redis_available,
            "pinecone_available": self._long_term_cloud is not None,
            "episodic": episodic_stats,
        }
