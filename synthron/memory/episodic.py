"""Episodic memory — full run history stored in SQLite for replay and learning."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from synthron.utils.config import settings
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    plan JSON,
    results JSON,
    success INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    time_s REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    tags JSON
);

CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    pattern_key TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_seen REAL NOT NULL,
    metadata JSON
);

CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_patterns_key ON patterns(pattern_key);
"""


class EpisodicMemory:
    """Stores complete task run history for self-improvement and replay.

    Uses SQLite via aiosqlite for async persistence.
    Enables the self-improvement system to learn from past successes/failures.
    """

    def __init__(self, db_url: str = "") -> None:
        self.db_url = db_url or settings.memory.sqlite_url
        # Strip SQLAlchemy prefix for raw aiosqlite
        self._db_path = self.db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
        self._initialized = False

    async def initialize(self) -> bool:
        """Create database tables if not exist."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                await db.executescript(CREATE_TABLE_SQL)
                await db.commit()
            self._initialized = True
            logger.debug(f"[episodic] SQLite initialized: {self._db_path}")
            return True
        except ImportError:
            logger.warning("[episodic] aiosqlite not installed. Run: pip install aiosqlite")
            return False
        except Exception as exc:
            logger.error(f"[episodic] Init failed: {exc}")
            return False

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
        """Store a complete task run.

        Args:
            task: Original task string.
            plan: Serialized TaskPlan dict.
            results: List of SubTaskResult dicts.
            success: Overall success flag.
            tokens: Total tokens consumed.
            time_s: Total execution time in seconds.
            tags: Optional classification tags.

        Returns:
            Episode ID string.
        """
        episode_id = str(uuid.uuid4())
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """INSERT INTO episodes
                       (id, task, plan, results, success, tokens, time_s, created_at, tags)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        episode_id,
                        task,
                        json.dumps(plan),
                        json.dumps(results),
                        1 if success else 0,
                        tokens,
                        time_s,
                        time.time(),
                        json.dumps(tags or []),
                    ),
                )
                await db.commit()
            logger.debug(f"[episodic] Episode stored: {episode_id} (success={success})")
        except Exception as exc:
            logger.error(f"[episodic] Store failed: {exc}")
        return episode_id

    async def get_episode(self, episode_id: str) -> dict | None:
        """Retrieve a specific episode by ID."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT * FROM episodes WHERE id = ?", (episode_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return self._row_to_dict(row, cursor.description)
        except Exception as exc:
            logger.error(f"[episodic] Get failed: {exc}")
        return None

    async def get_recent(self, limit: int = 20, success_only: bool = False) -> list[dict]:
        """Retrieve recent episodes ordered by creation time.

        Args:
            limit: Maximum number of episodes to return.
            success_only: If True, only return successful episodes.

        Returns:
            List of episode dicts.
        """
        try:
            import aiosqlite
            where = "WHERE success = 1" if success_only else ""
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    f"SELECT * FROM episodes {where} ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ) as cursor:
                    rows = await cursor.fetchall()
                    desc = cursor.description
                    return [self._row_to_dict(r, desc) for r in rows]
        except Exception as exc:
            logger.error(f"[episodic] Get recent failed: {exc}")
            return []

    async def search_similar(self, task: str, limit: int = 5) -> list[dict]:
        """Find episodes with similar task descriptions using LIKE search.

        Args:
            task: Task query to search for.
            limit: Maximum results.

        Returns:
            List of similar episode dicts.
        """
        # Extract keywords for LIKE search
        keywords = [w for w in task.split() if len(w) > 4][:5]
        if not keywords:
            return await self.get_recent(limit)

        try:
            import aiosqlite
            conditions = " OR ".join([f"task LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords] + [limit]
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    f"SELECT * FROM episodes WHERE {conditions} ORDER BY created_at DESC LIMIT ?",
                    params,
                ) as cursor:
                    rows = await cursor.fetchall()
                    desc = cursor.description
                    return [self._row_to_dict(r, desc) for r in rows]
        except Exception as exc:
            logger.debug(f"[episodic] Search failed: {exc}")
            return []

    async def get_stats(self) -> dict[str, Any]:
        """Return overall performance statistics."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    """SELECT COUNT(*) as total,
                              SUM(success) as successes,
                              AVG(tokens) as avg_tokens,
                              AVG(time_s) as avg_time,
                              SUM(tokens) as total_tokens
                       FROM episodes"""
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        total, successes, avg_tokens, avg_time, total_tokens = row
                        return {
                            "total_episodes": total or 0,
                            "successes": successes or 0,
                            "failures": (total or 0) - (successes or 0),
                            "success_rate": (
                                round(successes / total, 3) if total else 0.0
                            ),
                            "avg_tokens": round(avg_tokens or 0, 1),
                            "avg_time_s": round(avg_time or 0, 2),
                            "total_tokens": total_tokens or 0,
                        }
        except Exception as exc:
            logger.debug(f"[episodic] Stats failed: {exc}")
        return {}

    async def store_pattern(
        self,
        pattern_key: str,
        description: str,
        success: bool,
        metadata: dict | None = None,
    ) -> None:
        """Record a behavioral pattern observation.

        Args:
            pattern_key: Unique pattern identifier (e.g., 'finance+web_search').
            description: Human-readable pattern description.
            success: Whether this pattern instance succeeded.
            metadata: Optional metadata.
        """
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """INSERT INTO patterns (id, pattern_key, description, success_count,
                       failure_count, last_seen, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(pattern_key) DO UPDATE SET
                         success_count = success_count + excluded.success_count,
                         failure_count = failure_count + excluded.failure_count,
                         last_seen = excluded.last_seen,
                         metadata = excluded.metadata""",
                    (
                        str(uuid.uuid4()),
                        pattern_key,
                        description,
                        1 if success else 0,
                        0 if success else 1,
                        time.time(),
                        json.dumps(metadata or {}),
                    ),
                )
                await db.commit()
        except Exception as exc:
            logger.debug(f"[episodic] Pattern store failed: {exc}")

    async def get_patterns(self, min_observations: int = 3) -> list[dict]:
        """Return learned patterns with enough observations."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    """SELECT * FROM patterns
                       WHERE (success_count + failure_count) >= ?
                       ORDER BY (success_count * 1.0 / (success_count + failure_count)) DESC""",
                    (min_observations,),
                ) as cursor:
                    rows = await cursor.fetchall()
                    desc = cursor.description
                    return [self._row_to_dict(r, desc) for r in rows]
        except Exception:
            return []

    def _row_to_dict(self, row: tuple, description: Any) -> dict:
        """Convert a sqlite3 row to a dict, parsing JSON fields."""
        if not description:
            return {}
        d = {description[i][0]: row[i] for i in range(len(description))}
        for json_field in ("plan", "results", "tags", "metadata"):
            if json_field in d and isinstance(d[json_field], str):
                try:
                    d[json_field] = json.loads(d[json_field])
                except Exception:
                    pass
        return d
