"""Async priority task queue for the orchestrator."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from synthron.agents.base_agent import SubTask, TaskStatus
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(order=True)
class QueueEntry:
    """A subtask entry in the priority queue."""

    priority: int                           # lower = higher priority (heapq convention)
    created_at: float = field(compare=False)
    subtask: SubTask = field(compare=False)
    context: dict[str, Any] = field(compare=False, default_factory=dict)

    @classmethod
    def from_subtask(cls, subtask: SubTask, context: dict | None = None) -> "QueueEntry":
        return cls(
            priority=10 - subtask.priority,   # invert: higher priority → lower queue value
            created_at=time.monotonic(),
            subtask=subtask,
            context=context or {},
        )


class AsyncTaskQueue:
    """Async priority queue for subtask scheduling.

    Supports:
    - Priority-based ordering (1-10 scale, 10=urgent)
    - Dependency tracking (subtasks wait for deps)
    - Concurrent worker limit
    - Queue inspection and drain
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        self._queue: asyncio.PriorityQueue[QueueEntry] = asyncio.PriorityQueue()
        self._in_flight: dict[str, QueueEntry] = {}
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._lock = asyncio.Lock()
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._total_enqueued = 0
        self._total_processed = 0

    async def enqueue(self, subtask: SubTask, context: dict | None = None) -> None:
        """Add a subtask to the queue.

        Args:
            subtask: SubTask to enqueue.
            context: Shared context dict.
        """
        entry = QueueEntry.from_subtask(subtask, context)
        await self._queue.put(entry)
        self._total_enqueued += 1
        logger.debug(
            f"[queue] Enqueued '{subtask.title}' (priority={subtask.priority}, "
            f"deps={subtask.depends_on})"
        )

    async def enqueue_batch(
        self, subtasks: list[SubTask], context: dict | None = None
    ) -> None:
        """Enqueue a batch of subtasks."""
        for st in subtasks:
            await self.enqueue(st, context)

    async def dequeue(self, timeout: float = 5.0) -> QueueEntry | None:
        """Dequeue the highest-priority ready subtask.

        Args:
            timeout: Seconds to wait for an item.

        Returns:
            QueueEntry or None if timeout.
        """
        try:
            entry = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            async with self._lock:
                self._in_flight[entry.subtask.id] = entry
                entry.subtask.status = TaskStatus.RUNNING
            return entry
        except asyncio.TimeoutError:
            return None

    async def complete(self, subtask_id: str, success: bool = True) -> None:
        """Mark a subtask as complete.

        Args:
            subtask_id: ID of the completed subtask.
            success: Whether it succeeded.
        """
        async with self._lock:
            self._in_flight.pop(subtask_id, None)
            if success:
                self._completed.add(subtask_id)
            else:
                self._failed.add(subtask_id)
            self._total_processed += 1
            self._queue.task_done()

    def are_deps_met(self, subtask: SubTask) -> bool:
        """Check if all dependencies for a subtask are completed."""
        return all(dep in self._completed for dep in subtask.depends_on)

    async def drain(self) -> None:
        """Wait until all enqueued tasks are processed."""
        await self._queue.join()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    @property
    def failed_count(self) -> int:
        return len(self._failed)

    def is_empty(self) -> bool:
        return self._queue.empty() and not self._in_flight

    def stats(self) -> dict:
        return {
            "pending": self.pending_count,
            "in_flight": self.in_flight_count,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "total_enqueued": self._total_enqueued,
            "total_processed": self._total_processed,
        }
