"""Memory Agent — manages all memory operations across the agent system."""

from __future__ import annotations

from typing import Any

from synthron.agents.base_agent import AgentResult, BaseAgent
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

_MEMORY_SYSTEM = """You are SYNTHRON's MemoryAgent — the system's knowledge keeper.

Your responsibilities:
1. STORE: Save important results, facts, and patterns to long-term memory.
2. RECALL: Retrieve relevant past information for the current task.
3. SUMMARIZE: Condense large outputs into key facts for future use.
4. FORGET: Identify and remove stale or irrelevant information.

When storing, extract the key facts, entities, and conclusions.
When recalling, return only what is directly relevant to the query.
Be concise and structured in all outputs."""


class MemoryAgent(BaseAgent):
    """Manages memory operations: store, recall, summarize, and forget.

    Connects to the MemoryManager (short-term, long-term, episodic).
    Powered by Groq (fast + reliable) for quick memory operations.
    """

    name = "memory"
    role = "memory"
    agent_type = "memory"

    def __init__(self, memory_manager: Any = None, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", _MEMORY_SYSTEM)
        super().__init__(**kwargs)
        self._memory_manager = memory_manager  # set by orchestrator

    def _default_system_prompt(self) -> str:
        return _MEMORY_SYSTEM

    def attach_memory(self, memory_manager: Any) -> None:
        """Attach the MemoryManager instance."""
        self._memory_manager = memory_manager

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Run memory agent to recall relevant context for a task.

        Args:
            task: Query to retrieve relevant memories for.
            context: Optional context dict.

        Returns:
            AgentResult with recalled memories in output.
        """
        self._run_count += 1
        recalled = await self.recall(task)

        return AgentResult(
            agent_name=self.name,
            agent_type=self.agent_type,
            task=task,
            output=recalled,
            success=True,
            total_tokens=self._total_tokens,
            total_latency_ms=self._total_latency_ms,
        )

    async def store(self, key: str, content: str, metadata: dict | None = None) -> None:
        """Store content in long-term memory.

        Args:
            key: Identifier for this memory entry.
            content: The content to store.
            metadata: Optional metadata tags.
        """
        self._log.action("store", f"Key='{key}' ({len(content)} chars)")
        await self._emit_event("memory_store", f"Storing: {key}")

        if self._memory_manager:
            try:
                await self._memory_manager.store_long_term(key, content, metadata or {})
            except Exception as exc:
                self._log.warning(f"Long-term store failed: {exc}")

    async def recall(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant memories for a query.

        Args:
            query: The search query.
            top_k: Number of results to retrieve.

        Returns:
            Concatenated relevant memory content as a string.
        """
        self._log.thought(f"Recalling memories for: '{query[:60]}'")
        await self._emit_event("memory_recall", f"Searching: {query[:60]}")

        if self._memory_manager:
            try:
                results = await self._memory_manager.recall(query, top_k=top_k)
                if results:
                    return "\n\n".join(r["content"] for r in results if "content" in r)
            except Exception as exc:
                self._log.warning(f"Memory recall failed: {exc}")

        # Fallback: check short-term (in-memory) history via LLM
        return ""

    async def summarize(self, content: str, topic: str = "") -> str:
        """Summarize content for efficient storage.

        Args:
            content: Content to summarize.
            topic: Optional topic hint.

        Returns:
            Concise summary string.
        """
        self._log.thought(f"Summarizing {len(content)} chars")

        prompt = (
            f"Summarize the following content into key facts and conclusions "
            f"(max 200 words):{f' Topic: {topic}' if topic else ''}\n\n{content[:4000]}"
        )
        try:
            response = await self.generate(prompt, max_tokens=300, temperature=0.3)
            return response.content.strip()
        except Exception as exc:
            self._log.warning(f"Summarization failed: {exc}")
            return content[:500] + ("..." if len(content) > 500 else "")

    async def store_episode(
        self,
        task: str,
        plan_data: dict,
        results: list[dict],
        success: bool,
        total_tokens: int,
    ) -> None:
        """Store a complete task run in episodic memory.

        Args:
            task: Original task string.
            plan_data: Serialized TaskPlan dict.
            results: List of SubTaskResult dicts.
            success: Whether the overall task succeeded.
            total_tokens: Total tokens consumed.
        """
        self._log.action("store_episode", f"Task: '{task[:60]}', success={success}")

        if self._memory_manager:
            try:
                await self._memory_manager.store_episode(
                    task=task,
                    plan=plan_data,
                    results=results,
                    success=success,
                    tokens=total_tokens,
                )
            except Exception as exc:
                self._log.warning(f"Episode store failed: {exc}")

    async def get_relevant_context(self, task: str) -> dict[str, str]:
        """Get all relevant context from all memory stores for a new task.

        Args:
            task: The incoming task query.

        Returns:
            Dict with 'short_term', 'long_term', 'patterns' keys.
        """
        context: dict[str, str] = {}

        if not self._memory_manager:
            return context

        try:
            # Short-term: recent messages
            short = await self._memory_manager.get_short_term()
            if short:
                context["short_term"] = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')[:200]}"
                    for m in short[-5:]
                )

            # Long-term: semantic search
            long_results = await self._memory_manager.recall(task, top_k=3)
            if long_results:
                context["long_term"] = "\n\n".join(
                    r.get("content", "")[:500] for r in long_results
                )

        except Exception as exc:
            self._log.warning(f"Context retrieval failed: {exc}")

        return context
