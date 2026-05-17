"""Researcher Agent — deep web research with multi-source verification."""

from __future__ import annotations

import asyncio
from typing import Any

from synthron.agents.base_agent import AgentResult, BaseAgent, SubTask, SubTaskResult
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

_RESEARCHER_SYSTEM = """You are SYNTHRON's ResearcherAgent — a world-class research analyst.

Your capability: Deep web research, source verification, fact synthesis.

RESEARCH PROCESS:
1. Search multiple sources for the topic (minimum 2 searches with different queries).
2. Cross-verify facts across sources.
3. Extract key data points, numbers, and quotes.
4. Synthesize into a structured, cited report.
5. Flag any conflicting information or uncertainty.

ALWAYS:
- Include specific numbers and data when available.
- Note the source of key claims.
- Structure findings clearly (use headers and bullets).
- Be comprehensive but concise.

DO NOT:
- Make up data or speculate without flagging it.
- Repeat the same search query twice.
- Return raw search results without synthesis."""


class ResearcherAgent(BaseAgent):
    """Performs deep multi-step web research and synthesizes findings.

    Powered by Gemini 2.5 Flash (large context for processing search results).
    Uses web_search and browser_tool for data gathering.
    """

    name = "researcher"
    role = "researcher"
    agent_type = "researcher"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", _RESEARCHER_SYSTEM)
        super().__init__(**kwargs)

    def _default_system_prompt(self) -> str:
        return _RESEARCHER_SYSTEM

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Run research on a topic.

        Args:
            task: Research question or topic.
            context: Optional context dict.

        Returns:
            AgentResult with comprehensive research findings.
        """
        self._run_count += 1
        self._log.thought(f"Researching: {task[:80]}")
        await self._emit_event("researching", f"Starting research: {task[:80]}")

        result = await self.research(task, context=context)

        return AgentResult(
            agent_name=self.name,
            agent_type=self.agent_type,
            task=task,
            output=result.output,
            success=result.success,
            subtask_results=[result],
            total_tokens=self._total_tokens,
            total_latency_ms=self._total_latency_ms,
            error=result.error,
        )

    async def research(
        self, topic: str, depth: int = 2, context: dict[str, Any] | None = None
    ) -> SubTaskResult:
        """Perform multi-step research on a topic.

        Args:
            topic: Research topic or question.
            depth: Number of search iterations (1-3).
            context: Optional context from previous steps.

        Returns:
            SubTaskResult with synthesized research findings.
        """
        subtask = SubTask(title="Research", description=topic, tool_hint="web_search")
        search_tool = self.get_tool("web_search")
        collected_data: list[str] = []

        if search_tool:
            # Multi-step search with varied queries
            queries = self._generate_search_queries(topic, depth)
            self._log.action("web_search", f"{len(queries)} queries planned")

            search_tasks = [
                self._safe_search(search_tool, q, context) for q in queries
            ]
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._log.warning(f"Search {i+1} failed: {result}")
                elif result:
                    collected_data.append(f"[Search {i+1}: '{queries[i]}']\n{result}")

        raw_data = "\n\n---\n\n".join(collected_data) if collected_data else ""

        # Synthesize with LLM
        if raw_data:
            synthesis_prompt = (
                f"Research topic: {topic}\n\n"
                f"Raw search data:\n{raw_data[:6000]}\n\n"
                f"Synthesize this into a comprehensive, structured research report. "
                f"Include: key findings, data/numbers, source notes, and a summary."
            )
        else:
            synthesis_prompt = (
                f"Research topic: {topic}\n\n"
                f"No external search data available. Provide the best analysis "
                f"based on your training knowledge."
            )

        if context:
            context_str = "\n".join(f"- {k}: {str(v)[:200]}" for k, v in context.items())
            synthesis_prompt = f"Context:\n{context_str}\n\n{synthesis_prompt}"

        try:
            response = await self.generate(synthesis_prompt, max_tokens=4096, temperature=0.4)

            return SubTaskResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output=response.content,
                success=True,
                tool_used="web_search" if collected_data else "llm_knowledge",
                tokens_used=response.total_tokens,
                latency_ms=self._total_latency_ms,
            )

        except Exception as exc:
            self._log.error(f"Research synthesis failed: {exc}")
            return SubTaskResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output="",
                success=False,
                error=str(exc),
            )

    async def _safe_search(self, tool: Any, query: str, context: Any) -> str | None:
        """Run a single web search, handling exceptions gracefully."""
        try:
            result = await asyncio.wait_for(
                tool.run(query, context=context), timeout=15.0
            )
            return str(result) if result else None
        except asyncio.TimeoutError:
            self._log.warning(f"Search timed out: '{query}'")
            return None
        except Exception as exc:
            self._log.warning(f"Search error for '{query}': {exc}")
            return None

    def _generate_search_queries(self, topic: str, depth: int) -> list[str]:
        """Generate varied search queries for multi-step research.

        Args:
            topic: Research topic.
            depth: Number of queries to generate.

        Returns:
            List of search query strings.
        """
        queries = [topic]  # start with the exact topic

        if depth >= 2:
            # Add more specific query
            queries.append(f"{topic} data statistics 2025 2026")

        if depth >= 3:
            # Add analysis angle
            queries.append(f"{topic} analysis trends report")

        return queries[:depth]

    async def verify_claim(self, claim: str) -> dict[str, Any]:
        """Verify a specific claim using web search.

        Args:
            claim: The claim to verify.

        Returns:
            Dict with 'verified', 'confidence', 'evidence' keys.
        """
        search_tool = self.get_tool("web_search")
        if not search_tool:
            return {"verified": None, "confidence": 0.0, "evidence": "No search tool"}

        try:
            result = await search_tool.run(f"fact check: {claim}", context=None)
            prompt = (
                f"Claim to verify: {claim}\n\n"
                f"Search evidence: {str(result)[:2000]}\n\n"
                f"Is this claim verified? Reply as JSON: "
                f'{{"verified": true/false/null, "confidence": 0.0-1.0, "evidence": "..."}}'
            )
            response = await self.generate(prompt, max_tokens=300, temperature=0.2)

            import json, re
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as exc:
            self._log.warning(f"Claim verification failed: {exc}")

        return {"verified": None, "confidence": 0.0, "evidence": "Verification failed"}
