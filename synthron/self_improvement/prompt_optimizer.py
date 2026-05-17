"""Prompt Optimizer — A/B test and auto-improve agent prompts."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class PromptVariant:
    """A single prompt variant with performance tracking."""

    def __init__(self, prompt: str, name: str = "") -> None:
        self.prompt = prompt
        self.name = name or hashlib.md5(prompt.encode()).hexdigest()[:8]
        self.uses = 0
        self.total_score = 0.0
        self.successes = 0
        self.failures = 0
        self.created_at = time.time()

    def record(self, score: float, success: bool) -> None:
        self.uses += 1
        self.total_score += score
        if success:
            self.successes += 1
        else:
            self.failures += 1

    @property
    def avg_score(self) -> float:
        return self.total_score / self.uses if self.uses else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uses": self.uses,
            "avg_score": round(self.avg_score, 3),
            "success_rate": round(self.success_rate, 3),
        }


class PromptOptimizer:
    """Tracks prompt performance and selects the best variant for each agent type.

    Implements a simple epsilon-greedy strategy:
    - 80% of the time: use the best performing prompt (exploit)
    - 20% of the time: try a variant (explore)

    After enough data (≥10 uses per variant), locks in the winner.
    """

    EXPLORE_RATE = 0.2
    MIN_USES_TO_COMPARE = 10

    def __init__(self) -> None:
        self._variants: dict[str, list[PromptVariant]] = {}
        self._improvements: list[dict[str, Any]] = []

    def register(self, agent_type: str, prompt: str, name: str = "") -> None:
        """Register a prompt variant for an agent type.

        Args:
            agent_type: Agent role identifier (planner, executor, etc.)
            prompt: The system prompt text.
            name: Optional variant name.
        """
        if agent_type not in self._variants:
            self._variants[agent_type] = []

        variant = PromptVariant(prompt, name)
        self._variants[agent_type].append(variant)
        logger.debug(f"[prompt_optimizer] Registered variant '{variant.name}' for {agent_type}")

    def get_best_prompt(self, agent_type: str) -> str | None:
        """Return the best prompt for an agent type using epsilon-greedy selection.

        Args:
            agent_type: Agent role identifier.

        Returns:
            Best prompt string, or None if no variants registered.
        """
        variants = self._variants.get(agent_type, [])
        if not variants:
            return None
        if len(variants) == 1:
            return variants[0].prompt

        import random
        if random.random() < self.EXPLORE_RATE:
            # Explore: pick a random variant
            chosen = random.choice(variants)
        else:
            # Exploit: pick highest avg_score
            chosen = max(variants, key=lambda v: v.avg_score)

        return chosen.prompt

    def record_result(
        self,
        agent_type: str,
        prompt: str,
        score: float,
        success: bool,
    ) -> None:
        """Record the outcome of using a prompt.

        Args:
            agent_type: Agent role identifier.
            prompt: The prompt that was used.
            score: Critic score (0.0-1.0).
            success: Whether the task succeeded.
        """
        variants = self._variants.get(agent_type, [])
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]

        for v in variants:
            if v.name == prompt_hash or v.prompt == prompt:
                v.record(score, success)
                return

        # Auto-register unknown prompt
        v = PromptVariant(prompt)
        v.record(score, success)
        self._variants.setdefault(agent_type, []).append(v)

    def get_winner(self, agent_type: str) -> PromptVariant | None:
        """Return the statistically best variant if enough data exists.

        Args:
            agent_type: Agent role identifier.

        Returns:
            Winning PromptVariant, or None if insufficient data.
        """
        variants = self._variants.get(agent_type, [])
        mature = [v for v in variants if v.uses >= self.MIN_USES_TO_COMPARE]
        if not mature:
            return None
        return max(mature, key=lambda v: v.avg_score)

    async def generate_improved_prompt(
        self,
        agent_type: str,
        current_prompt: str,
        failure_feedback: str,
        provider: Any = None,
    ) -> str:
        """Use an LLM to generate an improved version of a prompt.

        Args:
            agent_type: Agent role.
            current_prompt: The prompt that underperformed.
            failure_feedback: Critic feedback explaining what went wrong.
            provider: Optional LLM provider to use.

        Returns:
            Improved prompt string.
        """
        meta_prompt = (
            f"You are improving an AI agent's system prompt.\n\n"
            f"Agent type: {agent_type}\n"
            f"Current prompt:\n{current_prompt[:1000]}\n\n"
            f"Problems observed:\n{failure_feedback}\n\n"
            f"Write an improved system prompt that fixes these problems. "
            f"Preserve the agent's core purpose. Be specific about what to do differently. "
            f"Output only the new prompt, no explanation."
        )

        if provider:
            try:
                from synthron.providers.base_provider import GenerationRequest, Message
                request = GenerationRequest(
                    messages=[Message(role="user", content=meta_prompt)],
                    max_tokens=1024,
                    temperature=0.4,
                )
                response = await provider.generate(request)
                improved = response.content.strip()
                if improved and len(improved) > 50:
                    self._improvements.append({
                        "agent_type": agent_type,
                        "old_prompt_hash": hashlib.md5(current_prompt.encode()).hexdigest()[:8],
                        "feedback": failure_feedback[:200],
                        "ts": time.time(),
                    })
                    logger.info(f"[prompt_optimizer] Generated improved prompt for {agent_type}")
                    return improved
            except Exception as exc:
                logger.debug(f"[prompt_optimizer] Prompt improvement failed: {exc}")

        return current_prompt

    def report(self) -> dict[str, Any]:
        """Return optimization report for all agent types."""
        report = {}
        for agent_type, variants in self._variants.items():
            winner = self.get_winner(agent_type)
            report[agent_type] = {
                "variants": [v.to_dict() for v in variants],
                "winner": winner.to_dict() if winner else None,
                "improvements_made": sum(
                    1 for i in self._improvements if i["agent_type"] == agent_type
                ),
            }
        return report
