"""Critic Agent — scores subtask results and triggers smart retries."""

from __future__ import annotations

import json
import re
from typing import Any

from synthron.agents.base_agent import (
    AgentResult,
    BaseAgent,
    CriticScore,
    SubTask,
    SubTaskResult,
)
from synthron.utils.config import settings
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

_CRITIC_SYSTEM = """You are SYNTHRON's CriticAgent — a ruthless quality evaluator for AI agent outputs.

Your job: evaluate whether an agent's output fully and accurately completes the given subtask.

SCORING RUBRIC:
- 0.9-1.0: Perfect. Complete, accurate, detailed, directly addresses the task.
- 0.8-0.9: Good. Minor gaps but fully usable.
- 0.5-0.8: Acceptable. Major gaps or vague but partially useful.
- 0.0-0.5: Poor. Doesn't address the task, wrong, or empty.

OUTPUT FORMAT (strict JSON):
{
  "score": 0.85,
  "verdict": "PASS",
  "feedback": "One sentence explaining the score",
  "improvement_hint": "Specific instruction to fix if retrying (empty string if PASS)"
}

VERDICT RULES:
- score >= 0.8: verdict = "PASS"
- score 0.5-0.79: verdict = "WARN"
- score < 0.5: verdict = "FAIL"

Be strict but fair. Do not give high scores for vague or incomplete outputs."""


class CriticAgent(BaseAgent):
    """Evaluates executor output quality and decides whether to retry.

    Powered by DeepSeek (best reasoning model) for accurate evaluation.
    Returns a CriticScore with verdict PASS / WARN / FAIL.
    """

    name = "critic"
    role = "critic"
    agent_type = "critic"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", _CRITIC_SYSTEM)
        super().__init__(**kwargs)

    def _default_system_prompt(self) -> str:
        return _CRITIC_SYSTEM

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Run critic in standalone mode (scores task as both subtask and result).

        Args:
            task: Task to evaluate (used as both subtask description and result).
            context: Optional context dict.

        Returns:
            AgentResult with CriticScore in metadata.
        """
        self._run_count += 1
        # When run standalone, create a dummy subtask/result
        subtask = SubTask(title="Standalone evaluation", description=task)
        result = SubTaskResult(
            subtask_id=subtask.id,
            subtask_title=subtask.title,
            output=context.get("output", task) if context else task,
        )
        score = await self.score(subtask, result)

        return AgentResult(
            agent_name=self.name,
            agent_type=self.agent_type,
            task=task,
            output=f"Score: {score.score:.2f} ({score.verdict}) — {score.feedback}",
            success=True,
            total_tokens=self._total_tokens,
            total_latency_ms=self._total_latency_ms,
            metadata={"critic_score": score.model_dump()},
        )

    async def score(
        self, subtask: SubTask, result: SubTaskResult
    ) -> CriticScore:
        """Score the quality of an executor result against the subtask.

        Args:
            subtask: The original subtask specification.
            result: The executor's output.

        Returns:
            CriticScore with numeric score, verdict, and feedback.
        """
        self._run_count += 1

        if not result.success or not result.output:
            score = CriticScore.from_score(
                subtask_id=subtask.id,
                score=0.0,
                feedback="Executor returned empty or failed output.",
            )
            score.improvement_hint = "Retry with a different tool or approach."
            self._log.score(score.score)
            return score

        prompt = (
            f"SUBTASK:\nTitle: {subtask.title}\nDescription: {subtask.description}\n\n"
            f"EXECUTOR OUTPUT:\n{result.output[:3000]}\n\n"
            f"Evaluate the output quality. Output JSON only."
        )

        try:
            response = await self.generate(prompt, max_tokens=512, temperature=0.2)
            score = self._parse_score(response.content, subtask.id)
        except Exception as exc:
            logger.warning(f"[critic] Scoring failed, defaulting to WARN: {exc}")
            score = CriticScore.from_score(
                subtask_id=subtask.id,
                score=0.6,
                feedback=f"Critic could not evaluate (error: {exc}). Treating as WARN.",
            )

        self._log.score(score.score, settings.agents.critic_pass_threshold)
        await self._emit_event(
            "score",
            f"{subtask.title}: {score.score:.2f} {score.verdict} — {score.feedback}",
        )
        return score

    async def suggest_improvement(self, critic_score: CriticScore) -> str:
        """Generate a specific improvement prompt to guide a retry.

        Args:
            critic_score: The failing or warning score.

        Returns:
            Detailed improvement instruction string.
        """
        if not critic_score.improvement_hint:
            return "Improve the completeness and accuracy of the output."

        prompt = (
            f"An AI agent received this feedback:\n{critic_score.feedback}\n"
            f"Hint: {critic_score.improvement_hint}\n\n"
            f"Write a specific instruction (1-2 sentences) telling the agent "
            f"exactly how to improve on the retry."
        )
        try:
            response = await self.generate(prompt, max_tokens=200, temperature=0.4)
            return response.content.strip()
        except Exception:
            return critic_score.improvement_hint

    def should_retry(self, score: CriticScore) -> bool:
        """Return True if the result should be retried.

        Args:
            score: The CriticScore to evaluate.

        Returns:
            True if score falls below the PASS threshold and retries remain.
        """
        return score.score < settings.agents.critic_pass_threshold and score.should_retry

    def _parse_score(self, raw: str, subtask_id: str) -> CriticScore:
        """Parse the LLM's JSON score response.

        Args:
            raw: Raw LLM response string.
            subtask_id: ID of the subtask being scored.

        Returns:
            Parsed CriticScore.
        """
        data = None
        # Strategy 1: direct parse
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        # Strategy 2: find { ... } span
        if data is None:
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e > s:
                try:
                    data = json.loads(raw[s:e + 1])
                except json.JSONDecodeError:
                    pass
        # Strategy 3: strip fence lines then retry
        if data is None:
            cleaned = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```"))
            s, e = cleaned.find("{"), cleaned.rfind("}")
            if s != -1 and e > s:
                try:
                    data = json.loads(cleaned[s:e + 1])
                except json.JSONDecodeError:
                    pass

        if data is None:
            nums = re.findall(r"0\.\d+|1\.0", raw)
            score_val = float(nums[0]) if nums else 0.6
            return CriticScore.from_score(subtask_id, score_val, raw[:200])

        try:
            raw_score = float(data.get("score", 0.6))
            score_val = max(0.0, min(1.0, raw_score))
            feedback = str(data.get("feedback", ""))
            improvement = str(data.get("improvement_hint", ""))

            cs = CriticScore.from_score(subtask_id, score_val, feedback)
            cs.improvement_hint = improvement
            return cs

        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug(f"[critic] JSON parse failed: {exc}, raw={raw[:100]}")
            return CriticScore.from_score(subtask_id, 0.6, "Could not parse critic response.")
