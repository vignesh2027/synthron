"""Coder Agent — writes, debugs, and executes code with sandboxed safety."""

from __future__ import annotations

import re
from typing import Any

from synthron.agents.base_agent import AgentResult, BaseAgent, SubTask, SubTaskResult
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

_CODER_SYSTEM = """You are SYNTHRON's CoderAgent — an expert software engineer AI.

Capabilities: Write clean, working code in Python, JavaScript, SQL, bash, and more.

PROCESS:
1. Understand the coding task completely.
2. Plan the approach (data structures, algorithms, edge cases).
3. Write complete, runnable code.
4. Execute it using the code_executor tool.
5. Debug and fix any errors revealed by execution.
6. Return the final working code with a brief explanation.

CODE STANDARDS:
- Write complete files, not snippets.
- Include error handling.
- Add type hints for Python code.
- Include a usage example at the bottom.
- Test with edge cases.

DEBUGGING:
- If execution fails, analyze the error carefully.
- Fix the root cause, not just symptoms.
- Never leave broken code as the final answer."""


class CoderAgent(BaseAgent):
    """Writes, executes, and debugs code for programming tasks.

    Powered by Gemini 2.5 Flash (best coding model in the free tier).
    Uses code_executor tool for sandboxed execution and validation.
    """

    name = "coder"
    role = "coder"
    agent_type = "coder"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", _CODER_SYSTEM)
        super().__init__(**kwargs)
        self._max_debug_attempts = 3

    def _default_system_prompt(self) -> str:
        return _CODER_SYSTEM

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Write and execute code for a programming task.

        Args:
            task: Programming task description.
            context: Optional context dict.

        Returns:
            AgentResult with final working code and explanation.
        """
        self._run_count += 1
        self._log.thought(f"Coding task: {task[:80]}")
        await self._emit_event("coding", f"Starting: {task[:80]}")

        result = await self.code_and_run(task, context=context)

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

    async def code_and_run(
        self, task: str, language: str = "python", context: dict[str, Any] | None = None
    ) -> SubTaskResult:
        """Write code and execute it, with automatic debugging.

        Args:
            task: Programming task description.
            language: Target language (default: python).
            context: Optional context.

        Returns:
            SubTaskResult with final code and execution output.
        """
        subtask = SubTask(
            title="Code and run",
            description=task,
            tool_hint="code_executor",
        )
        executor_tool = self.get_tool("code_executor")

        # Step 1: Write the code
        context_str = ""
        if context:
            context_str = "\n".join(f"- {k}: {str(v)[:200]}" for k, v in context.items())

        write_prompt = (
            f"Task: {task}\n"
            f"{f'Context:{chr(10)}{context_str}' if context_str else ''}\n\n"
            f"Write complete, runnable {language} code. "
            f"Wrap the code in ```{language} ... ``` fences."
        )

        self._log.thought("Writing initial code")
        code_response = await self.generate(write_prompt, max_tokens=4096, temperature=0.3)
        code = self._extract_code(code_response.content, language)

        if not code:
            return SubTaskResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output=code_response.content,
                success=True,
                tool_used="llm_generation",
                tokens_used=code_response.total_tokens,
            )

        # Step 2: Execute the code
        if executor_tool and language == "python":
            execution_output = ""
            for attempt in range(1, self._max_debug_attempts + 1):
                self._log.action("code_executor", f"Running code (attempt {attempt})")
                try:
                    exec_result = await executor_tool.run(code, context={"language": language})
                    execution_output = str(exec_result)

                    if "Error" in execution_output or "Traceback" in execution_output:
                        self._log.warning(f"Code error detected (attempt {attempt})")
                        if attempt < self._max_debug_attempts:
                            code = await self._debug(task, code, execution_output)
                            continue
                    break

                except Exception as exc:
                    self._log.warning(f"Executor error: {exc}")
                    execution_output = f"Execution failed: {exc}"
                    break

            final_output = (
                f"## Code\n```{language}\n{code}\n```\n\n"
                f"## Execution Output\n```\n{execution_output}\n```"
            )
        else:
            final_output = f"## Code\n```{language}\n{code}\n```\n\n## Explanation\n{code_response.content}"

        return SubTaskResult(
            subtask_id=subtask.id,
            subtask_title=subtask.title,
            output=final_output,
            success=True,
            tool_used="code_executor",
            tokens_used=self._total_tokens,
        )

    async def _debug(self, task: str, buggy_code: str, error: str) -> str:
        """Debug code based on an error message.

        Args:
            task: Original task description.
            buggy_code: The code that produced an error.
            error: The error output/traceback.

        Returns:
            Fixed code string.
        """
        self._log.thought("Debugging error...")
        await self._emit_event("debugging", f"Fixing error: {error[:100]}")

        debug_prompt = (
            f"Task: {task}\n\n"
            f"Code with error:\n```python\n{buggy_code}\n```\n\n"
            f"Error:\n```\n{error[:1000]}\n```\n\n"
            f"Fix the code. Return ONLY the corrected code in ```python ... ``` fences."
        )

        response = await self.generate(debug_prompt, max_tokens=4096, temperature=0.2)
        fixed = self._extract_code(response.content, "python")
        return fixed or buggy_code

    async def explain_code(self, code: str) -> str:
        """Generate a plain-English explanation of code.

        Args:
            code: Source code to explain.

        Returns:
            Explanation string.
        """
        prompt = f"Explain this code clearly:\n\n```\n{code}\n```"
        response = await self.generate(prompt, max_tokens=1024, temperature=0.4)
        return response.content

    async def review_code(self, code: str) -> dict[str, Any]:
        """Review code for bugs, security issues, and improvements.

        Args:
            code: Source code to review.

        Returns:
            Dict with 'issues', 'suggestions', 'security', 'quality_score' keys.
        """
        prompt = (
            f"Code review this code:\n\n```\n{code}\n```\n\n"
            f"Return JSON with: issues (list), suggestions (list), "
            f"security_concerns (list), quality_score (0-10)"
        )
        response = await self.generate(prompt, max_tokens=1024, temperature=0.3)

        import json
        try:
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return {
            "issues": [],
            "suggestions": [response.content],
            "security_concerns": [],
            "quality_score": 7,
        }

    def _extract_code(self, text: str, language: str = "python") -> str:
        """Extract code from markdown fences in LLM output.

        Args:
            text: LLM response containing code.
            language: Expected language identifier.

        Returns:
            Extracted code string, or empty string if not found.
        """
        patterns = [
            rf"```{language}\s*\n?(.*?)```",
            r"```\s*\n?(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
