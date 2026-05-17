"""Calculator tool — safe math expressions and data computations."""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)

# Safe builtins and math functions
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "inf": math.inf,
    "nan": math.nan,
}

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool(BaseTool):
    """Safe mathematical expression evaluator.

    Supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Math functions: sqrt, log, sin, cos, tan, exp, abs, round, etc.
    - Constants: pi, e
    - Multi-step calculations
    - Unit conversions (basic)
    """

    name = "calculator"
    description = (
        "Evaluate math expressions safely. Examples: '2 ** 10', 'sqrt(144)', "
        "'(100 * 1.08) ** 5', 'log(1000) / log(10)'"
    )
    category = "computation"
    requires_network = False

    async def run(self, input_text: str, context: Any = None) -> str:
        """Evaluate a mathematical expression.

        Args:
            input_text: Math expression or natural language math query.
            context: Unused.

        Returns:
            Result as formatted string.
        """
        expression = self._extract_expression(input_text.strip())
        if not expression:
            return "No valid math expression found."

        logger.debug(f"[calculator] Evaluating: {expression}")

        try:
            result = self._safe_eval(expression)

            if isinstance(result, float):
                if result == int(result) and abs(result) < 1e15:
                    formatted = str(int(result))
                else:
                    formatted = f"{result:.6g}"
            else:
                formatted = str(result)

            return f"{expression} = {formatted}"

        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as exc:
            return f"Math error: {exc}"
        except Exception as exc:
            return f"Calculation error: {exc}"

    def _safe_eval(self, expression: str) -> float | int:
        """Evaluate a math expression using AST parsing (no eval()).

        Args:
            expression: Math expression string.

        Returns:
            Numeric result.

        Raises:
            ValueError: For invalid or unsafe expressions.
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc}") from exc

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.expr) -> float | int:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Non-numeric constant: {node.value}")

        elif isinstance(node, ast.BinOp):
            op_func = SAFE_OPERATORS.get(type(node.op))
            if not op_func:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return op_func(left, right)

        elif isinstance(node, ast.UnaryOp):
            op_func = SAFE_OPERATORS.get(type(node.op))
            if not op_func:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op_func(self._eval_node(node.operand))

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls allowed")
            func_name = node.func.id
            func = SAFE_FUNCTIONS.get(func_name)
            if not func:
                raise ValueError(f"Unknown function: {func_name}")
            args = [self._eval_node(a) for a in node.args]
            return func(*args)

        elif isinstance(node, ast.Name):
            val = SAFE_FUNCTIONS.get(node.id)
            if isinstance(val, (int, float)):
                return val
            raise ValueError(f"Unknown name: {node.id}")

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def _extract_expression(self, text: str) -> str:
        """Extract a math expression from natural language input.

        Args:
            text: Input text (may be a pure expression or natural language).

        Returns:
            Clean math expression string.
        """
        # Already a pure expression
        if re.match(r"^[\d\s\+\-\*\/\(\)\.\^%,]+$", text):
            return text.replace("^", "**").replace(",", "")

        # Extract expression after common prefixes
        patterns = [
            r"calculate[:\s]+(.+)",
            r"compute[:\s]+(.+)",
            r"eval[:\s]+(.+)",
            r"what is[:\s]+(.+)\??",
            r"=?\s*([\d\s\(\)\.\+\-\*\/\^%]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                expr = match.group(1).strip().rstrip("?")
                return expr.replace("^", "**").replace(",", "")

        # Return as-is and let safe_eval handle it
        return text.replace("^", "**").replace(",", "")
