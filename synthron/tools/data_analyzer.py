"""Data analyzer tool — analyze CSV, JSON, and tabular data."""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

from synthron.tools.base_tool import BaseTool
from synthron.utils.exceptions import ToolExecutionError
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class DataAnalyzerTool(BaseTool):
    """Analyze tabular data from CSV, JSON, or raw text.

    Supports:
    - CSV parsing and statistical summary
    - JSON data analysis
    - Descriptive statistics (mean, median, std, min, max)
    - Column profiling
    - Missing value detection
    - Correlation analysis
    """

    name = "data_analyzer"
    description = (
        "Analyze CSV, JSON, or tabular data. Provide data inline or as a file path. "
        "Returns statistics, column profiles, and insights."
    )
    category = "data"
    requires_network = False

    def __init__(self, max_rows: int = 10_000) -> None:
        self.max_rows = max_rows

    async def run(self, input_text: str, context: Any = None) -> str:
        """Analyze the provided data.

        Args:
            input_text: CSV/JSON data string, or a file path.
            context: Optional dict with 'query' key for specific analysis.

        Returns:
            Analysis report as formatted string.
        """
        query = ""
        if isinstance(context, dict):
            query = context.get("query", "")

        # Check if input is a file path
        import os
        if os.path.exists(input_text.strip()):
            return await self._analyze_file(input_text.strip(), query)

        # Try to detect format and parse inline
        text = input_text.strip()
        if not text:
            return "No data provided."

        return await asyncio.to_thread(self._analyze_text, text, query)

    async def _analyze_file(self, path: str, query: str) -> str:
        """Analyze a data file."""
        import os
        ext = os.path.splitext(path)[1].lower()

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            return f"Could not read file {path}: {exc}"

        return await asyncio.to_thread(self._analyze_text, content, query)

    def _analyze_text(self, text: str, query: str = "") -> str:
        """Detect format and analyze text data."""
        # Try JSON first
        stripped = text.strip()
        if stripped.startswith(("[", "{")):
            return self._analyze_json(stripped, query)

        # Try CSV
        if "," in stripped or "\t" in stripped:
            return self._analyze_csv(stripped, query)

        return f"Data format not recognized. Provide CSV or JSON data.\nFirst 200 chars: {text[:200]}"

    def _analyze_csv(self, csv_text: str, query: str = "") -> str:
        """Analyze CSV data using pandas."""
        try:
            import pandas as pd
        except ImportError:
            return "pandas not installed. Run: pip install pandas"

        try:
            df = pd.read_csv(io.StringIO(csv_text), nrows=self.max_rows)
            return self._format_dataframe_report(df, query)
        except Exception as exc:
            # Try tab-separated
            try:
                df = pd.read_csv(io.StringIO(csv_text), sep="\t", nrows=self.max_rows)
                return self._format_dataframe_report(df, query)
            except Exception:
                return f"CSV parsing failed: {exc}"

    def _analyze_json(self, json_text: str, query: str = "") -> str:
        """Analyze JSON data."""
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return f"Invalid JSON: {exc}"

        if isinstance(data, list) and data:
            # List of records → try as DataFrame
            try:
                import pandas as pd
                df = pd.DataFrame(data[: self.max_rows])
                return self._format_dataframe_report(df, query)
            except ImportError:
                pass
            # Manual summary without pandas
            return self._summarize_list(data)
        elif isinstance(data, dict):
            return self._summarize_dict(data, query)
        else:
            return f"JSON value: {json.dumps(data)[:500]}"

    def _format_dataframe_report(self, df: Any, query: str = "") -> str:
        """Generate a comprehensive DataFrame analysis report."""
        import pandas as pd

        lines = [
            f"## Data Analysis Report",
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"Columns: {', '.join(df.columns.tolist())}",
            "",
            "### Column Types",
        ]
        for col in df.columns:
            dtype = df[col].dtype
            nulls = df[col].isna().sum()
            null_pct = (nulls / len(df) * 100) if len(df) else 0
            lines.append(f"  - {col}: {dtype} | {nulls} nulls ({null_pct:.1f}%)")

        # Numeric stats
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            lines.append("\n### Numeric Summary")
            try:
                desc = df[numeric_cols].describe()
                lines.append(desc.to_string())
            except Exception:
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if not col_data.empty:
                        lines.append(
                            f"  {col}: min={col_data.min():.2f}, "
                            f"max={col_data.max():.2f}, "
                            f"mean={col_data.mean():.2f}, "
                            f"std={col_data.std():.2f}"
                        )

        # Categorical columns
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            lines.append("\n### Categorical Columns")
            for col in cat_cols[:5]:
                vc = df[col].value_counts().head(5)
                lines.append(f"  {col} top values: {vc.to_dict()}")

        # Answer specific query
        if query:
            lines.append(f"\n### Query: {query}")
            try:
                result = df.query(query) if query.strip() else df
                lines.append(f"Query result: {len(result):,} rows")
                lines.append(result.head(10).to_string())
            except Exception:
                lines.append(f"(Could not execute query: '{query}')")

        lines.append(f"\n### Sample Data (first 5 rows)")
        lines.append(df.head(5).to_string())

        return "\n".join(lines)

    def _summarize_list(self, data: list) -> str:
        """Summarize a JSON list without pandas."""
        result = [f"JSON Array: {len(data)} items"]
        if data:
            result.append(f"First item type: {type(data[0]).__name__}")
            if isinstance(data[0], dict):
                result.append(f"Keys: {list(data[0].keys())}")
            result.append(f"Sample (first 3):\n{json.dumps(data[:3], indent=2)}")
        return "\n".join(result)

    def _summarize_dict(self, data: dict, query: str) -> str:
        """Summarize a JSON dict."""
        lines = [
            f"JSON Object: {len(data)} keys",
            f"Keys: {list(data.keys())}",
        ]
        if query and query in data:
            lines.append(f"\nValue for '{query}':\n{json.dumps(data[query], indent=2)}")
        else:
            lines.append(f"\nContent preview:\n{json.dumps(data, indent=2)[:2000]}")
        return "\n".join(lines)
