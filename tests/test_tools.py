"""Tests for Synthron tools."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from synthron.tools.calculator import CalculatorTool
from synthron.tools.base_tool import ToolRegistry, BaseTool


class TestCalculatorTool:
    @pytest.fixture
    def calc(self):
        return CalculatorTool()

    async def test_basic_arithmetic(self, calc):
        result = await calc.run("2 + 2")
        assert "4" in result

    async def test_power(self, calc):
        result = await calc.run("2 ** 10")
        assert "1024" in result

    async def test_sqrt(self, calc):
        result = await calc.run("sqrt(144)")
        assert "12" in result

    async def test_division_by_zero(self, calc):
        result = await calc.run("1 / 0")
        assert "zero" in result.lower() or "error" in result.lower()

    async def test_pi_constant(self, calc):
        result = await calc.run("pi")
        assert "3.14" in result

    async def test_complex_expression(self, calc):
        result = await calc.run("(100 * 1.05) ** 3")
        assert "=" in result

    async def test_log_function(self, calc):
        result = await calc.run("log10(1000)")
        assert "3" in result

    async def test_empty_input(self, calc):
        result = await calc.run("")
        assert "No valid" in result or "expression" in result.lower()


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)
        retrieved = registry.get("calculator")
        assert retrieved is calc

    def test_not_found_raises(self):
        from synthron.utils.exceptions import ToolNotFoundError
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent_tool")

    def test_list_names(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert "calculator" in registry.list_names()

    def test_schemas(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        schemas = registry.schemas()
        assert len(schemas) == 1
        assert "name" in schemas[0]


class TestFileTool:
    @pytest.fixture
    def file_tool(self, tmp_path):
        from synthron.tools.file_tool import FileTool
        return FileTool(workspace_dir=str(tmp_path))

    async def test_write_and_read(self, file_tool):
        write_result = await file_tool.run("write:test.txt:Hello, Synthron!")
        assert "Written" in write_result

        read_result = await file_tool.run("read:test.txt")
        assert "Hello, Synthron!" in read_result

    async def test_file_not_found(self, file_tool):
        result = await file_tool.run("read:nonexistent.txt")
        assert "not found" in result.lower()

    async def test_list_directory(self, file_tool):
        await file_tool.run("write:list_test.txt:content")
        result = await file_tool.run("list:")
        assert "list_test.txt" in result
