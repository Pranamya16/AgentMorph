"""Tests for the unified Tool + ToolRegistry."""

from __future__ import annotations

import pytest

from agentmorph.tools.base import Tool, ToolRegistry


def _square_tool() -> Tool:
    return Tool(
        name="square",
        description="Square a number.",
        parameters={
            "type": "object",
            "properties": {"n": {"type": "number"}},
            "required": ["n"],
            "additionalProperties": False,
        },
        func=lambda n: n * n,
        read_only=True,
        category="math",
    )


def test_tool_invokes_and_validates() -> None:
    tool = _square_tool()
    assert tool.invoke({"n": 3}) == 9
    with pytest.raises(Exception):
        tool.invoke({"n": "not a number"})
    with pytest.raises(Exception):
        tool.invoke({})


def test_tool_name_must_be_identifier() -> None:
    with pytest.raises(ValueError):
        Tool(
            name="not an identifier",
            description="",
            parameters={"type": "object", "properties": {}},
            func=lambda: None,
        )


def test_tool_parameters_must_be_object_schema() -> None:
    with pytest.raises(ValueError):
        Tool(
            name="ok",
            description="",
            parameters={"type": "string"},  # wrong shape
            func=lambda: None,
        )


def test_registry_preserves_insertion_order() -> None:
    reg = ToolRegistry()
    reg.register(_square_tool())
    reg.register(
        Tool(
            name="cube",
            description="Cube a number.",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "number"}},
                "required": ["n"],
            },
            func=lambda n: n ** 3,
        )
    )
    assert reg.names() == ["square", "cube"]
    assert len(reg) == 2


def test_registry_rejects_duplicates() -> None:
    reg = ToolRegistry()
    reg.register(_square_tool())
    with pytest.raises(ValueError):
        reg.register(_square_tool())


def test_registry_call_returns_structured_result() -> None:
    reg = ToolRegistry()
    reg.register(_square_tool())

    good = reg.call("square", {"n": 4})
    assert good.ok and good.output == 16 and good.error is None

    missing = reg.call("nope", {})
    assert not missing.ok and "unknown tool" in missing.error

    bad = reg.call("square", {"n": "nope"})
    assert not bad.ok and bad.error is not None


def test_registry_openai_schema_shape() -> None:
    reg = ToolRegistry()
    reg.register(_square_tool())
    schema = reg.openai_schema()
    assert len(schema) == 1
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "square"
    assert "parameters" in schema[0]["function"]
