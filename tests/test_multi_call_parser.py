"""Tests for the multi-tool-call parser and the system-role fold.

These pin the two fixes that unblocked Gemma-2-9B and improved native
finish rates on models (Qwen, Llama-3.2-3B, Phi-4) that emit multiple
tool calls per turn or `null`-valued args.
"""

from __future__ import annotations

import pytest

from agentmorph.agents.base import _parse_multi_step, _strip_nullish_args
from agentmorph.models import _fold_system_into_user


# -- _parse_multi_step --------------------------------------------------------


def test_single_fenced_tool_call_still_parses() -> None:
    text = '```json\n{"tool": "search_products", "arguments": {"query": "k"}}\n```'
    calls, final = _parse_multi_step(text)
    assert final is None
    assert len(calls) == 1
    assert calls[0]["tool"] == "search_products"
    assert calls[0]["arguments"] == {"query": "k"}


def test_qwen_multi_fence_pattern() -> None:
    """Two separate fenced blocks in one turn."""
    text = (
        '```json\n{"tool": "add_to_cart", "arguments": {"product_id": "P1"}}\n```\n'
        '```json\n{"tool": "view_cart", "arguments": {}}\n```'
    )
    calls, final = _parse_multi_step(text)
    assert final is None
    assert [c["tool"] for c in calls] == ["add_to_cart", "view_cart"]


def test_llama_multi_in_one_fence_pattern() -> None:
    """Three JSON objects concatenated inside one fenced block (the Llama-3.2-3B
    failure mode we saw on Colab)."""
    text = (
        '```json\n'
        '{"tool": "search_products", "arguments": {"query": "kettle"}}\n'
        '{"tool": "add_to_cart", "arguments": {"product_id": "P1", "quantity": 1}}\n'
        '{"tool": "view_cart", "arguments": {}}\n'
        '```'
    )
    calls, final = _parse_multi_step(text)
    assert final is None
    assert [c["tool"] for c in calls] == ["search_products", "add_to_cart", "view_cart"]


def test_multi_call_wins_over_final() -> None:
    """If both tool calls and FINAL appear, tools execute first (FINAL was premature)."""
    text = (
        '```json\n{"tool": "view_cart", "arguments": {}}\n```\n'
        'FINAL: Your cart is empty.'
    )
    calls, final = _parse_multi_step(text)
    assert final is None
    assert len(calls) == 1


def test_final_only_with_no_tool_calls() -> None:
    text = 'FINAL: The kettle costs $39.'
    calls, final = _parse_multi_step(text)
    assert calls == []
    assert final == "The kettle costs $39."


def test_null_args_are_stripped() -> None:
    """Phi-4 pattern: `null`-valued args become absent (not None)."""
    text = (
        '```json\n'
        '{"tool": "search_products", "arguments": '
        '{"query": "kettle", "min_price": null, "category": "null"}}\n'
        '```'
    )
    calls, _ = _parse_multi_step(text)
    assert len(calls) == 1
    # Null + "null" string both dropped.
    assert calls[0]["arguments"] == {"query": "kettle"}


def test_bare_json_fallback_still_works() -> None:
    text = 'Let me search: {"tool": "search_products", "arguments": {"query": "k"}}'
    calls, _ = _parse_multi_step(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "search_products"


def test_unparseable_returns_empty() -> None:
    calls, final = _parse_multi_step("I am not sure what to do.")
    assert calls == []
    assert final is None


def test_json_array_of_tool_calls() -> None:
    """Some models emit a JSON array rather than separate objects."""
    text = (
        '```json\n'
        '[{"tool": "search_products", "arguments": {"query": "a"}},\n'
        ' {"tool": "view_cart", "arguments": {}}]\n'
        '```'
    )
    calls, _ = _parse_multi_step(text)
    assert [c["tool"] for c in calls] == ["search_products", "view_cart"]


# -- _strip_nullish_args -----------------------------------------------------


@pytest.mark.parametrize("value", [None, "null", "None", "NIL", "n/a", "", " "])
def test_strip_nullish_removes_null_variants(value: object) -> None:
    out = _strip_nullish_args({"x": value, "y": 42})
    assert out == {"y": 42}


def test_strip_nullish_keeps_zero_and_false() -> None:
    """Don't confuse `0` / `False` with null."""
    out = _strip_nullish_args({"n": 0, "flag": False, "s": ""})
    assert out == {"n": 0, "flag": False}


# -- _fold_system_into_user --------------------------------------------------


def test_fold_system_into_following_user() -> None:
    """The Gemma fix: system messages merge into the next user message."""
    msgs = [
        {"role": "system", "content": "You are careful."},
        {"role": "user", "content": "Find me a kettle."},
    ]
    out = _fold_system_into_user(msgs)
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "You are careful." in out[0]["content"]
    assert "Find me a kettle." in out[0]["content"]


def test_fold_preserves_assistant_and_tool_turns() -> None:
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    out = _fold_system_into_user(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "user"]
    assert "S" in out[0]["content"] and "U1" in out[0]["content"]


def test_fold_handles_multiple_system_messages() -> None:
    msgs = [
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "hello"},
    ]
    out = _fold_system_into_user(msgs)
    assert len(out) == 1
    # Both system messages appear in the merged user turn.
    assert "A" in out[0]["content"]
    assert "B" in out[0]["content"]
    assert "hello" in out[0]["content"]


def test_fold_handles_dangling_system_with_no_user() -> None:
    """System message with nothing following — promoted to a standalone user turn."""
    msgs = [{"role": "system", "content": "S"}]
    out = _fold_system_into_user(msgs)
    assert out == [{"role": "user", "content": "S"}]


def test_fold_noop_when_no_system_present() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = _fold_system_into_user(msgs)
    assert out == msgs
