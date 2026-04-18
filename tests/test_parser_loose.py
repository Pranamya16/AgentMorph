"""Tests that pin the loosened native parser.

Small open models (3B-9B) diverge from the strict ``` ```json ``` + `FINAL:`
format in predictable ways. These tests lock in the looseness so a future
regex tightening can't silently regress us back to a 0% finish rate.
"""

from __future__ import annotations

import pytest

from agentmorph.agents.base import _parse_step


# -- Tool-call shapes that must parse as ('tool', {...}) --------------------


_TOOL_CALL_VARIANTS = [
    # The canonical format.
    '```json\n{"tool": "search_products", "arguments": {"query": "kettle"}}\n```',
    # ```python fence — Llama-3.2-3B emits this often.
    '```python\n{"tool": "search_products", "arguments": {"query": "kettle"}}\n```',
    # ```tool_code — some Gemma variants.
    '```tool_code\n{"tool": "search_products", "arguments": {"query": "kettle"}}\n```',
    # Bare ``` fence with no language tag.
    '```\n{"tool": "search_products", "arguments": {"query": "kettle"}}\n```',
    # Alternate key names: name + args (LangChain-style).
    '```json\n{"name": "search_products", "args": {"query": "kettle"}}\n```',
    # Unfenced bare JSON (last resort).
    'I think I should call: {"tool": "search_products", "arguments": {"query": "kettle"}}',
    # Preamble before the fence.
    'Let me search for kettles.\n\n```json\n{"tool": "search_products", "arguments": {"query": "kettle"}}\n```\n\nThat should do it.',
]


@pytest.mark.parametrize("raw", _TOOL_CALL_VARIANTS)
def test_tool_call_variants_parse(raw: str) -> None:
    kind, payload = _parse_step(raw)
    assert kind == "tool", f"parsed as {kind!r}: {payload!r}"
    assert isinstance(payload, dict)
    assert payload["tool"] == "search_products"
    assert payload["arguments"] == {"query": "kettle"}


# -- Final-answer shapes that must parse as ('final', ...) ------------------


_FINAL_VARIANTS = [
    ("FINAL: The kettle costs $39.", "The kettle costs $39."),
    ("FINAL ANSWER: Done.", "Done."),
    ("final answer: lowercase works too", "lowercase works too"),
    ("The final answer is: 42", "42"),
    ("My final answer is: 42", "42"),
    ("Some preamble.\nFINAL: clean answer", "clean answer"),
]


@pytest.mark.parametrize("raw,expected", _FINAL_VARIANTS)
def test_final_variants_parse(raw: str, expected: str) -> None:
    kind, payload = _parse_step(raw)
    assert kind == "final", f"parsed as {kind!r}: {payload!r}"
    assert payload.strip() == expected


# -- Noop (unparseable) --------------------------------------------------


def test_unparseable_prose_returns_noop() -> None:
    kind, _ = _parse_step("I'm not sure what to do here.")
    assert kind == "noop"


def test_json_without_name_returns_noop() -> None:
    kind, _ = _parse_step('```json\n{"foo": "bar"}\n```')
    assert kind == "noop"


# -- Ordering: tool call before FINAL wins tool ------------------------------


def test_tool_call_before_final_wins() -> None:
    raw = (
        '```json\n{"tool": "search_products", "arguments": {"query": "k"}}\n```\n'
        'FINAL: I would have answered this.'
    )
    kind, payload = _parse_step(raw)
    assert kind == "tool"
    assert payload["tool"] == "search_products"


def test_final_before_tool_call_wins_final() -> None:
    raw = (
        'FINAL: Here is the answer.\n'
        '```json\n{"tool": "search_products", "arguments": {"query": "k"}}\n```'
    )
    kind, payload = _parse_step(raw)
    assert kind == "final"
    assert payload.startswith("Here is the answer")
