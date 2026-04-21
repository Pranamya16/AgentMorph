"""Tests for the shared Stage-2 rule utilities in `agentmorph.rules._shared`.

All rule modules (Apr 22-27) consume at least one helper here — bugs in
these propagate to every rule. Keep the contract pinned.
"""

from __future__ import annotations

import dataclasses

import pytest

from agentmorph.environments.base import Scenario, ScenarioBundle
from agentmorph.rules._shared import (
    ScenarioPromptMutator,
    SystemPromptMutator,
    classify_simple_divergence,
    clone_registry,
    completed,
    final_answer_semantically_equal,
    normalise_answer,
    reorder_registry,
    snapshot_state,
    state_delta,
    tool_call_sequence_equal,
    tool_call_set_equal,
    tool_calls_of,
)
from agentmorph.rules.base import DivergenceType
from agentmorph.tools.base import Tool, ToolRegistry
from agentmorph.tools.ecommerce import build_ecommerce_registry


# -- Fixtures ----------------------------------------------------------------


def _dummy_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Dummy tool {name}",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": False,
        },
        func=lambda q: {"ok": q},
    )


def _dummy_registry(names: list[str]) -> ToolRegistry:
    reg = ToolRegistry()
    for n in names:
        reg.register(_dummy_tool(n))
    return reg


def _dummy_trajectory(tool_calls: list[tuple[str, dict]], final: str | None = "done") -> dict:
    steps = []
    for name, args in tool_calls:
        steps.append({"kind": "tool_call", "tool_name": name, "tool_args": args})
    if final:
        steps.append({"kind": "final_answer", "content": final})
    return {"steps": steps, "final_answer": final}


# -- clone_registry ---------------------------------------------------------


def test_clone_registry_identity_preserves_tools() -> None:
    reg = _dummy_registry(["a", "b", "c"])
    cloned = clone_registry(reg)
    assert cloned.names() == ["a", "b", "c"]
    # New registry is a distinct object.
    assert cloned is not reg
    # Same tools by value (we don't replace them, identity kept).
    for name in ("a", "b", "c"):
        assert cloned.get(name) is reg.get(name)


def test_clone_registry_transform_renames() -> None:
    reg = _dummy_registry(["a", "b"])
    cloned = clone_registry(reg, lambda t: dataclasses.replace(t, name=f"{t.name}_x"))
    assert cloned.names() == ["a_x", "b_x"]
    # Original unchanged.
    assert reg.names() == ["a", "b"]


def test_clone_registry_transform_can_drop_tools() -> None:
    reg = _dummy_registry(["a", "b", "c"])
    cloned = clone_registry(reg, lambda t: None if t.name == "b" else t)
    assert cloned.names() == ["a", "c"]


def test_clone_registry_extra_tools_appended() -> None:
    reg = _dummy_registry(["a"])
    extra = [_dummy_tool("x"), _dummy_tool("y")]
    cloned = clone_registry(reg, extra_tools=extra)
    assert cloned.names() == ["a", "x", "y"]


def test_reorder_registry_is_deterministic_under_seed() -> None:
    reg = _dummy_registry(["a", "b", "c", "d", "e"])
    r1, p1 = reorder_registry(reg, seed=42, scenario_id="s1")
    r2, p2 = reorder_registry(reg, seed=42, scenario_id="s1")
    assert p1 == p2
    assert r1.names() == r2.names()


def test_reorder_registry_permutation_differs_by_seed() -> None:
    reg = _dummy_registry(["a", "b", "c", "d", "e", "f", "g", "h"])
    _, p1 = reorder_registry(reg, seed=1, scenario_id="s")
    _, p2 = reorder_registry(reg, seed=2, scenario_id="s")
    assert p1 != p2  # vanishingly unlikely to coincide for 8 items


def test_reorder_registry_is_a_true_permutation() -> None:
    reg = _dummy_registry(["a", "b", "c", "d"])
    new, perm = reorder_registry(reg, seed=0, scenario_id="s")
    assert sorted(perm) == ["a", "b", "c", "d"]
    assert new.names() == perm


# -- SystemPromptMutator base -----------------------------------------------


def test_system_prompt_mutator_base_passes_scenario_through() -> None:
    class _Uppercase(SystemPromptMutator):
        rule_id = "test-rule"

        def _transform_tool(self, tool: Tool) -> Tool:
            return dataclasses.replace(tool, name=tool.name.upper())

    scn = Scenario(id="s1", env_id="e", prompt="p")
    reg = _dummy_registry(["a", "b"])
    result = _Uppercase().apply(scn, reg, seed=0)

    assert result.scenario is scn
    assert result.registry.names() == ["A", "B"]
    assert result.metadata["rule_id"] == "test-rule"
    assert result.metadata["original_tool_names"] == ["a", "b"]
    assert result.metadata["mutated_tool_names"] == ["A", "B"]


def test_system_prompt_mutator_can_append_extra_tools() -> None:
    class _WithDummy(SystemPromptMutator):
        rule_id = "irrelevant-tool-insensitivity"

        def _extra_tools(self, registry, *, seed, scenario):
            return [_dummy_tool("dummy_weather")]

    scn = Scenario(id="s1", env_id="e", prompt="p")
    reg = _dummy_registry(["a", "b"])
    result = _WithDummy().apply(scn, reg, seed=0)
    assert result.registry.names() == ["a", "b", "dummy_weather"]


def test_scenario_prompt_mutator_replaces_prompt_preserving_metadata() -> None:
    class _Prefix(ScenarioPromptMutator):
        rule_id = "persona-insensitivity"

        def _mutate_prompt(self, prompt, *, seed, scenario):
            return f"[persona] {prompt}", {"persona_idx": 0}

    scn = Scenario(id="s1", env_id="e", prompt="find kettle", metadata={"cat": "shop"})
    reg = _dummy_registry(["a"])
    result = _Prefix().apply(scn, reg, seed=0)

    # Registry passes through unchanged for prompt-only mutators.
    assert result.registry is reg
    # New scenario is a copy with mutated prompt + preserved metadata.
    assert result.scenario.prompt == "[persona] find kettle"
    assert result.scenario.id == "s1"
    assert result.scenario.metadata["cat"] == "shop"
    assert result.scenario.metadata["original_prompt"] == "find kettle"
    assert result.scenario.metadata["mutator_extra"] == {"persona_idx": 0}
    # Top-level mutation metadata.
    assert result.metadata["rule_id"] == "persona-insensitivity"
    assert result.metadata["original_prompt"] == "find kettle"
    assert result.metadata["new_prompt"] == "[persona] find kettle"


# -- state snapshot ---------------------------------------------------------


def test_snapshot_state_captures_empty_cart_and_profile() -> None:
    state, reg = build_ecommerce_registry(seed=0)
    bundle = ScenarioBundle(
        scenario=Scenario(id="s", env_id="ecommerce", prompt="p"),
        registry=reg,
        state=state,
    )
    snap = snapshot_state(bundle)
    assert snap["cart_items"] == []
    assert snap["orders"] == []
    assert snap["user_profile"]["user_id"] == "U1"
    assert "A1" in snap["addresses"]
    assert "P1" in snap["payment_methods"]


def test_snapshot_state_reflects_cart_mutation() -> None:
    state, reg = build_ecommerce_registry(seed=0)
    bundle = ScenarioBundle(
        scenario=Scenario(id="s", env_id="ecommerce", prompt="p"),
        registry=reg,
        state=state,
    )
    pid = next(iter(state.products))
    reg.call("add_to_cart", {"product_id": pid, "quantity": 2})
    snap = snapshot_state(bundle)
    assert snap["cart_items"] == [{"product_id": pid, "quantity": 2}]


def test_snapshot_state_returns_empty_for_non_shop_state() -> None:
    bundle = ScenarioBundle(
        scenario=Scenario(id="s", env_id="other", prompt="p"),
        registry=ToolRegistry(),
        state=None,
    )
    assert snapshot_state(bundle) == {}


def test_state_delta_detects_order_placement() -> None:
    pre = {"orders": [], "cart_items": []}
    post = {"orders": [{"id": "O1", "status": "placed"}], "cart_items": []}
    delta = state_delta(pre, post)
    assert "orders" in delta
    assert "cart_items" not in delta
    assert delta["orders"]["before"] == []
    assert delta["orders"]["after"][0]["id"] == "O1"


def test_state_delta_empty_when_states_match() -> None:
    a = {"orders": [], "cart_items": []}
    b = {"orders": [], "cart_items": []}
    assert state_delta(a, b) == {}


# -- tool-call primitives ---------------------------------------------------


def test_tool_calls_of_extracts_name_and_frozen_args() -> None:
    traj = _dummy_trajectory([("search_products", {"query": "k", "max_price": 50})])
    calls = tool_calls_of(traj)
    assert calls == [("search_products", frozenset([("query", "k"), ("max_price", 50)]))]


def test_tool_calls_of_applies_name_mapping() -> None:
    traj = _dummy_trajectory([("find_items", {"q": "x"})])
    calls = tool_calls_of(traj, name_mapping={"find_items": "search_products"})
    assert calls[0][0] == "search_products"


def test_tool_call_set_equal_ignores_order() -> None:
    t1 = _dummy_trajectory([("a", {"q": "1"}), ("b", {"q": "2"})])
    t2 = _dummy_trajectory([("b", {"q": "2"}), ("a", {"q": "1"})])
    assert tool_call_set_equal(t1, t2)
    assert not tool_call_sequence_equal(t1, t2)


def test_tool_call_sequence_equal_requires_same_order() -> None:
    t = _dummy_trajectory([("a", {"q": "1"}), ("b", {"q": "2"})])
    assert tool_call_sequence_equal(t, t)


# -- answer equivalence -----------------------------------------------------


def test_normalise_answer_lowercases_and_collapses_whitespace() -> None:
    assert normalise_answer("  Hello   WORLD\n\n") == "hello world"
    assert normalise_answer(None) == ""


def test_final_answer_semantically_equal_handles_prefix_match() -> None:
    t1 = {"steps": [], "final_answer": "The total is $39.00"}
    t2 = {"steps": [], "final_answer": "The total is $39.00 — kettle added."}
    assert final_answer_semantically_equal(t1, t2)


def test_final_answer_semantically_unequal_on_substantive_difference() -> None:
    t1 = {"steps": [], "final_answer": "We found a kettle."}
    t2 = {"steps": [], "final_answer": "No kettles available."}
    assert not final_answer_semantically_equal(t1, t2)


def test_both_missing_final_answer_counts_as_equivalent() -> None:
    t1 = {"steps": [], "final_answer": None}
    t2 = {"steps": [], "final_answer": ""}
    assert final_answer_semantically_equal(t1, t2)


def test_completed_true_only_with_nonempty_final_answer() -> None:
    assert completed({"steps": [], "final_answer": "done"})
    assert not completed({"steps": [], "final_answer": None})
    assert not completed({"steps": [], "final_answer": "  "})


# -- classify_simple_divergence --------------------------------------------


def test_classify_returns_none_for_identical_trajectories() -> None:
    t = _dummy_trajectory([("a", {"q": "1"})], final="ok")
    r = classify_simple_divergence(t, t)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


def test_classify_returns_completion_differs_when_one_finished() -> None:
    t1 = _dummy_trajectory([("a", {"q": "1"})], final="ok")
    t2 = _dummy_trajectory([("a", {"q": "1"})], final=None)
    r = classify_simple_divergence(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.COMPLETION_DIFFERS


def test_classify_returns_tool_set_differs_when_sets_diverge() -> None:
    t1 = _dummy_trajectory([("a", {"q": "1"})], final="ok")
    t2 = _dummy_trajectory([("b", {"q": "1"})], final="ok")
    r = classify_simple_divergence(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.TOOL_SET_DIFFERS


def test_classify_reorder_only_is_equivalent_by_default() -> None:
    t1 = _dummy_trajectory([("a", {"q": "1"}), ("b", {"q": "2"})], final="ok")
    t2 = _dummy_trajectory([("b", {"q": "2"}), ("a", {"q": "1"})], final="ok")
    r = classify_simple_divergence(t1, t2)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.REORDER_ONLY


def test_classify_reorder_only_can_be_flagged_as_bug() -> None:
    t1 = _dummy_trajectory([("a", {"q": "1"}), ("b", {"q": "2"})], final="ok")
    t2 = _dummy_trajectory([("b", {"q": "2"}), ("a", {"q": "1"})], final="ok")
    r = classify_simple_divergence(t1, t2, sequence_reorder_is_equivalent=False)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.REORDER_ONLY


def test_classify_returns_answer_differs_when_only_answers_diverge() -> None:
    t1 = _dummy_trajectory([("a", {"q": "1"})], final="We found it.")
    t2 = _dummy_trajectory([("a", {"q": "1"})], final="No results available.")
    r = classify_simple_divergence(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.ANSWER_DIFFERS


def test_classify_honours_name_mapping() -> None:
    t1 = _dummy_trajectory([("search_products", {"q": "k"})], final="ok")
    t2 = _dummy_trajectory([("find_items", {"q": "k"})], final="ok")
    r = classify_simple_divergence(
        t1, t2, name_mapping={"find_items": "search_products"}
    )
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE
