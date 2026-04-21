"""Tests for rule 7 — parameter-order-invariance."""

from __future__ import annotations

from agentmorph.environments.base import Scenario
from agentmorph.rules import (
    RULE_IDS,
    available_rules,
    make_equivalence_checker,
    make_mutator,
)
from agentmorph.rules.base import DivergenceType
from agentmorph.tools.ecommerce import build_ecommerce_registry


def _scn() -> Scenario:
    return Scenario(id="eco_shop_kettle", env_id="ecommerce", prompt="find kettle")


def _traj(tool_calls, final="done"):
    steps = [{"kind": "tool_call", "tool_name": n, "tool_args": a} for n, a in tool_calls]
    if final is not None:
        steps.append({"kind": "final_answer", "content": final})
    return {"steps": steps, "final_answer": final}


# -- Registry factory --------------------------------------------------------


def test_rule_7_is_registered_and_importable() -> None:
    assert "parameter-order-invariance" in RULE_IDS
    assert "parameter-order-invariance" in available_rules()


# -- Mutator contract --------------------------------------------------------


def test_mutator_reorders_properties_for_multi_param_tools() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("parameter-order-invariance").apply(_scn(), reg, seed=0)
    # search_products has 5 params — should be shuffled.
    search_original = reg.get("search_products")
    search_mutated = result.registry.get("search_products")
    original_keys = list(search_original.parameters["properties"].keys())
    mutated_keys = list(search_mutated.parameters["properties"].keys())
    assert sorted(original_keys) == sorted(mutated_keys)  # same set
    # Overwhelmingly likely the order changed (probability of identity ≈ 1/5! for a 5-key shuffle).
    # The test occasionally succeeds with no change if the RNG happens to pick the identity
    # permutation. We verify via metadata's `tools_shuffled` list instead:
    assert "search_products" in result.metadata["tools_shuffled"]


def test_mutator_is_noop_on_zero_or_one_param_tools() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("parameter-order-invariance").apply(_scn(), reg, seed=0)
    # list_categories has no params — should not appear in tools_shuffled.
    list_categories_orig = reg.get("list_categories").parameters
    list_categories_new = result.registry.get("list_categories").parameters
    assert list_categories_orig.get("properties", {}) == list_categories_new.get("properties", {})
    assert "list_categories" not in result.metadata["tools_shuffled"]


def test_mutator_is_deterministic_under_seed() -> None:
    _state, reg = build_ecommerce_registry()
    r1 = make_mutator("parameter-order-invariance").apply(_scn(), reg, seed=42)
    r2 = make_mutator("parameter-order-invariance").apply(_scn(), reg, seed=42)
    assert r1.metadata["mutated_property_orders"] == r2.metadata["mutated_property_orders"]


def test_mutator_preserves_tool_func_and_parameters_otherwise() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("parameter-order-invariance").apply(_scn(), reg, seed=0)
    orig = reg.get("search_products")
    new = result.registry.get("search_products")
    assert orig.func is new.func  # implementation unchanged
    assert orig.name == new.name
    assert orig.description == new.description
    assert orig.parameters["required"] == new.parameters["required"]
    assert set(orig.parameters["properties"].keys()) == set(new.parameters["properties"].keys())


def test_mutator_emits_expected_metadata_keys() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("parameter-order-invariance").apply(_scn(), reg, seed=0)
    assert "original_property_orders" in result.metadata
    assert "mutated_property_orders" in result.metadata
    assert "tools_shuffled" in result.metadata


# -- Checker decision tree ---------------------------------------------------


def test_checker_none_when_args_match_despite_key_order() -> None:
    # Python dicts are equal regardless of insertion order; frozenset in
    # tool_calls_of() makes this explicit. Both trajectories call the same
    # tool with the same values.
    t1 = _traj([("search_products", {"query": "k", "max_price": 50})])
    t2 = _traj([("search_products", {"max_price": 50, "query": "k"})])
    r = make_equivalence_checker("parameter-order-invariance").compare(t1, t2)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


def test_checker_tool_set_differs_when_args_values_differ() -> None:
    t1 = _traj([("search_products", {"query": "kettle", "max_price": 50})])
    t2 = _traj([("search_products", {"query": "headphones", "max_price": 100})])
    r = make_equivalence_checker("parameter-order-invariance").compare(t1, t2)
    assert r.is_equivalent is False
    # Same tool name but different (name, frozen-args) tuple → TOOL_SET_DIFFERS.
    assert r.divergence_type is DivergenceType.TOOL_SET_DIFFERS
