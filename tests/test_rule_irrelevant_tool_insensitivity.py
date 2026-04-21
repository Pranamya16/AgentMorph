"""Tests for rule 8 — irrelevant-tool-insensitivity."""

from __future__ import annotations

from agentmorph.environments.base import Scenario
from agentmorph.rules import (
    RULE_IDS,
    available_rules,
    make_equivalence_checker,
    make_mutator,
)
from agentmorph.rules.base import DivergenceType
from agentmorph.rules.irrelevant_tool_insensitivity import DUMMY_TOOL_NAME
from agentmorph.tools.ecommerce import (
    ECOMMERCE_TOOL_NAMES,
    build_ecommerce_registry,
)


def _scn() -> Scenario:
    return Scenario(id="eco_shop_kettle", env_id="ecommerce", prompt="find kettle")


def _traj(tool_calls, final="done"):
    steps = [{"kind": "tool_call", "tool_name": n, "tool_args": a} for n, a in tool_calls]
    if final is not None:
        steps.append({"kind": "final_answer", "content": final})
    return {"steps": steps, "final_answer": final}


# -- Registry factory --------------------------------------------------------


def test_rule_8_is_registered_and_importable() -> None:
    assert "irrelevant-tool-insensitivity" in RULE_IDS
    assert "irrelevant-tool-insensitivity" in available_rules()


# -- Mutator contract --------------------------------------------------------


def test_mutator_appends_dummy_tool_to_registry() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("irrelevant-tool-insensitivity").apply(_scn(), reg, seed=0)
    names = result.registry.names()
    assert DUMMY_TOOL_NAME in names
    assert len(names) == len(ECOMMERCE_TOOL_NAMES) + 1
    # Original 30 tools still present in the mutated registry.
    for original_name in ECOMMERCE_TOOL_NAMES:
        assert original_name in names


def test_mutator_does_not_modify_original_registry() -> None:
    _state, reg = build_ecommerce_registry()
    pre = reg.names()
    make_mutator("irrelevant-tool-insensitivity").apply(_scn(), reg, seed=0)
    assert reg.names() == pre  # original untouched


def test_mutator_metadata_records_dummy_tool() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("irrelevant-tool-insensitivity").apply(_scn(), reg, seed=0)
    assert result.metadata["dummy_tool"] == DUMMY_TOOL_NAME


def test_dummy_tool_is_callable_and_returns_expected_shape() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("irrelevant-tool-insensitivity").apply(_scn(), reg, seed=0)
    got = result.registry.call(DUMMY_TOOL_NAME, {"city": "Seattle"})
    assert got.ok
    assert got.output == {"temperature_f": 72, "condition": "sunny", "city": "Seattle"}


# -- Checker decision tree ---------------------------------------------------


def test_checker_tool_set_differs_when_agent_called_weather() -> None:
    t1 = _traj([("search_products", {"q": "k"})])
    t2 = _traj([("search_products", {"q": "k"}), (DUMMY_TOOL_NAME, {"city": "Seattle"})])
    r = make_equivalence_checker("irrelevant-tool-insensitivity").compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.TOOL_SET_DIFFERS
    assert DUMMY_TOOL_NAME in r.details


def test_checker_none_when_agent_did_not_call_weather() -> None:
    t = _traj([("search_products", {"q": "k"}), ("add_to_cart", {"product_id": "P1"})])
    r = make_equivalence_checker("irrelevant-tool-insensitivity").compare(t, t)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


def test_checker_classifies_other_divergences_normally_when_weather_absent() -> None:
    t1 = _traj([("search_products", {"q": "k"})])
    t2 = _traj([("list_categories", {})])
    r = make_equivalence_checker("irrelevant-tool-insensitivity").compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.TOOL_SET_DIFFERS
