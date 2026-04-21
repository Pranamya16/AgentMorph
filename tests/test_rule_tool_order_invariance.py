"""Tests for rule 1 — tool-order-invariance."""

from __future__ import annotations

from agentmorph.environments.base import Scenario
from agentmorph.rules import RULE_IDS, make_equivalence_checker, make_mutator
from agentmorph.rules.base import DivergenceType, MutationResult
from agentmorph.rules.registry import available_rules
from agentmorph.tools.ecommerce import build_ecommerce_registry


def _scn() -> Scenario:
    return Scenario(id="eco_shop_kettle", env_id="ecommerce", prompt="find kettle")


def _traj(tool_calls, final="done"):
    steps = [
        {"kind": "tool_call", "tool_name": n, "tool_args": a}
        for n, a in tool_calls
    ]
    if final is not None:
        steps.append({"kind": "final_answer", "content": final})
    return {"steps": steps, "final_answer": final}


# -- Registry factory --------------------------------------------------------


def test_rule_1_is_registered_and_importable() -> None:
    assert "tool-order-invariance" in RULE_IDS
    mut = make_mutator("tool-order-invariance")
    chk = make_equivalence_checker("tool-order-invariance")
    assert mut.rule_id == "tool-order-invariance"
    assert chk.rule_id == "tool-order-invariance"


def test_rule_1_shows_up_in_available_rules() -> None:
    assert "tool-order-invariance" in available_rules()


# -- Mutator contract --------------------------------------------------------


def test_mutator_permutes_tool_order_deterministically() -> None:
    _state, reg = build_ecommerce_registry()
    mut = make_mutator("tool-order-invariance")
    r1 = mut.apply(_scn(), reg, seed=42)
    r2 = mut.apply(_scn(), reg, seed=42)
    assert r1.metadata["permutation"] == r2.metadata["permutation"]


def test_mutator_permutation_differs_from_identity_on_usual_seed() -> None:
    _state, reg = build_ecommerce_registry()
    mut = make_mutator("tool-order-invariance")
    result = mut.apply(_scn(), reg, seed=0)
    # With 30 tools, the probability of seed=0 producing identity is ≈ 1/30! → 0.
    assert result.metadata["permutation"] != result.metadata["original_order"]


def test_mutator_produces_a_true_permutation() -> None:
    _state, reg = build_ecommerce_registry()
    mut = make_mutator("tool-order-invariance")
    result: MutationResult = mut.apply(_scn(), reg, seed=0)
    assert sorted(result.metadata["permutation"]) == sorted(result.metadata["original_order"])
    # And the new registry matches the recorded permutation.
    assert result.registry.names() == result.metadata["permutation"]


def test_mutator_preserves_scenario_identity() -> None:
    _state, reg = build_ecommerce_registry()
    mut = make_mutator("tool-order-invariance")
    scn = _scn()
    result = mut.apply(scn, reg, seed=7)
    assert result.scenario is scn  # scenario unchanged for this rule


def test_mutator_original_registry_is_not_mutated() -> None:
    _state, reg = build_ecommerce_registry()
    pre = reg.names()
    make_mutator("tool-order-invariance").apply(_scn(), reg, seed=0)
    assert reg.names() == pre


# -- Checker decision tree ---------------------------------------------------


def test_checker_none_when_trajectories_identical() -> None:
    t = _traj([("search_products", {"q": "k"}), ("add_to_cart", {"product_id": "P020"})])
    r = make_equivalence_checker("tool-order-invariance").compare(t, t)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


def test_checker_reorder_only_is_equivalent() -> None:
    t1 = _traj([("search_products", {"q": "k"}), ("add_to_cart", {"product_id": "P020"})])
    t2 = _traj([("add_to_cart", {"product_id": "P020"}), ("search_products", {"q": "k"})])
    r = make_equivalence_checker("tool-order-invariance").compare(t1, t2)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.REORDER_ONLY


def test_checker_tool_set_differs_is_a_bug() -> None:
    t1 = _traj([("search_products", {"q": "k"})])
    t2 = _traj([("list_categories", {})])
    r = make_equivalence_checker("tool-order-invariance").compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.TOOL_SET_DIFFERS


def test_checker_completion_differs_is_a_bug() -> None:
    t1 = _traj([("search_products", {"q": "k"})], final="ok")
    t2 = _traj([("search_products", {"q": "k"})], final=None)
    r = make_equivalence_checker("tool-order-invariance").compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.COMPLETION_DIFFERS
