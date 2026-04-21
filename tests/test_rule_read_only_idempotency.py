"""Tests for rule 4 — read-only-idempotency."""

from __future__ import annotations

from agentmorph.environments.base import Scenario
from agentmorph.rules import (
    RULE_IDS,
    available_rules,
    make_equivalence_checker,
    make_mutator,
)
from agentmorph.rules.base import DivergenceType
from agentmorph.rules.read_only_idempotency import (
    NUDGE_SUFFIX,
    READ_ONLY_NUDGE_TOOLS,
)
from agentmorph.tools.ecommerce import build_ecommerce_registry


def _scn() -> Scenario:
    return Scenario(
        id="eco_shop_kettle",
        env_id="ecommerce",
        prompt="Find a kettle and add it to the cart.",
    )


def _traj_with_state(post_state: dict | None, *, tool_calls=(), final="ok"):
    steps = [
        {"kind": "tool_call", "tool_name": n, "tool_args": a}
        for n, a in tool_calls
    ]
    if final is not None:
        steps.append({"kind": "final_answer", "content": final})
    metadata = {}
    if post_state is not None:
        metadata["state_delta"] = {"pre": {}, "post": post_state}
    return {"steps": steps, "final_answer": final, "metadata": metadata}


# -- Registry factory --------------------------------------------------------


def test_rule_4_registered_and_importable() -> None:
    assert "read-only-idempotency" in RULE_IDS
    assert "read-only-idempotency" in available_rules()


# -- Mutator contract --------------------------------------------------------


def test_mutator_appends_read_only_nudge_to_prompt() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("read-only-idempotency").apply(_scn(), reg, seed=0)
    assert result.scenario.prompt.endswith(NUDGE_SUFFIX)
    assert "Find a kettle" in result.scenario.prompt  # original preserved


def test_mutator_records_injected_reads_in_metadata() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("read-only-idempotency").apply(_scn(), reg, seed=0)
    assert set(result.scenario.metadata["mutator_extra"]["injected_reads"]) == set(
        READ_ONLY_NUDGE_TOOLS
    )


def test_mutator_passes_registry_through_unchanged() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("read-only-idempotency").apply(_scn(), reg, seed=0)
    assert result.registry is reg  # rule 4 never touches the registry


# -- Checker decision tree ---------------------------------------------------


def test_checker_none_when_states_match() -> None:
    state = {"orders": [], "cart_items": []}
    t1 = _traj_with_state(state)
    t2 = _traj_with_state(state)
    r = make_equivalence_checker("read-only-idempotency").compare(t1, t2)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


def test_checker_side_effects_differ_when_orders_placed() -> None:
    state_orig = {"orders": [], "cart_items": []}
    state_mut = {"orders": [{"id": "O1", "status": "placed"}], "cart_items": []}
    t1 = _traj_with_state(state_orig)
    t2 = _traj_with_state(state_mut)
    r = make_equivalence_checker("read-only-idempotency").compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.SIDE_EFFECTS_DIFFER
    assert "orders" in r.details


def test_checker_side_effects_differ_when_cart_differs() -> None:
    state_orig = {"cart_items": [{"product_id": "P020", "quantity": 1}], "orders": []}
    state_mut = {"cart_items": [{"product_id": "P020", "quantity": 2}], "orders": []}
    t1 = _traj_with_state(state_orig)
    t2 = _traj_with_state(state_mut)
    r = make_equivalence_checker("read-only-idempotency").compare(t1, t2)
    assert r.is_equivalent is False
    assert "cart_items" in r.details


def test_checker_returns_none_with_warning_when_state_not_captured() -> None:
    """If the runner didn't pass --capture-state, checker can't judge. It
    must return equivalent-with-note, not a false-positive bug."""
    t1 = _traj_with_state(None)  # no metadata.state_delta
    t2 = _traj_with_state(None)
    r = make_equivalence_checker("read-only-idempotency").compare(t1, t2)
    assert r.is_equivalent is True
    assert "state not captured" in r.details.lower()


def test_checker_summarises_multiple_diverging_fields() -> None:
    state_orig = {"cart_items": [], "orders": [], "reviews": [], "tickets": {}}
    state_mut = {
        "cart_items": [{"product_id": "P1", "quantity": 1}],
        "orders": [{"id": "O1"}],
        "reviews": [{"product_id": "P1", "rating": 5}],
        "tickets": {"T1": {"status": "open"}},
    }
    t1 = _traj_with_state(state_orig)
    t2 = _traj_with_state(state_mut)
    r = make_equivalence_checker("read-only-idempotency").compare(t1, t2)
    assert r.is_equivalent is False
    # Details lists at most 3 fields + a "(+N more)" suffix when more diverge.
    assert "(+1 more)" in r.details
