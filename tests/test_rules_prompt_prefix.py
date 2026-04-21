"""Tests for rules 9 (persona-insensitivity) and 10 (distractor-text-insensitivity).

Both rules follow the same ScenarioPromptMutator pattern — prepending text
to the user prompt via a seeded index into a fixed list. Tests share
structure; keep them in one module so the shared pattern is audit-friendly.
"""

from __future__ import annotations

import pytest

from agentmorph.environments.base import Scenario
from agentmorph.rules import (
    RULE_IDS,
    available_rules,
    make_equivalence_checker,
    make_mutator,
)
from agentmorph.rules.base import DivergenceType
from agentmorph.rules.distractor_text_insensitivity import DISTRACTORS
from agentmorph.rules.persona_insensitivity import PERSONAS
from agentmorph.tools.ecommerce import build_ecommerce_registry


def _scn() -> Scenario:
    return Scenario(id="eco_shop_kettle", env_id="ecommerce", prompt="Find a kettle.")


def _traj(tool_calls, final="done"):
    steps = [{"kind": "tool_call", "tool_name": n, "tool_args": a} for n, a in tool_calls]
    if final is not None:
        steps.append({"kind": "final_answer", "content": final})
    return {"steps": steps, "final_answer": final}


# -- Registry factory --------------------------------------------------------


@pytest.mark.parametrize("rule_id", ["persona-insensitivity", "distractor-text-insensitivity"])
def test_rule_is_registered_and_importable(rule_id: str) -> None:
    assert rule_id in RULE_IDS
    assert rule_id in available_rules()
    assert make_mutator(rule_id).rule_id == rule_id
    assert make_equivalence_checker(rule_id).rule_id == rule_id


# -- Mutator contracts -------------------------------------------------------


def test_persona_mutator_prepends_one_of_three_personas() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("persona-insensitivity").apply(_scn(), reg, seed=0)
    prefix = result.scenario.prompt.split("\n\n", 1)[0]
    assert prefix in PERSONAS


def test_distractor_mutator_prepends_one_of_three_distractors() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("distractor-text-insensitivity").apply(_scn(), reg, seed=0)
    prefix = result.scenario.prompt.split("\n\n", 1)[0]
    assert prefix in DISTRACTORS
    # And the pivot "Anyway, " appears between distractor and original prompt.
    assert "Anyway," in result.scenario.prompt


@pytest.mark.parametrize("rule_id", ["persona-insensitivity", "distractor-text-insensitivity"])
def test_prompt_mutator_preserves_original_prompt_in_metadata(rule_id: str) -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator(rule_id).apply(_scn(), reg, seed=0)
    assert result.scenario.metadata["original_prompt"] == "Find a kettle."
    assert result.metadata["original_prompt"] == "Find a kettle."
    assert result.metadata["new_prompt"] == result.scenario.prompt


@pytest.mark.parametrize("rule_id", ["persona-insensitivity", "distractor-text-insensitivity"])
def test_prompt_mutator_is_deterministic_under_seed(rule_id: str) -> None:
    _state, reg = build_ecommerce_registry()
    r1 = make_mutator(rule_id).apply(_scn(), reg, seed=42)
    r2 = make_mutator(rule_id).apply(_scn(), reg, seed=42)
    assert r1.scenario.prompt == r2.scenario.prompt


@pytest.mark.parametrize("rule_id", ["persona-insensitivity", "distractor-text-insensitivity"])
def test_prompt_mutator_passes_registry_through_unchanged(rule_id: str) -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator(rule_id).apply(_scn(), reg, seed=0)
    # Tool registry is not cloned for prompt-only mutators.
    assert result.registry is reg


# -- Checker decision tree ---------------------------------------------------


@pytest.mark.parametrize("rule_id", ["persona-insensitivity", "distractor-text-insensitivity"])
def test_checker_none_when_trajectories_match(rule_id: str) -> None:
    t = _traj([("search_products", {"q": "k"})])
    r = make_equivalence_checker(rule_id).compare(t, t)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


@pytest.mark.parametrize("rule_id", ["persona-insensitivity", "distractor-text-insensitivity"])
def test_checker_reorder_only_is_equivalent(rule_id: str) -> None:
    t1 = _traj([("search_products", {"q": "k"}), ("add_to_cart", {"product_id": "P1"})])
    t2 = _traj([("add_to_cart", {"product_id": "P1"}), ("search_products", {"q": "k"})])
    r = make_equivalence_checker(rule_id).compare(t1, t2)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.REORDER_ONLY


@pytest.mark.parametrize("rule_id", ["persona-insensitivity", "distractor-text-insensitivity"])
def test_checker_tool_set_differs_is_bug(rule_id: str) -> None:
    t1 = _traj([("search_products", {"q": "k"})])
    t2 = _traj([("list_categories", {})])
    r = make_equivalence_checker(rule_id).compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.TOOL_SET_DIFFERS


@pytest.mark.parametrize("rule_id", ["persona-insensitivity", "distractor-text-insensitivity"])
def test_checker_answer_differs_is_bug(rule_id: str) -> None:
    t1 = _traj([("search_products", {"q": "k"})], final="Kettle found: $39.")
    t2 = _traj([("search_products", {"q": "k"})], final="No kettles available here.")
    r = make_equivalence_checker(rule_id).compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.ANSWER_DIFFERS
