"""Tests for rule 6 — tool-name-insensitivity."""

from __future__ import annotations

from agentmorph.environments.base import Scenario
from agentmorph.rules import (
    RULE_IDS,
    available_rules,
    make_equivalence_checker,
    make_mutator,
)
from agentmorph.rules._name_map import INVERSE_NAME_MAP, NAME_MAP
from agentmorph.rules.base import DivergenceType
from agentmorph.tools.ecommerce import ECOMMERCE_TOOL_NAMES, build_ecommerce_registry


def _scn() -> Scenario:
    return Scenario(id="eco_shop_kettle", env_id="ecommerce", prompt="find kettle")


def _traj(tool_calls, final="done"):
    steps = [{"kind": "tool_call", "tool_name": n, "tool_args": a} for n, a in tool_calls]
    if final is not None:
        steps.append({"kind": "final_answer", "content": final})
    return {"steps": steps, "final_answer": final}


# -- Name map invariants (Stage 2 depends on these) --------------------------


def test_name_map_covers_every_ecommerce_tool() -> None:
    assert set(NAME_MAP.keys()) == set(ECOMMERCE_TOOL_NAMES)


def test_name_map_is_a_bijection() -> None:
    assert len(NAME_MAP) == len(set(NAME_MAP.values())) == 30


def test_inverse_name_map_round_trips() -> None:
    for old, new in NAME_MAP.items():
        assert INVERSE_NAME_MAP[new] == old


def test_rename_targets_dont_collide_with_original_tool_names() -> None:
    # If a renamed target matched a real tool, mutation would produce an
    # ambiguous registry. Guard that.
    assert set(NAME_MAP.keys()).isdisjoint(set(NAME_MAP.values()))


# -- Registry factory --------------------------------------------------------


def test_rule_6_is_registered_and_importable() -> None:
    assert "tool-name-insensitivity" in RULE_IDS
    assert "tool-name-insensitivity" in available_rules()
    assert make_mutator("tool-name-insensitivity").rule_id == "tool-name-insensitivity"
    assert make_equivalence_checker("tool-name-insensitivity").rule_id == "tool-name-insensitivity"


# -- Mutator contract --------------------------------------------------------


def test_mutator_renames_every_ecommerce_tool() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("tool-name-insensitivity").apply(_scn(), reg, seed=0)
    assert set(result.registry.names()) == set(NAME_MAP.values())


def test_mutator_preserves_original_registry() -> None:
    _state, reg = build_ecommerce_registry()
    pre = set(reg.names())
    make_mutator("tool-name-insensitivity").apply(_scn(), reg, seed=0)
    assert set(reg.names()) == pre


def test_mutator_emits_name_map_metadata() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("tool-name-insensitivity").apply(_scn(), reg, seed=0)
    assert result.metadata["name_map"] == NAME_MAP
    assert result.metadata["inverse_name_map"] == INVERSE_NAME_MAP


def test_renamed_tool_retains_original_behaviour() -> None:
    _state, reg = build_ecommerce_registry()
    result = make_mutator("tool-name-insensitivity").apply(_scn(), reg, seed=0)
    # `find_items` in new registry must behave like `search_products` in old.
    new_search = result.registry.get(NAME_MAP["search_products"])
    old_search = reg.get("search_products")
    # Same JSON-Schema params, same description.
    assert new_search.parameters == old_search.parameters
    assert new_search.description == old_search.description


# -- Checker decision tree ---------------------------------------------------


def test_checker_none_when_mutated_uses_renamed_version_of_same_tools() -> None:
    t1 = _traj([("search_products", {"q": "k"}), ("add_to_cart", {"product_id": "P020"})])
    t2 = _traj(
        [(NAME_MAP["search_products"], {"q": "k"}), (NAME_MAP["add_to_cart"], {"product_id": "P020"})]
    )
    r = make_equivalence_checker("tool-name-insensitivity").compare(
        t1, t2, mutation_metadata={"inverse_name_map": INVERSE_NAME_MAP}
    )
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


def test_checker_tool_set_differs_when_mutated_picked_wrong_tool() -> None:
    t1 = _traj([("search_products", {"q": "k"})])
    # Mutated agent called `enumerate_sections` (list_categories) instead of
    # `find_items` (search_products). That's the bug we flag.
    t2 = _traj([(NAME_MAP["list_categories"], {})])
    r = make_equivalence_checker("tool-name-insensitivity").compare(
        t1, t2, mutation_metadata={"inverse_name_map": INVERSE_NAME_MAP}
    )
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.TOOL_SET_DIFFERS


def test_checker_falls_back_to_module_inverse_map_without_metadata() -> None:
    """Resilience: if metadata is absent, checker still uses the static map."""
    t1 = _traj([("search_products", {"q": "k"})])
    t2 = _traj([(NAME_MAP["search_products"], {"q": "k"})])
    r = make_equivalence_checker("tool-name-insensitivity").compare(t1, t2)
    assert r.is_equivalent is True
