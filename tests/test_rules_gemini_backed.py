"""Tests for rules 2 (schema-paraphrase-invariance) and 3 (synonym-robustness).

Both rules depend on the Gemini paraphrase cache. Tests populate the cache
directly in `tmp_path` (no API calls) and verify offline-mode behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentmorph.environments.base import Scenario
from agentmorph.paraphrase import ParaphraseCache, ParaphraseCacheMiss
from agentmorph.rules import (
    RULE_IDS,
    available_rules,
    make_equivalence_checker,
    make_mutator,
)
from agentmorph.rules.base import DivergenceType
from agentmorph.rules.schema_paraphrase_invariance import (
    PARAPHRASE_INSTRUCTION as SCHEMA_INSTRUCTION,
    _SchemaParaphraseInvarianceMutator,
)
from agentmorph.rules.synonym_robustness import (
    PARAPHRASE_INSTRUCTION as SYNONYM_INSTRUCTION,
    _SynonymRobustnessMutator,
)
from agentmorph.tools.ecommerce import build_ecommerce_registry


def _scn() -> Scenario:
    return Scenario(id="eco_shop_kettle", env_id="ecommerce", prompt="Find me a kettle under $50.")


def _traj(tool_calls, final="done"):
    steps = [{"kind": "tool_call", "tool_name": n, "tool_args": a} for n, a in tool_calls]
    if final is not None:
        steps.append({"kind": "final_answer", "content": final})
    return {"steps": steps, "final_answer": final}


# -- Factory registration ----------------------------------------------------


def test_rules_2_and_3_registered_and_importable() -> None:
    assert "schema-paraphrase-invariance" in RULE_IDS
    assert "synonym-robustness" in RULE_IDS
    assert "schema-paraphrase-invariance" in available_rules()
    assert "synonym-robustness" in available_rules()


# -- Rule 2 — schema-paraphrase-invariance ---------------------------------


def test_rule_2_mutator_raises_on_empty_cache_offline(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    mut = _SchemaParaphraseInvarianceMutator(cache=cache)
    _state, reg = build_ecommerce_registry()
    with pytest.raises(ParaphraseCacheMiss):
        mut.apply(_scn(), reg, seed=0, offline=True)


def test_rule_2_mutator_reads_cache_and_replaces_descriptions(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    _state, reg = build_ecommerce_registry()
    # Pre-populate cache with a distinct paraphrase per tool.
    for tool in reg:
        cache.put(
            rule_id="schema-paraphrase-invariance",
            input_text=tool.description,
            output=f"REPHRASED: {tool.description}",
        )
    mut = _SchemaParaphraseInvarianceMutator(cache=cache)
    result = mut.apply(_scn(), reg, seed=0, offline=True)
    for tool in result.registry:
        assert tool.description.startswith("REPHRASED:")


def test_rule_2_mutator_keeps_tool_func_name_and_params(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    _state, reg = build_ecommerce_registry()
    for tool in reg:
        cache.put(
            rule_id="schema-paraphrase-invariance",
            input_text=tool.description,
            output=f"alt: {tool.description}",
        )
    mut = _SchemaParaphraseInvarianceMutator(cache=cache)
    result = mut.apply(_scn(), reg, seed=0, offline=True)
    for tool in result.registry:
        orig = reg.get(tool.name)
        assert tool.func is orig.func
        assert tool.name == orig.name
        assert tool.parameters == orig.parameters


def test_rule_2_checker_none_when_same_tool_and_args() -> None:
    t1 = _traj([("search_products", {"query": "kettle", "max_price": 50})])
    t2 = _traj([("search_products", {"query": "kettle", "max_price": 50})])
    r = make_equivalence_checker("schema-paraphrase-invariance").compare(t1, t2)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


def test_rule_2_checker_tool_set_differs_on_different_tool() -> None:
    t1 = _traj([("search_products", {"query": "kettle"})])
    t2 = _traj([("list_categories", {})])
    r = make_equivalence_checker("schema-paraphrase-invariance").compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.TOOL_SET_DIFFERS


def test_rule_2_instruction_stable() -> None:
    """Pin the paraphrase instruction so any change bumps the HF dataset version."""
    assert "different words" in SCHEMA_INSTRUCTION.lower()
    assert "preserving" in SCHEMA_INSTRUCTION.lower()
    assert "tool description" in SCHEMA_INSTRUCTION.lower()


# -- Rule 3 — synonym-robustness -------------------------------------------


def test_rule_3_mutator_raises_on_empty_cache_offline(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    mut = _SynonymRobustnessMutator(cache=cache)
    _state, reg = build_ecommerce_registry()
    with pytest.raises(ParaphraseCacheMiss):
        mut.apply(_scn(), reg, seed=0, offline=True)


def test_rule_3_mutator_substitutes_prompt_from_cache(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    scn = _scn()
    cache.put(
        rule_id="synonym-robustness",
        input_text=scn.prompt,
        output="Please locate an affordable kettle priced under fifty dollars.",
    )
    mut = _SynonymRobustnessMutator(cache=cache)
    _state, reg = build_ecommerce_registry()
    result = mut.apply(scn, reg, seed=0, offline=True)
    assert (
        result.scenario.prompt
        == "Please locate an affordable kettle priced under fifty dollars."
    )
    assert result.scenario.metadata["original_prompt"] == scn.prompt


def test_rule_3_mutator_falls_back_on_degenerate_paraphrase(tmp_path: Path) -> None:
    """If Gemini returns an identical string, mutator keeps original + logs."""
    cache = ParaphraseCache(cache_dir=tmp_path)
    scn = _scn()
    cache.put(rule_id="synonym-robustness", input_text=scn.prompt, output=scn.prompt)
    mut = _SynonymRobustnessMutator(cache=cache)
    _state, reg = build_ecommerce_registry()
    result = mut.apply(scn, reg, seed=0, offline=True)
    # Prompt unchanged …
    assert result.scenario.prompt == scn.prompt
    # … but the fallback is visible in metadata.
    assert result.scenario.metadata["mutator_extra"]["paraphrased"] is False
    assert (
        result.scenario.metadata["mutator_extra"]["reason"] == "degenerate_output"
    )


def test_rule_3_checker_none_on_matching_trajectories() -> None:
    t = _traj([("search_products", {"q": "kettle", "max_price": 50})])
    r = make_equivalence_checker("synonym-robustness").compare(t, t)
    assert r.is_equivalent is True
    assert r.divergence_type is DivergenceType.NONE


def test_rule_3_checker_answer_differs_flagged_as_bug() -> None:
    t1 = _traj([("search_products", {"q": "kettle"})], final="Found Electric Kettle $39.")
    t2 = _traj([("search_products", {"q": "kettle"})], final="No kettles found sorry.")
    r = make_equivalence_checker("synonym-robustness").compare(t1, t2)
    assert r.is_equivalent is False
    assert r.divergence_type is DivergenceType.ANSWER_DIFFERS


def test_rule_3_instruction_stable() -> None:
    assert "different words" in SYNONYM_INSTRUCTION.lower()
    assert "preserving" in SYNONYM_INSTRUCTION.lower()
