"""Tests for the Stage-2 metamorphic-rule protocol + Bug serialisation.

Pins:
  * `RULE_IDS` stays at exactly 10 and in its canonical order (treat it
    as a public API — reordering invalidates HF dataset bug_ids).
  * `make_mutator` / `make_equivalence_checker` give a clear error for
    unknown rule ids and for rules whose module isn't shipped yet.
  * `Bug.to_dict() / from_dict()` round-trips — HF dataset rows can be
    re-loaded into Python without data loss.
  * `DivergenceType` and `Severity` remain string-valued enums with the
    expected members (matches the HF dataset schema).
"""

from __future__ import annotations

import json

import pytest

from agentmorph.rules import RULE_IDS, make_mutator
from agentmorph.rules.base import (
    Bug,
    DivergenceType,
    EquivalenceResult,
    MutationResult,
    Severity,
)
from agentmorph.rules.registry import (
    available_rules,
    make_equivalence_checker,
)


# -- RULE_IDS is the public API ---------------------------------------------


def test_rule_ids_count_is_ten() -> None:
    assert len(RULE_IDS) == 10


def test_rule_ids_are_unique() -> None:
    assert len(set(RULE_IDS)) == len(RULE_IDS)


def test_rule_ids_are_kebab_case() -> None:
    for rid in RULE_IDS:
        assert rid == rid.lower()
        assert "_" not in rid, f"{rid!r} must use hyphens, not underscores"
        assert rid.strip() == rid


def test_rule_ids_canonical_order() -> None:
    """Order matters — it keys the HF dataset. Lock the sequence."""
    assert RULE_IDS == (
        "tool-order-invariance",
        "schema-paraphrase-invariance",
        "synonym-robustness",
        "read-only-idempotency",
        "refusal-consistency",
        "tool-name-insensitivity",
        "parameter-order-invariance",
        "irrelevant-tool-insensitivity",
        "persona-insensitivity",
        "distractor-text-insensitivity",
    )


# -- Factories ---------------------------------------------------------------


def test_make_mutator_unknown_rule_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        make_mutator("not-a-real-rule")


def test_make_mutator_unimplemented_rule_raises_notimplemented() -> None:
    # All 10 rules are unimplemented on Apr 20 — any real rule id here
    # should surface as NotImplementedError (not ImportError).
    with pytest.raises(NotImplementedError):
        make_mutator("tool-order-invariance")


def test_make_equivalence_checker_mirrors_mutator_errors() -> None:
    with pytest.raises(KeyError):
        make_equivalence_checker("not-a-real-rule")
    with pytest.raises(NotImplementedError):
        make_equivalence_checker("tool-order-invariance")


def test_available_rules_reflects_filesystem() -> None:
    """Before any rule ships, `available_rules()` returns an empty list —
    it probes each module for importability and skips missing ones."""
    assert available_rules() == []


# -- MutationResult / EquivalenceResult -------------------------------------


def test_mutation_result_metadata_defaults_empty() -> None:
    # MutationResult needs real scenario + registry; a sentinel pair suffices
    # because we only check metadata here.
    class _Sentinel:
        pass
    r = MutationResult(scenario=_Sentinel(), registry=_Sentinel())
    assert r.metadata == {}


def test_equivalence_result_defaults() -> None:
    r = EquivalenceResult(is_equivalent=True)
    assert r.divergence_type is DivergenceType.NONE
    assert r.details == ""
    assert r.signal is None


# -- DivergenceType / Severity ----------------------------------------------


def test_divergence_type_has_all_expected_members() -> None:
    expected = {
        "NONE", "REORDER_ONLY", "TOOL_SET_DIFFERS", "ANSWER_DIFFERS",
        "SIDE_EFFECTS_DIFFER", "COMPLETION_DIFFERS", "REFUSAL_DIFFERS",
    }
    assert {m.name for m in DivergenceType} == expected


def test_divergence_type_values_are_lowercase() -> None:
    for m in DivergenceType:
        assert m.value == m.value.lower()


def test_severity_has_all_expected_members() -> None:
    assert {m.name for m in Severity} == {"LOW", "MEDIUM", "HIGH", "UNCLASSIFIED"}


# -- Bug serialisation -------------------------------------------------------


def _sample_bug() -> Bug:
    return Bug(
        bug_id="abc123",
        rule_id="tool-order-invariance",
        model_id="Llama-3.2-3B",
        framework_id="langgraph",
        env_id="ecommerce",
        scenario_id="eco_shop_kettle",
        original_trajectory={"steps": [{"kind": "tool_call"}]},
        mutated_trajectory={"steps": [{"kind": "tool_call"}]},
        divergence_type=DivergenceType.TOOL_SET_DIFFERS,
        severity=Severity.MEDIUM,
        details="mutated trajectory skipped add_to_cart",
        mutation_metadata={"permutation": [2, 0, 1, 3]},
    )


def test_bug_to_dict_is_json_serialisable() -> None:
    bug = _sample_bug()
    d = bug.to_dict()
    # Round-trips through json.
    encoded = json.dumps(d, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["bug_id"] == "abc123"
    assert decoded["divergence_type"] == "tool_set_differs"
    assert decoded["severity"] == "medium"


def test_bug_roundtrip_via_from_dict() -> None:
    bug = _sample_bug()
    again = Bug.from_dict(bug.to_dict())
    assert again.bug_id == bug.bug_id
    assert again.divergence_type is DivergenceType.TOOL_SET_DIFFERS
    assert again.severity is Severity.MEDIUM
    assert again.mutation_metadata == {"permutation": [2, 0, 1, 3]}


def test_bug_from_dict_uses_safe_defaults_for_missing_fields() -> None:
    minimal = {
        "bug_id": "x",
        "rule_id": "synonym-robustness",
        "model_id": "Qwen2.5-7B",
        "framework_id": "native",
        "env_id": "ecommerce",
        "scenario_id": "eco_shop_kettle",
        "original_trajectory": {},
        "mutated_trajectory": {},
    }
    bug = Bug.from_dict(minimal)
    assert bug.divergence_type is DivergenceType.NONE
    assert bug.severity is Severity.UNCLASSIFIED
    assert bug.details == ""
    assert bug.mutation_metadata == {}
