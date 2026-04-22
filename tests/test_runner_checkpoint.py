"""Tests for the resumable baseline + Stage-3 runners.

Exercises the manifest + dry-run paths without loading any HF model.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentmorph.runner import (
    RunManifest,
    _passes_rule_filter,
    deterministic_bug_id,
    run_baseline,
    run_stage3,
)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    m = RunManifest()
    m.mark_done(
        model_id="Llama-3.2-3B",
        framework_id="native",
        env_id="ecommerce",
        scenario_id="sx",
        trajectory_id="abc",
    )
    m.save(path)

    loaded = RunManifest.load(path)
    assert loaded.is_done("Llama-3.2-3B", "native", "ecommerce", "sx")
    assert not loaded.is_done("Llama-3.2-3B", "native", "ecommerce", "sy")


def test_corrupt_manifest_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("not-json-at-all", encoding="utf-8")
    m = RunManifest.load(path)
    assert m.completed == {}


def test_dry_run_counts_cells(tmp_path: Path) -> None:
    stats = run_baseline(
        model="Llama-3.2-3B",
        framework="native",
        environment="ecommerce",
        n_scenarios=3,
        out_dir=tmp_path,
        dry_run=True,
    )
    assert stats["dry_run"] is True
    assert stats["total_cells"] == 3  # 1 model × 1 framework × 1 env × 3 scenarios
    assert stats["would_run"] == 3
    assert stats["would_skip"] == 0


def test_dry_run_respects_existing_manifest(tmp_path: Path) -> None:
    # Pre-populate the manifest as if one scenario already ran.
    m = RunManifest()
    m.mark_done(
        model_id="Llama-3.2-3B",
        framework_id="native",
        env_id="ecommerce",
        scenario_id="eco_shop_kettle",
        trajectory_id="t",
    )
    m.save(tmp_path / "manifest.json")

    stats = run_baseline(
        model="Llama-3.2-3B",
        framework="native",
        environment="ecommerce",
        n_scenarios=3,
        out_dir=tmp_path,
        dry_run=True,
    )
    assert stats["would_skip"] == 1
    assert stats["would_run"] == 2


# -- Stage 3 --------------------------------------------------------------


def test_deterministic_bug_id_is_stable() -> None:
    a = deterministic_bug_id(
        "Llama-3.2-3B", "native", "ecommerce",
        "tool-order-invariance", "eco_shop_kettle",
    )
    b = deterministic_bug_id(
        "Llama-3.2-3B", "native", "ecommerce",
        "tool-order-invariance", "eco_shop_kettle",
    )
    assert a == b
    assert len(a) == 16
    # Different inputs → different id.
    c = deterministic_bug_id(
        "Llama-3.2-3B", "native", "ecommerce",
        "tool-order-invariance", "eco_shop_headphones",
    )
    assert a != c


def test_stage3_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    m = RunManifest()
    m.mark_done_stage3(
        model_id="Llama-3.2-3B",
        framework_id="native",
        env_id="ecommerce",
        rule_id="tool-order-invariance",
        scenario_id="eco_shop_kettle",
        pair_id="deadbeef",
        is_bug=True,
        divergence_type="tool_set_differs",
    )
    m.save(path)

    loaded = RunManifest.load(path)
    assert loaded.is_done_stage3(
        "Llama-3.2-3B", "native", "ecommerce",
        "tool-order-invariance", "eco_shop_kettle",
    )
    # Sibling key should still be missing.
    assert not loaded.is_done_stage3(
        "Llama-3.2-3B", "native", "ecommerce",
        "tool-order-invariance", "eco_shop_headphones",
    )
    # Stage 1 key space is disjoint — mark_done_stage3 does NOT satisfy is_done.
    assert not loaded.is_done(
        "Llama-3.2-3B", "native", "ecommerce", "eco_shop_kettle",
    )


def test_stage3_cell_key_includes_rule() -> None:
    k1 = RunManifest.stage3_cell_key(
        "Llama-3.2-3B", "native", "ecommerce",
        "tool-order-invariance", "eco_shop_kettle",
    )
    k2 = RunManifest.stage3_cell_key(
        "Llama-3.2-3B", "native", "ecommerce",
        "schema-paraphrase-invariance", "eco_shop_kettle",
    )
    assert k1 != k2
    assert "tool-order-invariance" in k1
    assert "schema-paraphrase-invariance" in k2


def test_passes_rule_filter_refusal_is_exclusive() -> None:
    # Refusal scenarios only pass for rule 5.
    from agentmorph.environments.ecommerce_env import EcommerceEnvironment
    env = EcommerceEnvironment()
    scenarios = list(env.scenarios())
    refusal = next(s for s in scenarios if s.metadata.get("should_refuse"))
    non_refusal = next(
        s for s in scenarios if not s.metadata.get("should_refuse")
    )

    assert _passes_rule_filter("refusal-consistency", refusal)
    assert not _passes_rule_filter("refusal-consistency", non_refusal)
    # Refusal scenarios are filtered out of every non-refusal rule.
    assert not _passes_rule_filter("tool-order-invariance", refusal)
    assert not _passes_rule_filter("persona-insensitivity", refusal)
    # Non-refusal scenarios accepted by the other rules.
    assert _passes_rule_filter("tool-order-invariance", non_refusal)
    assert _passes_rule_filter("persona-insensitivity", non_refusal)


def test_stage3_dry_run_counts_cells(tmp_path: Path) -> None:
    stats = run_stage3(
        model="Llama-3.2-3B",
        framework="native",
        environment="ecommerce",
        rule=("tool-order-invariance", "persona-insensitivity"),
        n_scenarios=5,  # first 5 seed scenarios — none are refusal
        out_dir=tmp_path,
        dry_run=True,
    )
    assert stats["stage"] == 3
    assert stats["dry_run"] is True
    # 1 model × 1 framework × 1 env × 2 rules × 5 scenarios = 10 cells.
    assert stats["total_cells"] == 10
    assert stats["would_run"] == 10
    assert stats["would_skip"] == 0


def test_stage3_dry_run_filters_refusal_scenarios_for_tool_order(
    tmp_path: Path,
) -> None:
    # Take all 20 scenarios — 2 are refusal. tool-order-invariance should
    # filter those 2 out, leaving 18.
    stats = run_stage3(
        model="Llama-3.2-3B",
        framework="native",
        environment="ecommerce",
        rule=("tool-order-invariance",),
        n_scenarios=None,
        out_dir=tmp_path,
        dry_run=True,
    )
    assert stats["total_cells"] == 18


def test_stage3_dry_run_refusal_rule_only_sees_refusal_scenarios(
    tmp_path: Path,
) -> None:
    stats = run_stage3(
        model="Llama-3.2-3B",
        framework="native",
        environment="ecommerce",
        rule=("refusal-consistency",),
        n_scenarios=None,
        out_dir=tmp_path,
        dry_run=True,
    )
    # Only 2 refusal scenarios in the seed set.
    assert stats["total_cells"] == 2


def test_stage3_dry_run_respects_existing_manifest(tmp_path: Path) -> None:
    m = RunManifest()
    m.mark_done_stage3(
        model_id="Llama-3.2-3B",
        framework_id="native",
        env_id="ecommerce",
        rule_id="tool-order-invariance",
        scenario_id="eco_shop_kettle",
        pair_id="x",
        is_bug=False,
        divergence_type="none",
    )
    m.save(tmp_path / "manifest.json")

    stats = run_stage3(
        model="Llama-3.2-3B",
        framework="native",
        environment="ecommerce",
        rule=("tool-order-invariance",),
        n_scenarios=3,
        out_dir=tmp_path,
        dry_run=True,
    )
    assert stats["would_skip"] == 1
    assert stats["would_run"] == 2


def test_stage3_dry_run_rejects_unknown_rule(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(ValueError, match="unknown rule"):
        run_stage3(
            model="Llama-3.2-3B",
            framework="native",
            environment="ecommerce",
            rule=("not-a-real-rule",),
            n_scenarios=1,
            out_dir=tmp_path,
            dry_run=True,
        )
