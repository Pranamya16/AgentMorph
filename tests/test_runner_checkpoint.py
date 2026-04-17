"""Tests for the resumable baseline runner.

Exercises the manifest + dry-run path without loading any HF model.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentmorph.runner import RunManifest, run_baseline


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
