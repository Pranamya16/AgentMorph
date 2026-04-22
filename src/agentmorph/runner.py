"""Resumable experiment runner for Stage 1 and Stage 3.

Stage 1 (baseline) — runs all (model × framework × environment × scenario)
cells once and logs trajectories. That's `run_baseline()`, the original
runner.

Stage 3 (mutation-pair sweep) — for each (model × framework × env × rule ×
scenario) cell, runs the scenario TWICE: original + mutated-via-rule. Feeds
both to the rule's equivalence checker. Writes a bug entry iff the pair
is non-equivalent. That's `run_stage3()`.

Both loops are resumable via the shared `manifest.json`. A Colab kill
mid-run loses at most one cell.

Layout on disk
--------------
    out_dir/
        manifest.json                        # completion log
        trajectories/
            <model>__<framework>__<env>.jsonl           # Stage 1
            <model>__<framework>__<env>__<rule>.jsonl   # Stage 3 pairs
        bugs.jsonl                           # Stage 3 only — non-equivalent pairs

All writes are flushed + fsynced, so a SIGKILL leaves valid, truncated-on-
a-line-boundary JSONL.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from agentmorph.agents import AgentConfig, FRAMEWORK_IDS, make_agent
from agentmorph.environments import ENVIRONMENT_IDS, load_environment
from agentmorph.environments.base import Environment
from agentmorph.models import PRIMARY_MODEL_IDS, clear_cache, load_model, unload_model
from agentmorph.rules import (
    RULE_IDS,
    Bug,
    DivergenceType,
    make_equivalence_checker,
    make_mutator,
)
from agentmorph.rules._shared import snapshot_state
from agentmorph.trajectories import Trajectory, TrajectoryWriter


# -- Manifest ---------------------------------------------------------------


@dataclass
class RunManifest:
    """A tiny completion log persisted to `manifest.json`."""

    created_at: float = field(default_factory=time.time)
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt manifest: treat as empty and overwrite on next save.
            return cls()
        return cls(
            created_at=float(data.get("created_at", time.time())),
            completed=dict(data.get("completed", {})),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"created_at": self.created_at, "completed": self.completed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    @staticmethod
    def cell_key(model_id: str, framework_id: str, env_id: str, scenario_id: str) -> str:
        return f"{model_id}|{framework_id}|{env_id}|{scenario_id}"

    def is_done(self, model_id: str, framework_id: str, env_id: str, scenario_id: str) -> bool:
        return self.cell_key(model_id, framework_id, env_id, scenario_id) in self.completed

    def mark_done(
        self,
        *,
        model_id: str,
        framework_id: str,
        env_id: str,
        scenario_id: str,
        trajectory_id: str,
    ) -> None:
        self.completed[self.cell_key(model_id, framework_id, env_id, scenario_id)] = {
            "trajectory_id": trajectory_id,
            "timestamp": time.time(),
        }

    # -- Stage 3 keys (same manifest, extra `rule_id` slot) -----------------
    #
    # Stage 3 adds rules to the cell coordinates so the same manifest can
    # track both sweeps without collision. Kept as separate methods so the
    # Stage 1 surface stays untouched.

    @staticmethod
    def stage3_cell_key(
        model_id: str, framework_id: str, env_id: str, rule_id: str, scenario_id: str
    ) -> str:
        return f"{model_id}|{framework_id}|{env_id}|{rule_id}|{scenario_id}"

    def is_done_stage3(
        self,
        model_id: str,
        framework_id: str,
        env_id: str,
        rule_id: str,
        scenario_id: str,
    ) -> bool:
        return (
            self.stage3_cell_key(model_id, framework_id, env_id, rule_id, scenario_id)
            in self.completed
        )

    def mark_done_stage3(
        self,
        *,
        model_id: str,
        framework_id: str,
        env_id: str,
        rule_id: str,
        scenario_id: str,
        pair_id: str,
        is_bug: bool,
        divergence_type: str,
    ) -> None:
        self.completed[
            self.stage3_cell_key(model_id, framework_id, env_id, rule_id, scenario_id)
        ] = {
            "pair_id": pair_id,
            "is_bug": is_bug,
            "divergence_type": divergence_type,
            "timestamp": time.time(),
        }


# -- Defaults ---------------------------------------------------------------


DEFAULT_OUT_DIR = Path("runs/stage1_baseline")
DEFAULT_STAGE3_OUT_DIR = Path("runs/stage3_baseline")
# Keep the framework adapters' history short enough that T4 KV caches don't
# blow up. smolagents/LangGraph append every prior turn to the next prompt,
# so input context grows by thousands of tokens per step. With max_steps=3
# and max_new_tokens=192, peak context stays under ~15K — which still fits
# on T4 at 4-bit even with Llama-3.2-3B's full 30-tool system prompt.
DEFAULT_MAX_STEPS = 3

# The one rule that needs 3-way comparison (original + 2 paraphrases).
# Everything else is 2-way. Hard-coded here rather than inferring from the
# rule module so the runner's control flow is obvious at a glance.
REFUSAL_RULE_ID = "refusal-consistency"


def _reclaim_vram() -> None:
    """Best-effort cleanup between scenarios.

    Forces Python GC then returns CUDA allocator blocks to the pool so the
    next scenario starts with the largest possible contiguous VRAM region.
    Silent if torch isn't importable (CPU-only / CI paths).
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # ipc_collect is cheap and helps on long runs.
            torch.cuda.ipc_collect()
    except Exception:
        pass


# -- Core runner ------------------------------------------------------------


def _scenarios(env: Environment, limit: int | None) -> list[Any]:
    out = list(env.scenarios())
    if limit is not None:
        out = out[:limit]
    return out


def _writer_for(out_dir: Path, model_id: str, framework_id: str, env_id: str) -> TrajectoryWriter:
    fname = f"{model_id}__{framework_id}__{env_id}.jsonl"
    return TrajectoryWriter(out_dir / "trajectories" / fname)


def run_baseline(
    *,
    model: str | Iterable[str] = PRIMARY_MODEL_IDS,
    framework: str | Iterable[str] = ("smolagents", "langgraph"),
    environment: str | Iterable[str] = ("ecommerce",),
    n_scenarios: int | None = None,
    out_dir: os.PathLike[str] | str = DEFAULT_OUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float = 0.0,
    dry_run: bool = False,
    hf_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Drive the Stage-1 baseline sweep.

    Parameters match the README quickstart. One model is held on the GPU at a
    time — we fully finish every (framework, env, scenario) cell for one
    model before unloading and moving to the next, which matches how Stage 2
    will reuse KV caches.

    Returns a small dict of counters for the caller (how many cells ran, how
    many were skipped as already-complete, how many errored).
    """
    models = _as_tuple(model)
    frameworks = _as_tuple(framework)
    envs = _as_tuple(environment)

    for m in models:
        if m not in PRIMARY_MODEL_IDS:
            raise ValueError(f"unknown model {m!r}; expected one of {PRIMARY_MODEL_IDS}")
    for f in frameworks:
        if f not in FRAMEWORK_IDS:
            raise ValueError(f"unknown framework {f!r}; expected one of {FRAMEWORK_IDS}")
    for e in envs:
        if e not in ENVIRONMENT_IDS:
            raise ValueError(f"unknown environment {e!r}; expected one of {ENVIRONMENT_IDS}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = RunManifest.load(manifest_path)

    # Pre-compute scenario lists per env so we can do a dry-run cell count
    # without touching the GPU.
    env_objs = {eid: load_environment(eid) for eid in envs}
    env_scenarios = {eid: _scenarios(env_objs[eid], n_scenarios) for eid in envs}

    total_cells = sum(len(env_scenarios[e]) for e in envs) * len(models) * len(frameworks)
    done_cells = 0
    ran_cells = 0
    errored_cells = 0
    skipped_cells = 0

    if dry_run:
        # Report what would run without loading any model.
        for mid in models:
            for fw in frameworks:
                for eid in envs:
                    for s in env_scenarios[eid]:
                        if manifest.is_done(mid, fw, eid, s.id):
                            skipped_cells += 1
                        else:
                            ran_cells += 1
        return {
            "total_cells": total_cells,
            "would_run": ran_cells,
            "would_skip": skipped_cells,
            "dry_run": True,
        }

    # Outer loop: one model at a time, to respect T4 VRAM.
    for mid in models:
        # If every cell for this model is already done, skip the load.
        if all(
            manifest.is_done(mid, fw, eid, s.id)
            for fw in frameworks
            for eid in envs
            for s in env_scenarios[eid]
        ):
            skipped_cells += sum(len(env_scenarios[e]) for e in envs) * len(frameworks)
            continue

        loaded = load_model(mid, hf_cache_dir=hf_cache_dir)

        try:
            for fw in frameworks:
                for eid in envs:
                    env = env_objs[eid]
                    writer = _writer_for(out_dir, mid, fw, eid)
                    try:
                        for scenario in env_scenarios[eid]:
                            if manifest.is_done(mid, fw, eid, scenario.id):
                                skipped_cells += 1
                                done_cells += 1
                                continue

                            try:
                                bundle = env.reset(scenario)
                            except Exception as exc:
                                errored_cells += 1
                                print(
                                    f"[reset-fail] {mid}/{fw}/{eid}/{scenario.id}: "
                                    f"{type(exc).__name__}: {exc}",
                                    file=sys.stderr,
                                )
                                continue

                            config = AgentConfig(
                                model_id=mid,
                                framework_id=fw,
                                max_steps=max_steps,
                                temperature=temperature,
                            )
                            agent = make_agent(
                                fw,
                                loaded_model=loaded,
                                tools=bundle.registry,
                                config=config,
                            )

                            traj = agent.run(
                                prompt=scenario.prompt,
                                scenario_id=scenario.id,
                                env_id=eid,
                                metadata=scenario.metadata,
                            )
                            writer.write(traj)
                            manifest.mark_done(
                                model_id=mid,
                                framework_id=fw,
                                env_id=eid,
                                scenario_id=scenario.id,
                                trajectory_id=traj.trajectory_id,
                            )
                            manifest.save(manifest_path)
                            ran_cells += 1
                            done_cells += 1

                            # Free the KV cache + activation buffers between
                            # scenarios. Without this, T4 runs accumulate
                            # fragmented VRAM over ~3 scenarios and OOM on
                            # the 4th — which is exactly what we saw in the
                            # first real smolagents sweep.
                            _reclaim_vram()
                    finally:
                        writer.close()
        finally:
            unload_model(mid)

    return {
        "total_cells": total_cells,
        "ran": ran_cells,
        "skipped": skipped_cells,
        "errored": errored_cells,
        "out_dir": str(out_dir),
    }


# -- Stage 3: mutation-pair sweep -------------------------------------------


def deterministic_bug_id(
    model_id: str,
    framework_id: str,
    env_id: str,
    rule_id: str,
    scenario_id: str,
) -> str:
    """Stable 16-hex-char bug id.

    Re-running the sweep with the same coordinates produces the same
    bug_id — useful for HF dataset upload idempotency. Stage 3's pair
    id for the trajectory JSONL shares the same hash (just a longer
    prefix).
    """
    payload = f"{model_id}|{framework_id}|{env_id}|{rule_id}|{scenario_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _passes_rule_filter(rule_id: str, scenario: Any) -> bool:
    """Per-rule scenario filter for the Stage 3 sweep.

    Cheapest possible filters — stricter filtering lives in the rule
    checkers themselves. Rule 5 is the only hard gate (only refusal
    scenarios apply). Rules 1 and 7 prefer multi-tool scenarios but we
    relax the gate to ≥ 1 `expects_tool` so each rule still gets ~18-20
    scenarios against the 20-seed ecommerce set (runbook §8 risk 6).
    """
    meta = scenario.metadata if hasattr(scenario, "metadata") else {}
    expects = meta.get("expects_tools", [])

    if rule_id == REFUSAL_RULE_ID:
        return bool(meta.get("should_refuse", False))

    # Rule 1 and 7 need tool calls to be meaningful; skip the two refusal
    # scenarios for every other rule (they're noisy — refusal behaviour
    # masks the rule we're actually testing).
    if meta.get("should_refuse"):
        return False

    if rule_id == "tool-order-invariance":
        # Strictly prefers ≥2 expected tool calls, but rule still runs on
        # ≥1 so we get the full 18 non-refusal scenarios. The checker
        # records REORDER_ONLY for single-tool runs without flagging.
        return len(expects) >= 1

    if rule_id == "parameter-order-invariance":
        # Needs the agent to emit args; `search_products` alone qualifies.
        return len(expects) >= 1

    # All remaining rules (2, 3, 4, 6, 8, 9, 10) accept any non-refusal
    # scenario.
    return True


class _PairWriter:
    """Tiny append-only JSONL writer for Stage 3 trajectory pairs.

    Same durability guarantees as `TrajectoryWriter` (flush + fsync per
    write) — Stage 3 sweeps are the longest single run in the project, so
    a crash must leave the file truncated on a line boundary.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "_PairWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _stage3_pair_writer_for(
    out_dir: Path, model_id: str, framework_id: str, env_id: str, rule_id: str
) -> _PairWriter:
    safe_rule = rule_id.replace("-", "_")
    fname = f"{model_id}__{framework_id}__{env_id}__{safe_rule}.jsonl"
    return _PairWriter(out_dir / "trajectories" / fname)


def _capture_state_safe(bundle: Any) -> dict[str, Any] | None:
    """Best-effort state snapshot; returns None on unexpected shapes.

    Only the ecommerce env produces a meaningful snapshot; AgentDojo
    returns an empty dict today. Stage 3 runs only on ecommerce.
    """
    try:
        snap = snapshot_state(bundle)
    except Exception:
        return None
    return snap if snap else None


def _run_stage3_cell(
    *,
    mid: str,
    fw: str,
    eid: str,
    env: Environment,
    scenario: Any,
    loaded: Any,
    rule_id: str,
    mutator: Any,
    checker: Any,
    max_steps: int,
    temperature: float,
    capture_state: bool,
    pair_writer: _PairWriter,
    bugs_writer: _PairWriter,
) -> tuple[bool, str, str]:
    """Run ONE (model, framework, env, rule, scenario) 2-way pair.

    Returns `(is_bug, divergence_type, pair_id)`. Writes the pair line
    to `pair_writer`; writes a bug line to `bugs_writer` iff divergent.
    """
    seed = int(scenario.metadata.get("seed", 0))
    pair_id = deterministic_bug_id(mid, fw, eid, rule_id, scenario.id)

    # --- Original ----------------------------------------------------------
    bundle_orig = env.reset(scenario)
    state_pre_orig = _capture_state_safe(bundle_orig) if capture_state else None

    config = AgentConfig(
        model_id=mid,
        framework_id=fw,
        max_steps=max_steps,
        temperature=temperature,
    )
    agent_orig = make_agent(
        fw, loaded_model=loaded, tools=bundle_orig.registry, config=config
    )
    traj_orig = agent_orig.run(
        prompt=scenario.prompt,
        scenario_id=scenario.id,
        env_id=eid,
        metadata={**scenario.metadata, "stage": "stage3_original", "rule_id": rule_id},
    )
    state_post_orig = _capture_state_safe(bundle_orig) if capture_state else None
    if capture_state:
        traj_orig.metadata["state_delta"] = {
            "pre": state_pre_orig,
            "post": state_post_orig,
        }

    # --- Mutated -----------------------------------------------------------
    mut = mutator.apply(scenario, bundle_orig.registry, seed=seed)

    # Fresh env reset — do NOT reuse bundle_orig.state (it may have been
    # mutated by the original run's tool calls).
    bundle_mut = env.reset(mut.scenario)
    state_pre_mut = _capture_state_safe(bundle_mut) if capture_state else None

    # Use mutator-returned registry (may be renamed / reordered / with
    # injected tools) rather than bundle_mut.registry.
    agent_mut = make_agent(
        fw, loaded_model=loaded, tools=mut.registry, config=config
    )
    traj_mut = agent_mut.run(
        prompt=mut.scenario.prompt,
        scenario_id=mut.scenario.id,
        env_id=eid,
        metadata={
            **mut.scenario.metadata,
            "stage": "stage3_mutated",
            "rule_id": rule_id,
            "mutation": mut.metadata,
        },
    )
    state_post_mut = _capture_state_safe(bundle_mut) if capture_state else None
    if capture_state:
        traj_mut.metadata["state_delta"] = {
            "pre": state_pre_mut,
            "post": state_post_mut,
        }

    # --- Compare -----------------------------------------------------------
    result = checker.compare(
        traj_orig, traj_mut, mutation_metadata=mut.metadata
    )

    pair_writer.write(
        {
            "pair_id": pair_id,
            "model_id": mid,
            "framework_id": fw,
            "env_id": eid,
            "rule_id": rule_id,
            "scenario_id": scenario.id,
            "seed": seed,
            "is_bug": not result.is_equivalent,
            "divergence_type": result.divergence_type.value,
            "details": result.details,
            "signal": result.signal,
            "mutation_metadata": mut.metadata,
            "original": traj_orig.to_dict(),
            "mutated": traj_mut.to_dict(),
        }
    )

    if not result.is_equivalent:
        bug = Bug(
            bug_id=pair_id,
            rule_id=rule_id,
            model_id=mid,
            framework_id=fw,
            env_id=eid,
            scenario_id=scenario.id,
            original_trajectory=traj_orig.to_dict(),
            mutated_trajectory=traj_mut.to_dict(),
            divergence_type=result.divergence_type,
            details=result.details,
            mutation_metadata=mut.metadata,
        )
        bugs_writer.write(bug.to_dict())

    return (
        not result.is_equivalent,
        result.divergence_type.value,
        pair_id,
    )


def _run_stage3_cell_refusal(
    *,
    mid: str,
    fw: str,
    eid: str,
    env: Environment,
    scenario: Any,
    loaded: Any,
    mutator: Any,
    checker: Any,
    max_steps: int,
    temperature: float,
    pair_writer: _PairWriter,
    bugs_writer: _PairWriter,
) -> tuple[bool, str, str]:
    """Run ONE (model, framework, env, refusal-consistency, scenario) 3-way.

    Uses `mutator.apply_all(...)` to produce the 2 paraphrase variants,
    runs those plus the original, and feeds all 3 to `checker.compare_all`.
    """
    rule_id = REFUSAL_RULE_ID
    seed = int(scenario.metadata.get("seed", 0))
    pair_id = deterministic_bug_id(mid, fw, eid, rule_id, scenario.id)

    config = AgentConfig(
        model_id=mid,
        framework_id=fw,
        max_steps=max_steps,
        temperature=temperature,
    )

    # Original.
    bundle_orig = env.reset(scenario)
    agent_orig = make_agent(
        fw, loaded_model=loaded, tools=bundle_orig.registry, config=config
    )
    traj_orig = agent_orig.run(
        prompt=scenario.prompt,
        scenario_id=scenario.id,
        env_id=eid,
        metadata={
            **scenario.metadata,
            "stage": "stage3_original",
            "rule_id": rule_id,
            "refusal_variant": "original",
        },
    )

    # Paraphrase variants.
    variants = mutator.apply_all(scenario, bundle_orig.registry, seed=seed)
    variant_trajs: list[Any] = []
    variant_metas: list[dict[str, Any]] = []
    for i, v in enumerate(variants):
        bundle_v = env.reset(v.scenario)
        agent_v = make_agent(fw, loaded_model=loaded, tools=v.registry, config=config)
        traj_v = agent_v.run(
            prompt=v.scenario.prompt,
            scenario_id=v.scenario.id,
            env_id=eid,
            metadata={
                **v.scenario.metadata,
                "stage": "stage3_mutated",
                "rule_id": rule_id,
                "refusal_variant": f"variant_{i}",
                "mutation": v.metadata,
            },
        )
        variant_trajs.append(traj_v)
        variant_metas.append(v.metadata)

    # 3-way compare.
    trajectories = [traj_orig, *variant_trajs]
    result = checker.compare_all(
        trajectories,
        mutation_metadata=[{}, *variant_metas],
    )

    pair_writer.write(
        {
            "pair_id": pair_id,
            "model_id": mid,
            "framework_id": fw,
            "env_id": eid,
            "rule_id": rule_id,
            "scenario_id": scenario.id,
            "seed": seed,
            "is_bug": not result.is_equivalent,
            "divergence_type": result.divergence_type.value,
            "details": result.details,
            "signal": result.signal,
            "trajectories": [t.to_dict() for t in trajectories],
            "variant_mutation_metadata": variant_metas,
        }
    )

    if not result.is_equivalent:
        # For 3-way, the "original_trajectory" / "mutated_trajectory" HF
        # fields are populated with original + first divergent variant so
        # the row stays schema-compatible with other rules.
        # The full 3-way record is preserved in the pair JSONL.
        from agentmorph.rules.refusal_consistency import refused
        verdicts = [refused(t) for t in trajectories]
        # Pick first variant whose verdict differs from original (or index 1).
        differing_idx = next(
            (i for i, v in enumerate(verdicts[1:], start=1) if v != verdicts[0]),
            1,
        )
        bug = Bug(
            bug_id=pair_id,
            rule_id=rule_id,
            model_id=mid,
            framework_id=fw,
            env_id=eid,
            scenario_id=scenario.id,
            original_trajectory=traj_orig.to_dict(),
            mutated_trajectory=variant_trajs[differing_idx - 1].to_dict(),
            divergence_type=result.divergence_type,
            details=result.details,
            mutation_metadata={
                "rule_id": rule_id,
                "variants": variant_metas,
                "all_verdicts": verdicts,
                "three_way": True,
            },
        )
        bugs_writer.write(bug.to_dict())

    return (
        not result.is_equivalent,
        result.divergence_type.value,
        pair_id,
    )


def run_stage3(
    *,
    model: str | Iterable[str] = PRIMARY_MODEL_IDS,
    framework: str | Iterable[str] = ("smolagents", "langgraph"),
    environment: str | Iterable[str] = ("ecommerce",),
    rule: str | Iterable[str] | None = None,
    n_scenarios: int | None = None,
    out_dir: os.PathLike[str] | str = DEFAULT_STAGE3_OUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float = 0.0,
    capture_state: bool = True,
    dry_run: bool = False,
    hf_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Drive the Stage-3 mutation-pair sweep.

    For each (model × framework × env × rule × scenario) cell, runs 2
    trajectories (3 for rule 5) and emits a pair line + optional bug.
    Resumes from `manifest.json` via `stage3_cell_key`.
    """
    models = _as_tuple(model)
    frameworks = _as_tuple(framework)
    envs = _as_tuple(environment)
    rules = _as_tuple(rule) if rule is not None else RULE_IDS

    for m in models:
        if m not in PRIMARY_MODEL_IDS:
            raise ValueError(f"unknown model {m!r}; expected one of {PRIMARY_MODEL_IDS}")
    for f in frameworks:
        if f not in FRAMEWORK_IDS:
            raise ValueError(f"unknown framework {f!r}; expected one of {FRAMEWORK_IDS}")
    for e in envs:
        if e not in ENVIRONMENT_IDS:
            raise ValueError(f"unknown environment {e!r}; expected one of {ENVIRONMENT_IDS}")
    for r in rules:
        if r not in RULE_IDS:
            raise ValueError(f"unknown rule {r!r}; expected one of {RULE_IDS}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = RunManifest.load(manifest_path)

    env_objs: dict[str, Environment] = {eid: load_environment(eid) for eid in envs}
    env_scenarios: dict[str, list[Any]] = {
        eid: _scenarios(env_objs[eid], n_scenarios) for eid in envs
    }

    # Eagerly resolve mutator/checker singletons for each rule so we fail
    # fast if any of the 10 modules is missing.
    rule_mutators: dict[str, Any] = {rid: make_mutator(rid) for rid in rules}
    rule_checkers: dict[str, Any] = {rid: make_equivalence_checker(rid) for rid in rules}

    # Pre-compute per-rule scenario lists (post-filter).
    per_rule_scenarios: dict[tuple[str, str], list[Any]] = {}
    for rid in rules:
        for eid in envs:
            per_rule_scenarios[(rid, eid)] = [
                s for s in env_scenarios[eid] if _passes_rule_filter(rid, s)
            ]

    total_cells = sum(
        len(per_rule_scenarios[(rid, eid)])
        for rid in rules
        for eid in envs
    ) * len(models) * len(frameworks)

    ran_cells = 0
    bug_cells = 0
    skipped_cells = 0
    errored_cells = 0
    filtered_cells = 0

    if dry_run:
        # Count without touching any model.
        for mid in models:
            for fw in frameworks:
                for rid in rules:
                    for eid in envs:
                        for s in per_rule_scenarios[(rid, eid)]:
                            if manifest.is_done_stage3(mid, fw, eid, rid, s.id):
                                skipped_cells += 1
                            else:
                                ran_cells += 1
        filtered_cells = sum(
            len(env_scenarios[eid]) - len(per_rule_scenarios[(rid, eid)])
            for rid in rules
            for eid in envs
        ) * len(models) * len(frameworks)
        return {
            "stage": 3,
            "dry_run": True,
            "rules": list(rules),
            "total_cells": total_cells,
            "would_run": ran_cells,
            "would_skip": skipped_cells,
            "filtered_out_by_rule": filtered_cells,
            "out_dir": str(out_dir),
        }

    for mid in models:
        # If every cell for this model is already done, skip the model load.
        all_done = True
        for fw in frameworks:
            for rid in rules:
                for eid in envs:
                    for s in per_rule_scenarios[(rid, eid)]:
                        if not manifest.is_done_stage3(mid, fw, eid, rid, s.id):
                            all_done = False
                            break
                    if not all_done:
                        break
                if not all_done:
                    break
            if not all_done:
                break
        if all_done:
            skipped_cells += sum(
                len(per_rule_scenarios[(rid, eid)])
                for rid in rules
                for eid in envs
            ) * len(frameworks)
            continue

        loaded = load_model(mid, hf_cache_dir=hf_cache_dir)

        try:
            for fw in frameworks:
                for rid in rules:
                    mutator = rule_mutators[rid]
                    checker = rule_checkers[rid]
                    is_refusal = rid == REFUSAL_RULE_ID
                    for eid in envs:
                        env = env_objs[eid]
                        scenarios = per_rule_scenarios[(rid, eid)]
                        if not scenarios:
                            continue
                        pair_writer = _stage3_pair_writer_for(
                            out_dir, mid, fw, eid, rid
                        )
                        bugs_writer = _PairWriter(out_dir / "bugs.jsonl")
                        try:
                            for scenario in scenarios:
                                if manifest.is_done_stage3(
                                    mid, fw, eid, rid, scenario.id
                                ):
                                    skipped_cells += 1
                                    continue
                                try:
                                    if is_refusal:
                                        is_bug, dv, pair_id = _run_stage3_cell_refusal(
                                            mid=mid,
                                            fw=fw,
                                            eid=eid,
                                            env=env,
                                            scenario=scenario,
                                            loaded=loaded,
                                            mutator=mutator,
                                            checker=checker,
                                            max_steps=max_steps,
                                            temperature=temperature,
                                            pair_writer=pair_writer,
                                            bugs_writer=bugs_writer,
                                        )
                                    else:
                                        is_bug, dv, pair_id = _run_stage3_cell(
                                            mid=mid,
                                            fw=fw,
                                            eid=eid,
                                            env=env,
                                            scenario=scenario,
                                            loaded=loaded,
                                            rule_id=rid,
                                            mutator=mutator,
                                            checker=checker,
                                            max_steps=max_steps,
                                            temperature=temperature,
                                            capture_state=capture_state,
                                            pair_writer=pair_writer,
                                            bugs_writer=bugs_writer,
                                        )
                                except Exception as exc:
                                    errored_cells += 1
                                    print(
                                        f"[stage3-fail] {mid}/{fw}/{eid}/{rid}/{scenario.id}: "
                                        f"{type(exc).__name__}: {exc}",
                                        file=sys.stderr,
                                    )
                                    continue

                                if is_bug:
                                    bug_cells += 1
                                ran_cells += 1
                                manifest.mark_done_stage3(
                                    model_id=mid,
                                    framework_id=fw,
                                    env_id=eid,
                                    rule_id=rid,
                                    scenario_id=scenario.id,
                                    pair_id=pair_id,
                                    is_bug=is_bug,
                                    divergence_type=dv,
                                )
                                manifest.save(manifest_path)
                                _reclaim_vram()
                        finally:
                            pair_writer.close()
                            bugs_writer.close()
        finally:
            unload_model(mid)

    return {
        "stage": 3,
        "rules": list(rules),
        "total_cells": total_cells,
        "ran": ran_cells,
        "bugs": bug_cells,
        "skipped": skipped_cells,
        "errored": errored_cells,
        "out_dir": str(out_dir),
    }


# -- CLI --------------------------------------------------------------------


def _as_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(x, str):
        return (x,)
    return tuple(x)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agentmorph-baseline",
        description="Run the Stage-1 baseline or Stage-3 mutation-pair sweep "
        "(resumable via manifest.json).",
    )
    p.add_argument("--model", action="append", default=None,
                   help="Model id (repeatable). Defaults to all 5 primary models.")
    p.add_argument("--framework", action="append", default=None,
                   help="Framework id (repeatable). Defaults to smolagents + langgraph.")
    p.add_argument("--env", action="append", default=None,
                   help="Environment id (repeatable). Defaults to ecommerce.")
    p.add_argument("--n-scenarios", type=int, default=None,
                   help="Truncate each env's scenario list to this many.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir. Defaults to runs/stage1_baseline "
                        "(or runs/stage3_baseline with --stage3).")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--hf-cache-dir", type=str, default=None)

    # Stage 3 flags.
    p.add_argument("--stage3", action="store_true",
                   help="Run the Stage-3 mutation-pair sweep instead of the "
                        "Stage-1 baseline.")
    p.add_argument("--rule", action="append", default=None,
                   help="Rule id (repeatable, Stage-3 only). Defaults to all "
                        "10 rules in RULE_IDS.")
    p.add_argument("--capture-state", action="store_true",
                   help="Stage-3 only: snapshot ShopState before/after each "
                        "run. Required by rule 4 (read-only-idempotency).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    if ns.stage3:
        out_dir = ns.out_dir if ns.out_dir is not None else DEFAULT_STAGE3_OUT_DIR
        stats = run_stage3(
            model=ns.model or PRIMARY_MODEL_IDS,
            framework=ns.framework or ("smolagents", "langgraph"),
            environment=ns.env or ("ecommerce",),
            rule=ns.rule,  # None → defaults to RULE_IDS inside run_stage3
            n_scenarios=ns.n_scenarios,
            out_dir=out_dir,
            max_steps=ns.max_steps,
            temperature=ns.temperature,
            capture_state=ns.capture_state,
            dry_run=ns.dry_run,
            hf_cache_dir=ns.hf_cache_dir,
        )
    else:
        out_dir = ns.out_dir if ns.out_dir is not None else DEFAULT_OUT_DIR
        stats = run_baseline(
            model=ns.model or PRIMARY_MODEL_IDS,
            framework=ns.framework or ("smolagents", "langgraph"),
            environment=ns.env or ("ecommerce",),
            n_scenarios=ns.n_scenarios,
            out_dir=out_dir,
            max_steps=ns.max_steps,
            temperature=ns.temperature,
            dry_run=ns.dry_run,
            hf_cache_dir=ns.hf_cache_dir,
        )
    clear_cache()
    json.dump(stats, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
