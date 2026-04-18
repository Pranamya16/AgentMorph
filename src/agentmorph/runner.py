"""Resumable Stage-1 baseline runner.

Runs all (model × framework × environment × scenario) cells, one model at a
time, loading+unloading each LoadedModel once to keep peak T4 VRAM bounded.

Resume semantics
----------------
A `manifest.json` lives alongside the per-cell JSONL files. Each trajectory
appended to disk is paired with an entry `{key, trajectory_id, timestamp}`
in the manifest. On startup we load the manifest and skip any (model,
framework, env, scenario) tuple whose key already appears, so a Colab kill
mid-run loses at most the scenario that was in flight.

Layout on disk
--------------
    out_dir/
        manifest.json                    # completion log
        trajectories/
            <model>__<framework>__<env>.jsonl

All writes are flushed + fsynced, so a SIGKILL leaves valid, truncated-on-
a-line-boundary JSONL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agentmorph.agents import AgentConfig, FRAMEWORK_IDS, make_agent
from agentmorph.environments import ENVIRONMENT_IDS, load_environment
from agentmorph.environments.base import Environment
from agentmorph.models import PRIMARY_MODEL_IDS, clear_cache, load_model, unload_model
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


# -- Defaults ---------------------------------------------------------------


DEFAULT_OUT_DIR = Path("runs/stage1_baseline")
# Keep the framework adapters' history short enough that T4 KV caches don't
# blow up. smolagents/LangGraph append every prior turn to the next prompt,
# so input context grows by thousands of tokens per step. With max_steps=4
# and max_new_tokens=256, peak context stays under ~10K — well inside T4.
DEFAULT_MAX_STEPS = 4


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


# -- CLI --------------------------------------------------------------------


def _as_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(x, str):
        return (x,)
    return tuple(x)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agentmorph-baseline",
        description="Run the Stage-1 baseline sweep with resumable checkpoints.",
    )
    p.add_argument("--model", action="append", default=None,
                   help="Model id (repeatable). Defaults to all 5 primary models.")
    p.add_argument("--framework", action="append", default=None,
                   help="Framework id (repeatable). Defaults to smolagents + langgraph.")
    p.add_argument("--env", action="append", default=None,
                   help="Environment id (repeatable). Defaults to ecommerce.")
    p.add_argument("--n-scenarios", type=int, default=None,
                   help="Truncate each env's scenario list to this many.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--hf-cache-dir", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    stats = run_baseline(
        model=ns.model or PRIMARY_MODEL_IDS,
        framework=ns.framework or ("smolagents", "langgraph"),
        environment=ns.env or ("ecommerce",),
        n_scenarios=ns.n_scenarios,
        out_dir=ns.out_dir,
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
