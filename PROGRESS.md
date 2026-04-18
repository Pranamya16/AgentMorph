# AgentMorph — Stage 1 Progress Report

**Date:** 2026-04-18   **Stage:** 1 (Foundation)   **Target:** NeurIPS 2026 D&B, deadline ~4 May 2026

## Delivered (code on `main`)

| Component | Path | Status |
|---|---|---|
| `agentmorph` Python package (pip-installable) | `pyproject.toml`, `src/agentmorph/` | ✅ installable via `pip install -e ".[all]"` |
| Unified Tool abstraction (JSON-Schema params + `read_only` flag) | `tools/base.py` | ✅ |
| Synthetic 30-tool e-commerce suite (8 domain modules, deterministic fixture) | `tools/ecommerce/` | ✅ exact count asserted at import |
| 4-bit quantized model loader for all 5 primary models | `models.py` | ✅ lazy torch/transformers imports |
| Trajectory schema + fsync'd JSONL writer | `trajectories.py` | ✅ Colab-kill-safe |
| Agent adapters: native / smolagents / LangGraph | `agents/` | ✅ 3 adapters, ⚠️ 2 have in-flight bugs (see below) |
| Environment adapters: ecommerce + AgentDojo | `environments/` | ✅ 20 seed scenarios for ecommerce |
| Resumable baseline runner + CLI (`agentmorph-baseline`) | `runner.py` | ✅ manifest-based resume |
| Colab bootstrap + Stage 1 notebook | `notebooks/` | ✅ tested on Colab T4 |
| Unit tests | `tests/` | ✅ **32 / 32 passing** (~0.7s, no GPU required) |

**Repository:** https://github.com/Pranamya16/AgentMorph (1 merged PR: adapter signature fix)
**Total LoC (src/):** ~2,200   **Total LoC (tests/):** ~430

## Executed on Colab T4 (first real sweep, in progress)

| Metric | Value |
|---|---|
| GPU verified | Tesla T4, 14.6 GB VRAM |
| HF gated-model access obtained | Llama-3.2-3B, Llama-3.1-8B, Gemma-2-9B |
| Models successfully loaded + run at 4-bit | Llama-3.2-3B, Qwen2.5-7B, Gemma-2-9B (3 / 5) |
| Model weights cached on Drive | ~25 GB, persists across session kills |
| Completed (model × framework × env × scenario) cells | **120 / 200** |
| Trajectory files on Drive | 6 files, 20 rows each, all schema-valid |
| Mechanical pipeline end-to-end | ✅ runs, resumes, fsyncs, writes |

## Open Issues Blocking Stage 1 "usefulness"

Stage 1 deliverable per the execution plan: *"Working pipeline: any of 5 models × 2 frameworks × 2 environments, reproducible end-to-end. ~200 baseline trajectories logged."* — currently **mechanically** true but finish rate on useful trajectories is low.

1. **smolagents adapter** — `AgentGenerationError: '_Wrapped' object has no attribute …` (truncated) on every run. Secondary bug downstream of the PR #1 fix.
2. **LangGraph adapter** — 0 / 60 finished trajectories. Chat-model shim likely not emitting structured `AIMessage.tool_calls`.
3. **Native adapter** — parser too strict (rejects `python`-fenced JSON, prose-prefixed FINAL lines). Smoke test: 2 / 5 scenarios finished.
4. **Runner counter misleading** — `errored: 0` reports only runner-level reset failures, not in-trajectory agent failures.

**Net finish rate across the deliverable frameworks (smolagents + LangGraph):** 0 / 120 so far.

## Immediate Next Actions

1. Fix all three adapters in a single PR (~40 diff lines; error strings captured from Colab).
2. Split Stage 1 notebook into 5 per-model variants (same out-dir; manifest keys prevent collisions) to isolate Phi-4 OOM risk from the other 4 models.
3. Wipe `runs/stage1_baseline/` and re-run on the canary model (Llama-3.2-3B). Target finish rate ≥ 80% on smolagents + LangGraph before committing compute to the full 5-model sweep.
4. Let current 120 → 200 sweep complete (Phi-4 + Llama-3.1-8B) to confirm those two models load on T4 independent of adapter bugs.

## Stage Gate Status

| Stage | % complete (my est.) | Gating item |
|---|---|---|
| **1 — Foundation** | ~75 % | Adapter bugs + one clean full sweep |
| **2 — Rule library + mutation engine** | 0 % | Rule catalog (chat-side, grounded in MAST) not yet drafted |
| **3 — Large-scale experiments + transfer** | 0 % | Blocked on Stages 1 & 2 |
| **4 — Human validation + paper** | 0 % | Blocked on Stage 3 |

## Tangibles You Can Hand to a Reviewer Today

- A running, resumable experiment runner that survives Colab kills.
- A deterministic 30-tool synthetic environment with 20 curated scenarios.
- 120 schema-validated trajectories on Google Drive.
- 32 passing unit tests covering tools, trajectories, ecommerce suite, agent parser, and checkpoint logic.
- CLI: `agentmorph-baseline --dry-run` counts cells + verifies resume on any machine, no GPU required.
