# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

Stage 1 foundation has landed. The `agentmorph` package is installable and the baseline runner is wired end-to-end, though the HF model-loading path is only exercisable on a Colab T4 (or equivalent CUDA box). The execution plan (`AgentMorph_Execution_Plan.docx`) remains the single source of truth for scope, model choices, and deliverables — consult it before making architectural decisions.

## Project goal

AgentMorph is a solo research project targeting a **NeurIPS 2026 Datasets & Benchmarks submission (deadline ~4 may 2026)**. It applies metamorphic testing to LLM agents: mutate agent scenarios under semantics-preserving rules, then flag behavior divergence as bugs.

Deliverables: `agentmorph` pip-installable library + HuggingFace bug dataset + 9-page D&B paper. Fallback venues: ICLR 2027 workshops, IEEE TSE.

## Execution constraints that shape design

- **Compute budget: Colab Pro+ only.** All 5 primary models run 4-bit quantized on a T4. Any code that assumes persistent GPU state or long-running processes is wrong — Colab sessions die unpredictably.
- **Experiment runner must be resumable** via checkpoints, and must aggressively reuse the KV cache across mutation pairs of the same scenario. This is a hard requirement for Stage 3's ~12,500 mutation pairs to fit in the compute budget.
- **Solo author.** No code review partner, no ops team — favor fewer moving parts over clever abstractions.

## Stage roadmap (what to build, in order)

| Stage | Deliverable |
|---|---|
| 1 — Foundation | Agent test-bed: 5 open models × 2 frameworks (smolagents + LangGraph) × 2 environments (AgentDojo + synthetic 30-tool e-commerce suite). ~200 baseline trajectories logged. |
| 2 — Rule library + mutation engine | `agentmorph-core` Python library: ~25 mutators (tool-order invariance, schema paraphrase, read-only idempotency, synonym robustness, refusal consistency, …) grounded in the MAST failure taxonomy, plus equivalence checkers, all unit-tested. |
| 3 — Large-scale experiments + transfer | Run 5 models × 25 rules × ~100 scenarios ≈ 12,500 mutation pairs. Transfer study on 300 bugs against Groq Llama-3.3-70B, Gemini 2.5 Flash, Mistral Small, Claude Haiku. Output: ~1,500 labeled agent bugs, per-model reliability leaderboard. |
| 4 — Human validation, packaging, paper | Human-validate 100 sampled bugs (recruit 2–3 labmates). Pip-package `agentmorph`, release HF dataset, write NeurIPS D&B paper. |

## Fixed technical choices (from the plan — do not substitute without reason)

- **Primary models** (local, 4-bit on T4): Llama-3.2-3B, Qwen2.5-7B, Gemma-2-9B, Phi-4, Llama-3.1-8B.
- **Agent frameworks**: smolagents + LangGraph (both must be supported).
- **Environments**: AgentDojo + a synthetic 30-tool e-commerce suite built in Stage 1.
- **LLM-guided paraphrasing** inside the mutation engine: Gemini 2.5 Flash (free tier) — chosen to keep the paraphraser off the critical-path compute budget.
- **Transfer-study models** (API-based, Stage 3): Groq Llama-3.3-70B, Gemini 2.5 Flash, Mistral Small, Claude Haiku (starter credits).

## Division of labor between Claude (chat) and Claude Code

The plan explicitly separates these roles — preserve the split when deciding what to do yourself vs. what to hand back:

- **Claude Code (this tool)**: agent wrappers, tool definitions, orchestration scripts, mutators + unit tests, experiment runner with resumable checkpoints, pip packaging, CI, figure generation.
- **Claude (chat, not this tool)**: tool-schema design for the synthetic suite, rule catalog drafting (grounded in MAST), failure debugging, violation-logging schema, analysis notebooks, paper drafting, README/docstrings/dataset card.

If a task is on the chat side, surface it to the user rather than fully implementing it.

## Package layout (Stage 1)

```
src/agentmorph/
  tools/base.py                — unified Tool, ToolRegistry, ToolCall, ToolResult
  tools/ecommerce/             — synthetic 30-tool e-commerce suite (8 domain files)
  trajectories.py              — Trajectory, TrajectoryStep, StepKind, TrajectoryWriter
  models.py                    — MODEL_REGISTRY + 4-bit `load_model` (lazy torch/transformers)
  agents/base.py               — AgentConfig + NativeAgent (ReAct loop; works without extras)
  agents/smolagents_agent.py   — smolagents adapter (lazy import)
  agents/langgraph_agent.py    — LangGraph adapter (lazy import)
  agents/registry.py           — `make_agent(framework_id, ...)`
  environments/ecommerce_env.py — ~20 seed scenarios + fixture resets
  environments/agentdojo_env.py — AgentDojo suite adapter (lazy import)
  runner.py                    — `run_baseline()` + CLI entrypoint, manifest-based resume
```

The CLI entry point is `agentmorph-baseline` (declared in `pyproject.toml`).

## Commands

Install locally (CPU-only, no models):

```
pip install -e ".[dev]"
```

Install on Colab (all extras — models + both frameworks + AgentDojo):

```
pip install -e ".[all]"
```

Run the test suite (framework-free; ~0.5s, no GPU required):

```
python -m pytest tests/ -x
```

Run a single test file or node:

```
python -m pytest tests/test_ecommerce_suite.py -x
python -m pytest tests/test_runner_checkpoint.py::test_dry_run_counts_cells
```

Dry-run the baseline sweep (no GPU needed — counts cells + verifies resume):

```
python -m agentmorph.runner --model Llama-3.2-3B --framework native --env ecommerce --n-scenarios 3 --dry-run
```

Run the full Stage-1 baseline on Colab T4 (all 5 models × smolagents+langgraph × ecommerce, resumable):

```
python -m agentmorph.runner --hf-cache-dir /content/drive/MyDrive/AgentMorph/hf_cache \
    --out-dir /content/drive/MyDrive/AgentMorph/runs/stage1_baseline
```

Trajectories land in `<out_dir>/trajectories/<model>__<framework>__<env>.jsonl`; `<out_dir>/manifest.json` tracks completed (model, framework, env, scenario) cells and makes the runner safe to kill and re-launch.

## GPU preferred, CPU fallback everywhere

**GPU is the primary execution target** — 4-bit nf4 quantization via bitsandbytes on a T4 is how Stage 1/3 actually runs at scale. But the **entire pipeline must also run on CPU as a degraded fallback**, so a developer without a GPU can exercise adapters, mutation rules, runner logic, and even small-model inference end-to-end.

How the fallback works:

- `load_model()` auto-detects `torch.cuda.is_available()`. On CUDA it loads at 4-bit nf4. On CPU it falls back to **fp32 without quantization** (bitsandbytes requires CUDA — there's no way to keep 4-bit without it), prints a loud warning about the ~100× speed penalty, and continues.
- `LoadedModel.chat()` uses `self.model.device` throughout — already device-agnostic.
- `unload_model()` + `_reclaim_vram()` gate their CUDA calls behind `torch.cuda.is_available()` — safe on CPU.
- Notebooks print GPU availability in §1 as a warning, never a hard assert. Only the `runner` subprocess actually needs the decision — and `load_model()` makes it automatically.

**Practical CPU-fallback envelope** (Colab free tier, ~12 GB RAM):

- ✅ `agentmorph` package imports, tool definitions, 30-tool ecommerce suite, agent adapters, trajectory I/O, manifest, mutation rules + equivalence checkers, figure generation, unit tests.
- ✅ `load_model("Llama-3.2-3B")` in fp32 ≈ ~6 GB RAM, inference minutes-per-token — viable for small-scale smoke runs.
- ⚠️ `load_model("Qwen2.5-7B")` or larger in fp32 ≈ 14 GB+ RAM — **will likely OOM on standard Colab CPU**. These need either more RAM or a GPU.

**Test the fallback without a GPU:**
```
python -m agentmorph.runner --model Llama-3.2-3B --framework native --env ecommerce --n-scenarios 1
```
Works locally on any machine with 8+ GB RAM. The runner will print the CPU fallback warning and complete (slowly) one scenario.

**Do NOT add GPU hard-asserts to new notebook cells or Python modules.** If a cell happens to need CUDA (e.g. a hypothetical fused-kernel accelerator cell), let `load_model()` / torch raise the underlying error with its own message. The rest of the pipeline stays CPU-compatible.

## Stage 1 invariants future work must preserve

- **`ECOMMERCE_TOOL_NAMES` stays at exactly 30.** It's asserted at module load (`tools/ecommerce/__init__.py`) and tested — add/remove tools only by updating both the tuple and the domain builders.
- **`MODEL_REGISTRY` holds exactly the 5 primary models from the plan.** Changing this set changes Stage 3 scope; do not edit silently.
- **`LoadedModel` is the only type that touches torch/transformers.** Keep every other module importable without the `[models]` extra installed — the unit tests depend on this.
- **Every `TrajectoryWriter.write()` flushes + fsyncs.** Colab kills are the baseline threat model; do not batch writes.
- **The runner holds at most one `LoadedModel` in VRAM.** `unload_model()` is called before moving to the next model. Tighten, don't loosen, this invariant.
