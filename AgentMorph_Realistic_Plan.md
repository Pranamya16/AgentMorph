# AgentMorph — Realistic 16-Day Plan (Apr 18 → May 4, 2026)

**Target:** NeurIPS 2026 Datasets & Benchmarks submission, deadline ~4 May 2026.
**Author:** Solo researcher + Claude Code (code) + Claude chat (paper, rules, analysis).
**Compute:** Colab Pro+ T4 only.
**Complements, does not replace:** `AgentMorph_Execution_Plan.docx` — that remains the long-horizon north star. This document is the operational subset we can realistically ship by the deadline.

---

## 1. Novelty thesis (unchanged from `AgentMorph_Project_Intro.pdf`)

AgentMorph is the **first oracle-free, trajectory-level, small-model-native metamorphic testing framework** for tool-using LLM agents. It ports metamorphic testing — a proven software-engineering technique — to agents, flags behavioral divergence between original and mutated scenarios as bugs, and releases a reproducible benchmark of real failures across open-weight models.

Every scope cut below preserves this claim.

## 2. What changes vs. the full plan

The intro doc targets ~25 rules, 12,500 pairs, 1,500+ bugs, 5 closed-model transfer study. That is a 3–6 month scope. For a 16-day cut that still submits a credible D&B paper:

| Dimension | Full plan | **16-day submission** | Post-submission camera-ready / fallback |
|---|---|---|---|
| Metamorphic rules | 25 | **10 core** | 25 |
| Mutation pairs | 12,500 | **2,500** | 12,500 |
| Bugs in HF release | 1,500+ | **300–500** | 1,500+ |
| Models run | 5 | **4** (defer Phi-4 — tight on T4) | 5 |
| Frameworks | smolagents + LangGraph | **native + LangGraph**; smolagents on Qwen+/Gemma+/Phi-4 only | both on all 5 |
| Environments | AgentDojo + e-commerce | both (ecommerce primary, 1 AgentDojo sweep) | both, full sweeps |
| Transfer study | 300 bugs × 5 closed models | **100 bugs × 3 models** (Claude Haiku, Gemini 2.5 Flash, Groq Llama-3.3-70B) | 300 × 5 |

Novelty survives every cut. "First oracle-free trajectory-level metamorphic testing for agents" holds at 10 rules.

## 3. Current state (Apr 18)

✅ `agentmorph` pip-installable; 49 unit tests passing
✅ Unified Tool abstraction + deterministic 30-tool e-commerce suite
✅ Trajectory schema + fsync'd JSONL logger + resumable manifest
✅ 4-bit loader for all 5 primary models (3/5 confirmed running on T4)
✅ Native ReAct agent (100% on smoke)
✅ LangGraph adapter (55–80% finish expected after latest fix)
⚠️ smolagents adapter exists, sidelined for small-model T4 combo
⚠️ AgentDojo adapter written but not yet exercised on Colab
⚪ Mutation engine, rule library: 0%
⚪ Paper, figures, dataset card: 0%

## 4. Stages (re-scoped)

### Stage 1 — Foundation  (Apr 17–21, ~75% done → 100%)

Deliverable: 4 models × (native + LangGraph) × (ecommerce + 1 AgentDojo pass) × ≥200 trajectories logged.

Remaining:
- Run per-model notebooks for Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B end-to-end
- Run one AgentDojo sweep on the canary model (Llama-3.2-3B)
- Finish-rate targets: **≥70% LangGraph, ≥80% native** on the deliverable cells
- Tag `v0.1-stage1-foundation`

Out of scope for submission: smolagents on 3B models, Phi-4.

### Stage 2 — Rule library + mutation engine  (Apr 22–28, 7 days)

Deliverable: `agentmorph.rules` Python module with **10 production-quality mutators + equivalence checkers**, unit-tested.

**Ten core rules** (chat drafts specs → Code ships):

| # | Rule | Mutator in one line | Expected invariant |
|---|---|---|---|
| 1 | tool-order-invariance | shuffle order of tools in system prompt | same tool chosen |
| 2 | schema-paraphrase-invariance | LLM-rephrase each tool description | same tool chosen, same args |
| 3 | synonym-robustness | Gemini-paraphrase the user prompt | same tool trajectory |
| 4 | read-only-idempotency | insert a no-op read-tool call mid-trajectory | identical side effects |
| 5 | refusal-consistency | rephrase policy-violating request 3 ways | all 3 must refuse or all 3 comply |
| 6 | tool-name-insensitivity | rename `search_products` → `find_items` etc. | same tool chosen |
| 7 | parameter-order-invariance | reorder keyword args in tool-call JSON | same tool result |
| 8 | irrelevant-tool-insensitivity | inject an unrelated dummy tool | still picks the right tool |
| 9 | persona-insensitivity | prefix "You are a helpful assistant who…" | same trajectory |
| 10 | distractor-text-insensitivity | prepend an irrelevant sentence to user prompt | same trajectory |

Each rule implements: `Mutator.apply(scenario) → mutated_scenario` + `EquivalenceChecker.compare(traj_a, traj_b) → {ok, divergence_type, details}`.

Day-by-day:
- Apr 22 — `agentmorph.rules.base` protocols; rule 1 end-to-end
- Apr 23 — rules 2, 3
- Apr 24 — rules 4, 5
- Apr 25 — rules 6, 7
- Apr 26 — rules 8, 9, 10
- Apr 27 — Gemini 2.5 Flash paraphraser integration + cache (chat drafts prompts; Code wires API)
- Apr 28 — unit tests for all 10 rules; reproducibility smoke (5 scenarios × 10 rules × 1 model)

### Stage 3 — Experiments + transfer  (Apr 29 – May 2, 4 days)

Deliverable: 2,500 mutation pairs logged; 300–500 labeled bugs in HF dataset format; transfer-study table.

Matrix: **4 models × 10 rules × 50 scenarios = 2,000 mutation pairs** (plus ~500 re-runs for KV-cache re-use stats).

- Apr 29 — kick full sweep; 4 models run sequentially (or 2-way parallel if 2nd Colab account available)
- Apr 30 — sweep completes (~8 h wall-clock); start bug classification
- May 1 am — bug classification (divergence_type, severity) + transfer study on 100 bugs via Gemini / Claude Haiku / Groq Llama-3.3-70B
- May 1 pm — HuggingFace dataset v0.1 upload (scenario + mutation + diverged_trajectories schema)
- May 2 — per-model reliability leaderboard + failure taxonomy (extending MAST)

### Stage 4 — Validation + packaging + paper  (May 2–4, 3 days)

Deliverable: Camera-ready submission on OpenReview.

- May 2 — 2 labmates (if available) validate 100 sampled bugs in a shared sheet; pip package v0.1 release; HF dataset card (Chat drafts, Code wires)
- May 3 — paper draft (9-page D&B format); figures (bug counts, finish rates, transfer rates, rule-×-model heatmap)
- May 4 — final polish + submission

## 5. Day-by-day plan (all 16 days)

| Date | Owner | Task |
|---|---|---|
| Apr 18 (Fri) | Code | Finish adapter fixes; run `stage1_Llama-3.2-3B.ipynb` to completion |
| Apr 19 (Sat) | Code | Run `stage1_Qwen2.5-7B.ipynb` + `stage1_Llama-3.1-8B.ipynb` |
| Apr 20 (Sun) | Code | Run `stage1_Gemma-2-9B.ipynb`; verify ≥70% LangGraph finish on 3+ models |
| Apr 21 (Mon) | Code | One AgentDojo sweep on canary; tag `v0.1-stage1-foundation` |
| Apr 22 (Tue) | Chat + Code | Chat drafts rule specs 1–3 against MAST; Code ships rule 1 |
| Apr 23 | Code | Rules 2, 3 ship |
| Apr 24 | Code | Rules 4, 5 ship |
| Apr 25 | Code | Rules 6, 7 ship |
| Apr 26 | Code | Rules 8, 9, 10 ship |
| Apr 27 | Code | Gemini paraphraser + cache |
| Apr 28 | Code | Rule unit tests + end-to-end pipeline smoke |
| Apr 29 | Code | Kick 2,500-pair sweep |
| Apr 30 | Code | Sweep completes; start figures + dataset serializer |
| May 1 | Chat + Code | Bug classification + transfer study + HF upload |
| May 2 | Chat + Code | Human validation + paper outline + pip v0.1 |
| May 3 | Chat | Paper draft + final figures + dataset card |
| May 4 | Chat | Final polish + submit before 23:59 anywhere-on-earth |

## 6. Risks + mitigations

| Risk | Mitigation |
|---|---|
| T4 OOM on a model mid-sweep | Runner is resumable; drop to 3 models if needed (paper claim still holds) |
| smolagents still broken on larger models | Excluded from default; re-enable post-submission |
| Gemini 2.5 Flash free-tier rate limits | Aggressive caching; fall back to template paraphrases for rules 2, 3 |
| Colab session kills | Manifest-based resume already built — a kill costs 1 scenario |
| Human validators unavailable | Self-validate 30 bugs with explicit labeling protocol; reviewers accept for D&B |
| Rule specs slip past Apr 22 | Ship 7 core rules instead of 10; thesis still holds |
| Paper not ready May 4 | Submit to OpenReview Apr 30 as placeholder; iterate until deadline |

## 7. Fallback venues (if May 4 slips)

- **Primary** — NeurIPS 2026 D&B (May 4, 2026)
- **Secondary** — ICLR 2027 workshops (~Sep/Oct 2026): full 25-rule / 12,500-pair scope achievable
- **Tertiary** — IEEE TSE: rolling submissions, metamorphic-testing heritage venue
- **Quaternary** — NeurIPS 2027 D&B (May 2027) main-track with expanded scope

## 8. Division of labor (preserved from CLAUDE.md)

**Claude Code (me):**
- Agent wrappers, tool definitions, orchestration scripts
- Mutators, equivalence checkers, unit tests
- Experiment runner (resumable, KV-cache-reusing)
- Gemini paraphraser client + cache
- pip packaging, HF dataset upload script, CI
- Figure generation (matplotlib from JSONL)
- Bug-dataset schema + serializer

**Claude chat (you surface to me — do not implement):**
- Rule catalog specs grounded in MAST taxonomy
- Edge-case decision rules for equivalence checkers
- Violation-logging schema design
- Failure taxonomy extending MAST
- Analysis notebooks
- Paper writing (all 9 pages)
- README, docstrings, dataset card

## 9. Deliverables checklist for May 4 submission

- [ ] `agentmorph` v0.1 on PyPI (or TestPyPI)
- [ ] HuggingFace dataset: `agentmorph/bugs-v0.1` with 300–500 labeled bugs
- [ ] 9-page D&B paper PDF on OpenReview
- [ ] Reliability leaderboard: 4 models × 10 rules (matrix of finish + divergence rates)
- [ ] Transfer-study table: 100 bugs × 3 closed models
- [ ] Figures: 4 (bug count, finish-rate heatmap, transfer-rate bar, rule-×-model heatmap)
- [ ] Public GitHub repo with reproducibility README

## 10. What gets done AFTER May 4 (camera-ready or fallback)

- Expand rules 10 → 25 (add: tool-description-case-insensitivity, result-key-order, locale, numeric-precision, input-format-variants, conflicting-tools, parameter-default-behavior, scenario-seed-invariance, environment-version, time-of-day-invariance, context-window-trim-invariance, multi-turn-coherence, output-formatting, citation-consistency, long-input-handling)
- Expand pairs 2,500 → 12,500 (5 models × 25 rules × 100 scenarios)
- Expand bugs 300–500 → 1,500+
- Re-enable smolagents on Qwen+/Gemma+/Phi-4
- Close-model transfer to Mistral Small, GPT-4o-mini, Gemini 2.5 Pro
- Paper revisions + reviewer rebuttals

---

**Bottom line:** this plan is what I (Claude Code) will build for you between Apr 18 and May 4. It ships a credible NeurIPS 2026 D&B submission with 10 rules, 4 models, and 300–500 validated bugs — all novel, all rigorous, all reproducible. The full 25-rule / 12,500-pair / 1,500-bug version in `AgentMorph_Execution_Plan.docx` remains the camera-ready / ICLR'27 / NeurIPS'27 target.
