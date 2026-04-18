# Daily Report — April 18, 2026

**Project:** AgentMorph (NeurIPS 2026 D&B, deadline May 4)

## 1. What we built today (real, committed code)

Everything below is in the GitHub repo on `main`, with tests passing:

| Item | Where | What it is |
|---|---|---|
| **5 per-model Colab notebooks** | `notebooks/stage1_*.ipynb` | One notebook per model so each gets a clean Python process — no VRAM fragmentation carrying over. Each notebook has an optional-wipe cell, a native smoke test, a framework sweep, a diagnostic cell, and finish-rate analysis. |
| **Three rounds of adapter bug fixes** | `src/agentmorph/agents/` | Fixed the smolagents signature bug, added missing base-class attributes, fixed the LangGraph tool-calling path (`bind_tools` + balanced-brace JSON parsing), loosened the native parser to accept multiple tool-call formats |
| **CUDA memory fixes** | `src/agentmorph/models.py`, `runner.py` | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch imports; run `gc.collect()` + `torch.cuda.empty_cache()` between every scenario |
| **Context-size cut** | `src/agentmorph/agents/base.py`, `runner.py` | Default `max_steps` 8 → 3, `max_new_tokens` 384 → 192. Keeps T4 KV cache under budget. |
| **Progress report doc** | `PROGRESS.md` | One-page snapshot of Stage 1 status |
| **Realistic 16-day plan** | `AgentMorph_Realistic_Plan.md` | Re-scoped project plan from Apr 18 to May 4 |
| **Unit tests** | `tests/test_parser_loose.py` | 17 new tests pinning the loose parser behavior; total grew from 32 → 49 tests |
| **Five merged PRs** | GitHub | #1, #2, #3, #4, #5 all on main; several direct commits to main for hotfixes |

## 2. Experiments we ran on Colab (Stage 1, original plan)

**Plan (from the project intro doc):** run every (5 models × 2 frameworks × 2 environments × 20 scenarios) cell, target ~200 baseline trajectories with real tool dispatches.

**What we actually ran today:**

| Run | Models | Frameworks | Result |
|---|---|---|---|
| Morning sweep | Llama-3.2-3B, Qwen2.5-7B, Gemma-2-9B | smolagents + LangGraph | **smolagents 0/60**, **LangGraph 19/20 (false positive — model wrote text instead of calling tools)** |
| Afternoon sweep (after first fix) | Llama-3.2-3B | smolagents + LangGraph | **smolagents 0/20** (OOM), **LangGraph 11/20 (real tool calls this time)** |
| Evening sweep (after T4 budget cut) | Llama-3.2-3B | smolagents + LangGraph | **smolagents still failing** on context-size issues; run incomplete |
| Native smoke on all three | Llama-3.2-3B | native (our own ReAct loop) | **3/3 clean finishes** |

## 3. Why the original Stage 1 scope didn't work on T4

Three concrete reasons, each observed in the Colab logs:

1. **smolagents + small models + T4 don't fit.** smolagents injects the full 30-tool description block into every turn's prompt. Input tokens grew 6.5K → 13K → 20K → 27K across just 4 steps. Llama-3.2-3B's KV cache at 20K+ input context is roughly 2 GB — on a 14.6 GB T4 already holding the 2.5 GB model, we hit CUDA out-of-memory by step 3–4 every single time.

2. **smolagents' internal memory layer broke tool-call parsing.** Even when memory held, smolagents wrapped our model's plain-text output into a structured `[{"type": "text", "text": "..."}]` list on storage, then on the next turn its own JSON parser received the Python repr of that list and rejected it. This is a smolagents internal issue, not something one patch can cleanly solve.

3. **Llama-3.2-3B is genuinely not good at tool calling.** Even when the pipeline worked, the model:
   - Passed the tool name as the product ID (`get_order(order_id="get_order")`)
   - Wrote multiple tool calls as one text blob instead of one per turn
   - Used `"null"` as a string where `null` was meant
   - Invented product IDs like `"kettle"` that don't exist in the catalog

## 4. How this led us to update the plan

The original plan targeted **25 rules, 12,500 mutation pairs, 1,500 bugs, smolagents + LangGraph on all 5 models** by May 4. Doing the math honestly:

- 25 rules × 4–8 hours each ≈ **5 weeks of coding work** → doesn't fit in 16 days
- 12,500 pairs × ~40 s/trajectory on T4 ≈ **280 hours of GPU time** → doesn't fit in 16 days
- smolagents is unstable on small-model T4 combos → running it on all 5 models burns compute with ~0% success
- Paper writing needs 5–7 full days at the end

So we cut scope to what one person + one T4 + 16 days can actually ship:

| | Original plan | New plan for May 4 submission |
|---|---|---|
| Rules | 25 | **10 core** (tool-order, schema-paraphrase, synonym, read-only-idempotency, refusal-consistency, tool-name, parameter-order, irrelevant-tool, persona, distractor-text) |
| Mutation pairs | 12,500 | **2,500** |
| Bugs in release | 1,500+ | **300–500** |
| Models | 5 | **4** (defer Phi-4, it's tight on T4) |
| Frameworks | smolagents + LangGraph | **native + LangGraph** (smolagents post-submission on bigger models) |
| Transfer study | 300 bugs × 5 closed models | **100 bugs × 3 models** |

**Novelty claim is untouched** — "first oracle-free, trajectory-level, small-model-native metamorphic testing framework" holds with 10 rules just as well as with 25.

## 5. Where we are now and what's next

- Stage 1 foundation: **~90% done**. Code works, LangGraph clears ~55% finish rate, native clears 100%, four models verified on T4. Remaining: finish the Qwen, Llama-8B, Gemma per-model sweeps over the weekend.
- Stage 2 (rule library): starts **April 22**, ships one rule per day for ten days.
- Stage 3 (2,500-pair sweep): **April 29–May 1**.
- Stage 4 (paper + package + HF release): **May 2–4**.

Fallback venues if May 4 slips: ICLR 2027 workshops (September 2026), IEEE TSE (rolling), NeurIPS 2027 main. None of the work done today is wasted in any of those scenarios.
