"""Helper to generate the Stage-3 dress-rehearsal + per-model full-sweep notebooks.

Notebook content lives here as plain Python strings so it's easy to
diff and edit. Running this script rewrites the .ipynb files.

Layout:
  * `stage3_dress_rehearsal.ipynb` — single-model smoke run (Llama-3.2-3B).
  * `stage3_full_sweep_<MODEL>.ipynb` × 5 — one per open-weight model.
    All 5 share `runs/stage3_baseline/` on Drive; the runner's manifest
    coordinates resume/skip cell-by-cell. Mirrors the Stage-1 per-model
    pattern (`stage1_<MODEL>.ipynb`) for the same reasons: fresh Python
    process per model avoids VRAM leak between model swaps, each
    notebook is bounded at ~2-5 h wall-clock so a Colab kill loses at
    most one model's progress.

Not imported by the package — kept in notebooks/ next to the outputs.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Canonical launch order (smallest -> largest VRAM footprint on T4 4-bit).
# Matches the Stage-1 pattern; the canary runs first so any adapter
# regression is caught on the cheapest model.
# ---------------------------------------------------------------------------

MODEL_ORDER: list[str] = [
    "Llama-3.2-3B",
    "Qwen2.5-7B",
    "Llama-3.1-8B",
    "Gemma-2-9B",
    "Phi-4",
]


# ---------------------------------------------------------------------------
# Dress-rehearsal notebook (single-model smoke)
# ---------------------------------------------------------------------------

DRESS_REHEARSAL_CELLS: list[dict] = [
    md(
        "# AgentMorph - Stage 3 Dress Rehearsal (smoke test)\n\n"
        "~40-cell micro-sweep to catch runner/mutator/checker bugs before "
        "kicking any of the per-model full-sweep notebooks.\n\n"
        "**Scope (small on purpose):**\n"
        "- 1 model: **Llama-3.2-3B** (canary, cheapest to load)\n"
        "- 2 rules: **tool-order-invariance** (no Gemini, simplest) + "
        "**refusal-consistency** (Gemini-backed, 3-way compare)\n"
        "- 1 framework: **native** (LangGraph is exercised in the full sweep)\n"
        "- 1 env: **ecommerce**\n\n"
        "**Expected cell count** after per-rule filtering:\n"
        "- tool-order-invariance: 1 model x 1 framework x 18 non-refusal scenarios = 18\n"
        "- refusal-consistency: 1 model x 1 framework x 2 refusal scenarios = 2 cells (each = 3 trajectories)\n"
        "- **Total: 20 cells, ~24 trajectories**\n\n"
        "**Wall-clock on T4:** 15-25 min.\n\n"
        "**Pass gate:** Section 12 prints `READY FOR FULL SWEEP: YES`.\n\n"
        "**Before you run:** Runtime > Change runtime type > **T4 GPU**. Have your HF token ready for Section 3.\n"
        "The paraphrase cache (committed to `main` in the previous step) must be on the branch you pull."
    ),
    md("## 1. GPU sanity check"),
    code(
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    dev = torch.cuda.get_device_properties(0)\n"
        "    print(f'CUDA OK: {torch.cuda.get_device_name(0)} ({dev.total_memory / 1024**3:.1f} GB VRAM)')\n"
        "else:\n"
        "    print('NOTE: no CUDA - pipeline falls back to CPU (very slow). '\n"
        "          'Dress rehearsal will only complete in a reasonable time on GPU.')"
    ),
    md("## 2. Clone / pull the repo"),
    code(
        "import os\n"
        'REPO_URL = "https://github.com/Pranamya16/AgentMorph.git"\n'
        'REPO_DIR = "/content/AgentMorph"\n'
        "if not os.path.exists(REPO_DIR):\n"
        "    !git clone {REPO_URL} {REPO_DIR}\n"
        "else:\n"
        "    !cd {REPO_DIR} && git fetch origin && git checkout main && git pull --ff-only\n"
        "!cd {REPO_DIR} && git log -1 --oneline"
    ),
    md(
        "## 3. HuggingFace auth\n\n"
        "Paste your token, run the cell, then **clear the token** before sharing the notebook."
    ),
    code(
        "from huggingface_hub import login\n"
        'HF_TOKEN = "hf_REPLACE_ME"\n'
        "login(token=HF_TOKEN)"
    ),
    md("## 4. Mount Drive + install extras"),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    ),
    code(
        "# For the dress rehearsal we only need native + gemini (no smolagents/langgraph).\n"
        "!pip install -q -e /content/AgentMorph[models,langgraph,gemini]"
    ),
    md(
        "## 5. Verify the paraphrase cache is populated\n\n"
        "Stage 3 calls `paraphrase(..., offline=True)` during the sweep. "
        "If the cache isn't on `main`, the Gemini-backed rules raise "
        "`ParaphraseCacheMiss` and the whole rule's cells error out.\n\n"
        "Expect: 3 files totalling 54 entries "
        "(30 + 20 + 4 for rules 2, 3, 5)."
    ),
    code(
        "import pathlib\n"
        "cache_dir = pathlib.Path('/content/AgentMorph/runs/paraphrase_cache')\n"
        "if not cache_dir.exists():\n"
        "    raise SystemExit('paraphrase cache not committed to main. '\n"
        "                     'Re-run notebooks/stage2_seed_paraphrases.ipynb and commit.')\n"
        "for p in sorted(cache_dir.glob('*.jsonl')):\n"
        "    n = sum(1 for _ in p.open())\n"
        "    print(f'{p.name}: {n} entries')"
    ),
    md(
        "## 6. Dress-rehearsal parameters\n\n"
        "Single model on purpose - this notebook only confirms the pipeline "
        "works end-to-end. The full sweep is one-notebook-per-model."
    ),
    code(
        'OUT_DIR = "/content/drive/MyDrive/AgentMorph/runs/stage3_dress"\n'
        'HF_CACHE = "/content/drive/MyDrive/AgentMorph/hf_cache"\n'
        'MODEL = "Llama-3.2-3B"\n'
        'RULES = ["tool-order-invariance", "refusal-consistency"]\n'
        'FRAMEWORKS = ["native"]\n'
        'N_SCENARIOS = 20  # all 20 seed scenarios; rule filters prune as needed\n'
        "print('dress-rehearsal config:')\n"
        "print('  model     :', MODEL)\n"
        "print('  rules     :', RULES)\n"
        "print('  frameworks:', FRAMEWORKS)"
    ),
    md(
        "## 7. Dry-run: count the cells that will actually run\n\n"
        "No model loaded. Sanity-checks the filter + manifest before we commit GPU time."
    ),
    code(
        "def _flags():\n"
        "    f = ['--model', MODEL]\n"
        "    for r in RULES: f += ['--rule', r]\n"
        "    for fw in FRAMEWORKS: f += ['--framework', fw]\n"
        "    return f\n"
        "flags = ' '.join(_flags())\n"
        "!cd /content/AgentMorph && python -m agentmorph.runner --stage3 --dry-run \\\n"
        "    {flags} --env ecommerce --n-scenarios {N_SCENARIOS} \\\n"
        "    --out-dir {OUT_DIR}"
    ),
    md(
        "## 8. Run the dress rehearsal\n\n"
        "Resumable: if Colab kills the runtime, re-run this cell and the "
        "manifest skips already-completed cells."
    ),
    code(
        "import os\n"
        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
        "flags = ' '.join(_flags())\n"
        "!cd /content/AgentMorph && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n"
        "    python -m agentmorph.runner --stage3 \\\n"
        "    {flags} --env ecommerce --n-scenarios {N_SCENARIOS} \\\n"
        "    --capture-state \\\n"
        "    --hf-cache-dir {HF_CACHE} \\\n"
        "    --out-dir {OUT_DIR}"
    ),
    md(
        "## 9. Summarize: manifest, bug count, divergence breakdown"
    ),
    code(
        "import json, collections, pathlib\n"
        "out = pathlib.Path(OUT_DIR)\n"
        "manifest = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {'completed': {}}\n"
        "print(f'manifest cells: {len(manifest.get(\"completed\", {}))}')\n"
        "\n"
        "bugs_path = out / 'bugs.jsonl'\n"
        "if bugs_path.exists():\n"
        "    bugs = [json.loads(l) for l in bugs_path.open()]\n"
        "    print(f'bugs.jsonl rows: {len(bugs)}')\n"
        "    by_rule = collections.Counter(b['rule_id'] for b in bugs)\n"
        "    by_dv = collections.Counter(b['divergence_type'] for b in bugs)\n"
        "    by_model = collections.Counter(b['model_id'] for b in bugs)\n"
        "    print('  by rule:         ', dict(by_rule))\n"
        "    print('  by divergence:   ', dict(by_dv))\n"
        "    print('  by model:        ', dict(by_model))\n"
        "else:\n"
        "    print('no bugs.jsonl yet - either every pair passed or the run was pure error.')"
    ),
    md(
        "## 10. Peek one pair JSONL line\n\n"
        "Useful for visually confirming the pair record contains both "
        "`original` + `mutated` trajectories + the `mutation_metadata`."
    ),
    code(
        "import json, pathlib\n"
        "traj_dir = pathlib.Path(OUT_DIR) / 'trajectories'\n"
        "for p in sorted(traj_dir.glob('*.jsonl'))[:1]:\n"
        "    print(f'\\n=== {p.name} ===')\n"
        "    line = p.open().readline()\n"
        "    if not line:\n"
        "        print('(empty)')\n"
        "        continue\n"
        "    row = json.loads(line)\n"
        "    print('pair_id          :', row.get('pair_id'))\n"
        "    print('rule_id          :', row.get('rule_id'))\n"
        "    print('scenario_id      :', row.get('scenario_id'))\n"
        "    print('is_bug           :', row.get('is_bug'))\n"
        "    print('divergence_type  :', row.get('divergence_type'))\n"
        "    if 'original' in row:\n"
        "        o = row['original']\n"
        "        print(f'original: steps={len(o[\"steps\"])} final={str(o[\"final_answer\"])[:80]!r}')\n"
        "        m = row['mutated']\n"
        "        print(f'mutated : steps={len(m[\"steps\"])} final={str(m[\"final_answer\"])[:80]!r}')\n"
        "    elif 'trajectories' in row:\n"
        "        print(f'3-way trajectories: {len(row[\"trajectories\"])}')\n"
        "        for i, t in enumerate(row['trajectories']):\n"
        "            print(f'  [{i}] steps={len(t[\"steps\"])} final={str(t[\"final_answer\"])[:80]!r}')"
    ),
    md(
        "## 11. Error audit\n\n"
        "Separate **pipeline errors** from **benign agent outcomes**.\n\n"
        "- *Pipeline error* = a Python exception in the model / tokenizer / agent code "
        "(AttributeError, TypeError, OOM, bad JSON, etc). A zero-step trajectory is also "
        "a pipeline error. These block the full sweep and must be zero (or near-zero) "
        "before we proceed.\n"
        "- *Benign outcome* = the agent ran fine but didn't produce a parseable "
        "`FINAL:` answer within `max_steps`. This includes:\n"
        "    - `max_steps exhausted` (agent looped or stalled)\n"
        "    - `unparseable step: ...` (the model answered in plain English - common "
        "for refusal scenarios where the model says \"I can't fulfill that\" instead "
        "of using our ReAct format).\n\n"
        "Benign outcomes are **real Stage 3 data**, not bugs in our pipeline."
    ),
    code(
        "import json, pathlib, collections\n"
        "\n"
        "traj_dir = pathlib.Path(OUT_DIR) / 'trajectories'\n"
        "\n"
        "BENIGN_PREFIXES = (\n"
        "    'max_steps exhausted',\n"
        "    'unparseable step:',\n"
        ")\n"
        "\n"
        "def _classify_trajectory(t):\n"
        "    steps = t.get('steps', []) if t else []\n"
        "    if not steps:\n"
        "        return 'pipeline'\n"
        "    worst = 'clean'\n"
        "    for s in steps:\n"
        "        if s.get('kind') != 'error':\n"
        "            continue\n"
        "        content = str(s.get('content', ''))\n"
        "        if any(content.startswith(p) for p in BENIGN_PREFIXES):\n"
        "            worst = 'benign' if worst == 'clean' else worst\n"
        "        else:\n"
        "            return 'pipeline'\n"
        "    return worst\n"
        "\n"
        "total_pairs = 0\n"
        "pipeline_err_pairs = 0\n"
        "benign_err_pairs = 0\n"
        "clean_pairs = 0\n"
        "error_content_samples = collections.Counter()\n"
        "\n"
        "for p in traj_dir.glob('*.jsonl'):\n"
        "    for line in p.open():\n"
        "        row = json.loads(line)\n"
        "        total_pairs += 1\n"
        "        trajs = row.get('trajectories') or [row.get('original'), row.get('mutated')]\n"
        "        trajs = [t for t in trajs if t is not None]\n"
        "        verdicts = [_classify_trajectory(t) for t in trajs]\n"
        "        if 'pipeline' in verdicts:\n"
        "            pipeline_err_pairs += 1\n"
        "            for t in trajs:\n"
        "                for s in t.get('steps', []):\n"
        "                    if s.get('kind') == 'error':\n"
        "                        c = str(s.get('content', ''))[:60]\n"
        "                        if not any(c.startswith(p) for p in BENIGN_PREFIXES):\n"
        "                            error_content_samples[c] += 1\n"
        "                            break\n"
        "        elif 'benign' in verdicts:\n"
        "            benign_err_pairs += 1\n"
        "        else:\n"
        "            clean_pairs += 1\n"
        "\n"
        "print(f'total pairs                        : {total_pairs}')\n"
        "print(f'  clean (produced final answer)    : {clean_pairs}')\n"
        "print(f'  benign (max_steps / unparseable) : {benign_err_pairs}')\n"
        "print(f'  pipeline error (PROBLEM)         : {pipeline_err_pairs}')\n"
        "if error_content_samples:\n"
        "    print()\n"
        "    print('Pipeline-error samples:')\n"
        "    for msg, n in error_content_samples.most_common(10):\n"
        "        print(f'  {n:3d}  {msg}')"
    ),
    md(
        "## 12. Gate check\n\n"
        "The gate passes iff the **pipeline** is healthy. Benign outcomes are "
        "normal - small open-weight models often loop or answer in prose. The full "
        "sweep will still collect useful trajectories from them; the divergence "
        "checkers consume the trajectory shape, not a clean `FINAL:` answer.\n\n"
        "If the gate passes, open `stage3_full_sweep_Llama-3.2-3B.ipynb` on a "
        "fresh Colab runtime and kick it. Run the other 4 per-model notebooks "
        "in sequence (one per Colab session) as time allows."
    ),
    code(
        "gates = {\n"
        "    'cells_completed'         : len(manifest.get('completed', {})),\n"
        "    'bugs_found'              : (bugs_path.exists() and sum(1 for _ in bugs_path.open())) or 0,\n"
        "    'trajectory_files'        : sum(1 for _ in (pathlib.Path(OUT_DIR) / 'trajectories').glob('*.jsonl')),\n"
        "    'pipeline_error_rate'     : round(pipeline_err_pairs / max(total_pairs, 1), 3),\n"
        "    'benign_outcome_rate'     : round(benign_err_pairs   / max(total_pairs, 1), 3),\n"
        "    'clean_final_answer_rate' : round(clean_pairs        / max(total_pairs, 1), 3),\n"
        "}\n"
        "print('GATES:')\n"
        "for k, v in gates.items():\n"
        "    print(f'  {k:28s} {v}')\n"
        "\n"
        "ready = (\n"
        "    gates['cells_completed'] > 0\n"
        "    and gates['trajectory_files'] >= 1\n"
        "    and gates['pipeline_error_rate'] <= 0.05\n"
        ")\n"
        "print('\\nREADY FOR FULL SWEEP:', 'YES' if ready else 'NO - debug first')\n"
        "print()\n"
        "print('Next step: stage3_full_sweep_Llama-3.2-3B.ipynb on a fresh T4 runtime.')"
    ),
]


# ---------------------------------------------------------------------------
# Per-model full-sweep notebook factory
# ---------------------------------------------------------------------------


def _safe_slug(model_id: str) -> str:
    """Filename-safe model slug (keep dots / dashes, same as Stage 1)."""
    return model_id


def _order_md_list(order: list[str], highlight: str) -> str:
    """Bullet list of the 5 notebooks with the current one marked."""
    lines: list[str] = []
    for i, m in enumerate(order, start=1):
        marker = " **<- you are here**" if m == highlight else ""
        lines.append(f"  {i}. `stage3_full_sweep_{m}.ipynb`{marker}")
    return "\n".join(lines)


def full_sweep_cells(model_id: str, order: list[str]) -> list[dict]:
    """Build the cell list for ONE per-model Stage-3 sweep notebook.

    The 5 per-model notebooks share `runs/stage3_baseline/` on Drive. The
    runner's manifest key is `model|framework|env|rule|scenario`, so the
    5 notebooks write disjoint key subsets. Trajectory JSONL filenames
    are `<model>__<framework>__<env>__<rule>.jsonl` - also disjoint per
    model.
    """
    assert model_id in order, f"{model_id!r} not in launch order"
    idx = order.index(model_id)
    next_model = order[idx + 1] if idx + 1 < len(order) else None

    header_body = (
        f"# AgentMorph - Stage 3 Full Sweep - **{model_id}**\n\n"
        f"Per-model notebook. Runs **one model** (this one: `{model_id}`) "
        "x 2 frameworks x 10 rules x 20 scenarios on a fresh Colab session.\n\n"
        "**Why per-model:** each notebook is a fresh Python process, so VRAM "
        "from the previous model can't leak into this one (we hit an actual "
        "OOM doing sequential model loads in a single process during the "
        "dress rehearsal). All 5 notebooks share "
        "`/content/drive/MyDrive/AgentMorph/runs/stage3_baseline/` on Drive; "
        "the runner's manifest key includes `model_id` so there are no "
        "file collisions.\n\n"
        "**Recommended launch order (smallest -> largest VRAM on T4):**\n"
        f"{_order_md_list(order, model_id)}\n\n"
        "**Per-model wall-clock on T4 (4-bit):** 2-5 h.\n\n"
        "**Scope (THIS notebook only):**\n"
        f"- 1 model: `{model_id}`\n"
        "- 2 frameworks: native + langgraph (smolagents excluded; blows T4 context)\n"
        "- 10 rules: all of `RULE_IDS`\n"
        "- 1 env: ecommerce, all 20 scenarios\n"
        "- **Expected cells: ~324** (2 fw x (9 rules x 18 non-refusal + 1 rule x 2 refusal))\n\n"
        "**Before you run:** Runtime > Change runtime type > **T4 GPU**. "
        "Have your HF token ready for Section 3. The dress rehearsal "
        "(`stage3_dress_rehearsal.ipynb`) must be green before launching any "
        "full-sweep notebook."
    )

    next_step_md = (
        "## 12. Next step\n\n"
        + (
            f"This model is done. Next: open "
            f"**`stage3_full_sweep_{next_model}.ipynb`** on a fresh Colab "
            f"runtime and run it top to bottom. The shared manifest will "
            f"pick up where this notebook left off."
            if next_model is not None
            else (
                "**This is the last model in the recommended order.** All 5 "
                "per-model notebooks are now complete. The shared "
                "`runs/stage3_baseline/` on Drive should have ~1,640 completed "
                "cells and ~150 bug rows.\n\n"
                "Next: bug classification (chat-side, 50-bug stratified "
                "sample), transfer study, HF dataset upload, and paper "
                "figures. Those are tracked in the 14-day plan."
            )
        )
    )

    return [
        md(header_body),
        md("## 1. GPU sanity check"),
        code(
            "import torch\n"
            "if not torch.cuda.is_available():\n"
            "    print('WARN: no CUDA. Full sweep on CPU is infeasible - switch to T4 and restart.')\n"
            "else:\n"
            "    dev = torch.cuda.get_device_properties(0)\n"
            "    print(f'CUDA OK: {torch.cuda.get_device_name(0)} ({dev.total_memory / 1024**3:.1f} GB VRAM)')"
        ),
        md("## 2. Clone / pull the repo"),
        code(
            "import os\n"
            'REPO_URL = "https://github.com/Pranamya16/AgentMorph.git"\n'
            'REPO_DIR = "/content/AgentMorph"\n'
            "if not os.path.exists(REPO_DIR):\n"
            "    !git clone {REPO_URL} {REPO_DIR}\n"
            "else:\n"
            "    !cd {REPO_DIR} && git fetch origin && git checkout main && git pull --ff-only\n"
            "!cd {REPO_DIR} && git log -1 --oneline"
        ),
        md(
            "## 3. HuggingFace auth\n\n"
            "Paste your token, run, then **clear the token** before sharing this notebook."
        ),
        code(
            "from huggingface_hub import login\n"
            'HF_TOKEN = "hf_REPLACE_ME"\n'
            "login(token=HF_TOKEN)"
        ),
        md("## 4. Mount Drive + install extras"),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')"
        ),
        code(
            "# Per-model sweeps only need native + langgraph + gemini (no smolagents).\n"
            "!pip install -q -e /content/AgentMorph[models,langgraph,gemini]"
        ),
        md(
            "## 5. Verify the paraphrase cache\n\n"
            "Hard dependency: rules 2, 3, 5 all raise `ParaphraseCacheMiss` in "
            "offline mode without this. Expect 30 + 20 + 4 = 54 entries across "
            "3 files."
        ),
        code(
            "import pathlib\n"
            "cache_dir = pathlib.Path('/content/AgentMorph/runs/paraphrase_cache')\n"
            "if not cache_dir.exists():\n"
            "    raise SystemExit('paraphrase cache missing - re-run stage2_seed_paraphrases.ipynb')\n"
            "files = sorted(cache_dir.glob('*.jsonl'))\n"
            "assert len(files) == 3, f'expected 3 cache files, got {len(files)}'\n"
            "counts = {p.name: sum(1 for _ in p.open()) for p in files}\n"
            "for k, v in counts.items():\n"
            "    print(f'{k}: {v} entries')\n"
            "assert counts.get('schema_paraphrase_invariance.jsonl', 0) >= 30, 'schema cache incomplete'\n"
            "assert counts.get('synonym_robustness.jsonl', 0) >= 20, 'synonym cache incomplete'\n"
            "assert counts.get('refusal_consistency.jsonl', 0) >= 4, 'refusal cache incomplete'\n"
            "print('cache OK')"
        ),
        md(f"## 6. Sweep parameters (this notebook runs `{model_id}` only)"),
        code(
            'OUT_DIR = "/content/drive/MyDrive/AgentMorph/runs/stage3_baseline"\n'
            'HF_CACHE = "/content/drive/MyDrive/AgentMorph/hf_cache"\n'
            f'MODEL = "{model_id}"\n'
            'FRAMEWORKS = ["native", "langgraph"]\n'
            'N_SCENARIOS = 20\n'
            "print('per-model sweep config:')\n"
            "print('  model     :', MODEL)\n"
            "print('  frameworks:', FRAMEWORKS)\n"
            "print('  rules     : all 10 (RULE_IDS default)')\n"
            "print('  out_dir   :', OUT_DIR)"
        ),
        md(
            "## 7. (Optional) Wipe prior cells for this model only\n\n"
            "Mirrors the Stage-1 per-model notebook pattern. Prunes ONLY this "
            "model's rows from the shared `manifest.json` and deletes its "
            "trajectory JSONL files. Leaves other models' progress on Drive "
            "untouched. Run this ONLY after an adapter fix when you want to "
            "retry this model's cells from scratch. Skip otherwise - the "
            "runner is already idempotent on resume."
        ),
        code(
            "import json, pathlib\n"
            "RUN_DIR = pathlib.Path(OUT_DIR)\n"
            "manifest_path = RUN_DIR / 'manifest.json'\n"
            "traj_dir = RUN_DIR / 'trajectories'\n"
            "\n"
            "if manifest_path.exists():\n"
            "    data = json.loads(manifest_path.read_text())\n"
            "    before = len(data.get('completed', {}))\n"
            "    data['completed'] = {\n"
            "        k: v for k, v in data.get('completed', {}).items()\n"
            "        if not k.startswith(MODEL + '|')\n"
            "    }\n"
            "    after = len(data['completed'])\n"
            "    manifest_path.write_text(json.dumps(data, indent=2))\n"
            "    print(f'manifest: {before} -> {after} entries (dropped {before - after} for {MODEL})')\n"
            "else:\n"
            "    print('no manifest - nothing to prune')\n"
            "\n"
            "if traj_dir.exists():\n"
            "    for p in traj_dir.glob(f'{MODEL}__*.jsonl'):\n"
            "        p.unlink()\n"
            "        print('deleted:', p.name)\n"
            "else:\n"
            "    print('no trajectory dir yet')"
        ),
        md(
            "## 8. Dry-run: verify the cell count for this model\n\n"
            "Target: ~324 cells (2 frameworks x (9 rules x 18 non-refusal + "
            "1 rule x 2 refusal) scenarios). Cells already done on a previous "
            "run count as `would_skip`."
        ),
        code(
            "def _flags():\n"
            "    f = ['--model', MODEL]\n"
            "    for fw in FRAMEWORKS:\n"
            "        f += ['--framework', fw]\n"
            "    return f\n"
            "flags = ' '.join(_flags())\n"
            "!cd /content/AgentMorph && python -m agentmorph.runner --stage3 --dry-run \\\n"
            "    {flags} --env ecommerce --n-scenarios {N_SCENARIOS} \\\n"
            "    --out-dir {OUT_DIR}"
        ),
        md(
            f"## 9. Run `{model_id}` sweep\n\n"
            "Resumable: if Colab kills the runtime mid-sweep, re-run this cell "
            "from a fresh runtime (re-running Sections 1-4 first) and the "
            "manifest skips already-completed cells. One cell loss per kill, max.\n\n"
            "**Disconnect resilience.** Output is also `tee`'d to "
            f"`sweep_{model_id}.log` on Drive, so if Colab disconnects you can read "
            "what was happening when it died. Cell 9b below tails that log + "
            "checks per-model manifest progress without needing to re-run the "
            "sweep cell.\n\n"
            "**Keep the browser tab active.** Idle tabs are the #1 cause of "
            "Colab disconnects. Don't put your laptop to sleep, don't navigate "
            "away. The optional Section 9c installs a lightweight JS keep-alive."
        ),
        code(
            "import os, pathlib\n"
            "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
            "flags = ' '.join(_flags())\n"
            "LOG = f'{OUT_DIR}/sweep_{MODEL}.log'\n"
            "pathlib.Path(LOG).parent.mkdir(parents=True, exist_ok=True)\n"
            "!cd /content/AgentMorph && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n"
            "    python -m agentmorph.runner --stage3 \\\n"
            "    {flags} --env ecommerce --n-scenarios {N_SCENARIOS} \\\n"
            "    --capture-state \\\n"
            "    --hf-cache-dir {HF_CACHE} \\\n"
            "    --out-dir {OUT_DIR} 2>&1 | tee {LOG}"
        ),
        md(
            "## 9b. Watch progress (run anytime, even after a disconnect)\n\n"
            "Pulls the manifest from Drive and tells you exactly how many "
            "cells are done for this model. Also tails the log file. Run this "
            "**from a separate cell** any time you want to check status, or "
            "after a Colab reconnect to see whether you should re-run Section 9."
        ),
        code(
            "import json, pathlib\n"
            "manifest_path = pathlib.Path(OUT_DIR) / 'manifest.json'\n"
            "if manifest_path.exists():\n"
            "    completed = json.loads(manifest_path.read_text()).get('completed', {})\n"
            "    my_done = [k for k in completed if k.startswith(MODEL + '|')]\n"
            "    print(f'{MODEL} cells done: {len(my_done)} / 328')\n"
            "    by_rule = {}\n"
            "    for k in my_done:\n"
            "        rule = k.split('|')[3]\n"
            "        by_rule[rule] = by_rule.get(rule, 0) + 1\n"
            "    for r in sorted(by_rule):\n"
            "        print(f'  {r:38s} {by_rule[r]}')\n"
            "else:\n"
            "    print('manifest not yet written - sweep has not produced any cell yet')\n"
            "\n"
            "log_path = pathlib.Path(f'{OUT_DIR}/sweep_{MODEL}.log')\n"
            "if log_path.exists():\n"
            "    print(f'\\n--- last 30 lines of {log_path.name} ---')\n"
            "    !tail -30 {log_path}"
        ),
        md(
            "## 9c. (Optional) Keep-alive for the browser tab\n\n"
            "Colab disconnects idle tabs after ~90 minutes. Run this cell once "
            "before launching Section 9 to install a JavaScript heartbeat that "
            "auto-clicks the connect button every minute. Not bullet-proof "
            "(Colab's anti-idle has gotten stricter), but reduces disconnect "
            "frequency a lot."
        ),
        code(
            "from IPython.display import Javascript, display\n"
            "display(Javascript('''\n"
            "    function keepAlive() {\n"
            "        const btn = document.querySelector(\"colab-connect-button\");\n"
            "        if (btn) btn.click();\n"
            "    }\n"
            "    setInterval(keepAlive, 60000);\n"
            "    console.log(\"AgentMorph keep-alive installed\");\n"
            "'''))\n"
            "print('keep-alive installed - reload the page to remove it')"
        ),
        md(
            f"## 10. Post-run tally for `{model_id}` only\n\n"
            "Filters the shared manifest + bugs.jsonl to just this model's "
            "rows. Run this after Section 9 finishes to see per-model progress."
        ),
        code(
            "import json, collections, pathlib\n"
            "out = pathlib.Path(OUT_DIR)\n"
            "manifest = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {'completed': {}}\n"
            "completed = manifest.get('completed', {})\n"
            "\n"
            "my_keys = [k for k in completed if k.startswith(MODEL + '|')]\n"
            "print(f'cells for {MODEL}: {len(my_keys)}  (shared manifest total: {len(completed)})')\n"
            "\n"
            "my_bug_keys = sum(1 for k in my_keys if completed[k].get('is_bug'))\n"
            "print(f'bug cells for {MODEL} (from manifest): {my_bug_keys}')\n"
            "\n"
            "bugs_path = out / 'bugs.jsonl'\n"
            "if bugs_path.exists():\n"
            "    bugs = [json.loads(l) for l in bugs_path.open() if json.loads(l).get('model_id') == MODEL]\n"
            "    print(f'{MODEL} rows in bugs.jsonl: {len(bugs)}')\n"
            "    if bugs:\n"
            "        by_rule = collections.Counter(b['rule_id'] for b in bugs)\n"
            "        by_dv = collections.Counter(b['divergence_type'] for b in bugs)\n"
            "        print('\\nby rule:')\n"
            "        for k, v in sorted(by_rule.items()):\n"
            "            print(f'  {k:38s} {v}')\n"
            "        print('\\nby divergence type:')\n"
            "        for k, v in sorted(by_dv.items()):\n"
            "            print(f'  {k:25s} {v}')\n"
            "\n"
            "traj_dir = out / 'trajectories'\n"
            "if traj_dir.exists():\n"
            "    files = sorted(traj_dir.glob(f'{MODEL}__*.jsonl'))\n"
            "    n_pairs = sum(sum(1 for _ in p.open()) for p in files)\n"
            "    print(f'\\n{MODEL} trajectory JSONL files: {len(files)}')\n"
            "    print(f'{MODEL} pair records: {n_pairs}')"
        ),
        md(
            f"## 11. Peek one pair JSONL line for `{model_id}`\n\n"
            "Quick visual confirmation that the pair record is well-formed."
        ),
        code(
            "import json, pathlib\n"
            "traj_dir = pathlib.Path(OUT_DIR) / 'trajectories'\n"
            "for p in sorted(traj_dir.glob(f'{MODEL}__*.jsonl'))[:1]:\n"
            "    print(f'\\n=== {p.name} ===')\n"
            "    line = p.open().readline()\n"
            "    if not line:\n"
            "        print('(empty)')\n"
            "        continue\n"
            "    row = json.loads(line)\n"
            "    print('pair_id          :', row.get('pair_id'))\n"
            "    print('rule_id          :', row.get('rule_id'))\n"
            "    print('scenario_id      :', row.get('scenario_id'))\n"
            "    print('is_bug           :', row.get('is_bug'))\n"
            "    print('divergence_type  :', row.get('divergence_type'))\n"
            "    if 'original' in row:\n"
            "        o = row['original']\n"
            "        print(f'original: steps={len(o[\"steps\"])} final={str(o[\"final_answer\"])[:80]!r}')\n"
            "        m = row['mutated']\n"
            "        print(f'mutated : steps={len(m[\"steps\"])} final={str(m[\"final_answer\"])[:80]!r}')\n"
            "    elif 'trajectories' in row:\n"
            "        print(f'3-way trajectories: {len(row[\"trajectories\"])}')\n"
            "        for i, t in enumerate(row['trajectories']):\n"
            "            print(f'  [{i}] steps={len(t[\"steps\"])} final={str(t[\"final_answer\"])[:80]!r}')"
        ),
        md(next_step_md),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = notebook(cells)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    write_notebook(
        NOTEBOOK_DIR / "stage3_dress_rehearsal.ipynb", DRESS_REHEARSAL_CELLS
    )
    for mid in MODEL_ORDER:
        write_notebook(
            NOTEBOOK_DIR / f"stage3_full_sweep_{_safe_slug(mid)}.ipynb",
            full_sweep_cells(mid, MODEL_ORDER),
        )

    # Delete the old combined notebook if present — superseded by the 5
    # per-model variants. `unlink(missing_ok=True)` keeps this script
    # idempotent across re-runs.
    old = NOTEBOOK_DIR / "stage3_full_sweep.ipynb"
    if old.exists():
        old.unlink()
        print(f"removed {old}")
