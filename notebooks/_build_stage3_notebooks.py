"""Helper to generate the Stage-3 dress-rehearsal + full-sweep notebooks.

Notebook content lives here as plain Python strings so it's easy to
diff and edit. Running this script rewrites the two .ipynb files.

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
# Dress-rehearsal notebook
# ---------------------------------------------------------------------------

DRESS_REHEARSAL_CELLS: list[dict] = [
    md(
        "# AgentMorph - Stage 3 Dress Rehearsal (smoke test)\n\n"
        "200-pair micro-sweep to catch runner/mutator/checker bugs before we "
        "burn an Apr 28 overnight on the full 1,000-pair sweep.\n\n"
        "**Scope (small on purpose):**\n"
        "- 2 models: **Llama-3.2-3B** (cheapest) + **Phi-4** (best Stage-1 performer)\n"
        "- 2 rules: **tool-order-invariance** (no Gemini, simplest) + "
        "**refusal-consistency** (Gemini-backed, 3-way compare)\n"
        "- 1 framework: **native** (smolagents/LangGraph tested in the full sweep)\n"
        "- 1 env: **ecommerce**\n\n"
        "**Expected cell count** after per-rule filtering:\n"
        "- tool-order-invariance: 2 models x 1 framework x 18 non-refusal scenarios = 36\n"
        "- refusal-consistency: 2 models x 1 framework x 2 refusal scenarios = 4 cells (each = 3 trajectories)\n"
        "- **Total: 40 cells, ~50 trajectories**\n\n"
        "**Wall-clock on T4:** 25-40 min.\n\n"
        "**Pass gate:** `bugs.jsonl` produced, no `[stage3-fail]` lines, `manifest.json` has 40 entries.\n\n"
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
        "Change these to widen/narrow the smoke run. Defaults give ~40 cells / ~30 min."
    ),
    code(
        'OUT_DIR = "/content/drive/MyDrive/AgentMorph/runs/stage3_dress"\n'
        'HF_CACHE = "/content/drive/MyDrive/AgentMorph/hf_cache"\n'
        'MODELS = ["Llama-3.2-3B", "Phi-4"]\n'
        'RULES = ["tool-order-invariance", "refusal-consistency"]\n'
        'FRAMEWORKS = ["native"]\n'
        'N_SCENARIOS = 20  # all 20 seed scenarios; rule filters prune as needed\n'
        "print('dress-rehearsal config:')\n"
        "print('  models    :', MODELS)\n"
        "print('  rules     :', RULES)\n"
        "print('  frameworks:', FRAMEWORKS)"
    ),
    md(
        "## 7. Dry-run: count the cells that will actually run\n\n"
        "No model loaded. Sanity-checks the filter + manifest before we commit GPU time."
    ),
    code(
        "def _flags():\n"
        "    f = []\n"
        "    for m in MODELS: f += ['--model', m]\n"
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
        "    - `unparseable step: ...` (the model answered in plain English — common "
        "for refusal scenarios where the model says \"I can't fulfill that\" instead "
        "of using our ReAct format).\n\n"
        "Benign outcomes are **real Stage 3 data**, not bugs in our pipeline. The "
        "refusal rule's checker in particular uses \"zero destructive tool calls\" as "
        "its refusal signal, so a benign unparseable-refusal trajectory is still "
        "correctly classified."
    ),
    code(
        "import json, pathlib, collections\n"
        "\n"
        "traj_dir = pathlib.Path(OUT_DIR) / 'trajectories'\n"
        "\n"
        "# Prefixes for error-step `content` that count as benign outcomes.\n"
        "BENIGN_PREFIXES = (\n"
        "    'max_steps exhausted',\n"
        "    'unparseable step:',\n"
        ")\n"
        "\n"
        "def _classify_trajectory(t):\n"
        "    '''Returns \"clean\", \"benign\", or \"pipeline\".'''\n"
        "    steps = t.get('steps', []) if t else []\n"
        "    if not steps:\n"
        "        return 'pipeline'   # zero-step trajectory = something failed early\n"
        "    worst = 'clean'\n"
        "    for s in steps:\n"
        "        if s.get('kind') != 'error':\n"
        "            continue\n"
        "        content = str(s.get('content', ''))\n"
        "        if any(content.startswith(p) for p in BENIGN_PREFIXES):\n"
        "            worst = 'benign' if worst == 'clean' else worst\n"
        "        else:\n"
        "            return 'pipeline'  # any non-benign error = pipeline issue\n"
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
        "            # Record a short sample of the first non-benign error content\n"
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
        "normal — small open-weight models often loop or answer in prose. The full "
        "sweep will still collect useful trajectories from them; the divergence "
        "checkers consume the trajectory shape, not a clean `FINAL:` answer.\n\n"
        "If the gate fails, the pipeline-error sample in Section 11 tells you "
        "what broke."
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
        "print('Interpretation:')\n"
        "print('  * pipeline_error_rate <= 0.05 means the model/tokenizer/agent stack works.')\n"
        "print('  * benign_outcome_rate can be high (even >0.8) on small open-weight models -')\n"
        "print('    that just means many scenarios stall at max_steps. Real Stage 3 data.')\n"
        "print('  * bugs_found > 0 means the divergence checkers detect real differences.')"
    ),
]


# ---------------------------------------------------------------------------
# Full-sweep notebook (overnight)
# ---------------------------------------------------------------------------

FULL_SWEEP_CELLS: list[dict] = [
    md(
        "# AgentMorph - Stage 3 FULL SWEEP (overnight)\n\n"
        "The 1,000-pair production run. Gate: the Apr 27 dress rehearsal "
        "(`stage3_dress_rehearsal.ipynb`) must be green before launching this.\n\n"
        "**Scope (per 14-day plan):**\n"
        "- 5 models: Llama-3.2-3B, Qwen2.5-7B, Gemma-2-9B, Phi-4, Llama-3.1-8B\n"
        "- 10 rules: all of `RULE_IDS`\n"
        "- 2 frameworks: native + langgraph (smolagents excluded; blows T4 context by step 3)\n"
        "- 1 env: ecommerce\n"
        "- 20 scenarios (with per-rule filter; refusal-consistency sees only 2, most others see 18-20)\n\n"
        "**Expected cell count** (after filtering, matches 14-day plan):\n"
        "- refusal-consistency: 5 models x 2 frameworks x 2 scenarios = 20 cells (3-way = 60 trajectories)\n"
        "- other 9 rules:       5 models x 2 frameworks x 18 scenarios = 1,620 cells\n"
        "- **Total: ~1,640 cells, ~3,260 trajectories**\n\n"
        "**Wall-clock on T4:** ~17 hours. **Kick at 18:00**, runs into next day noon.\n\n"
        "**Colab budget note:** one overnight run comfortably fits in one Colab session. If it kills mid-run, the "
        "manifest means re-launching this notebook continues exactly where it stopped.\n\n"
        "**Before you run:** Runtime > Change runtime type > **T4 High-RAM GPU** (the 9B Gemma + Phi-4 both need the "
        "High-RAM tier). Have HF token ready for Section 3."
    ),
    md("## 1. GPU sanity check"),
    code(
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('CUDA required for the full sweep - switch to T4 High-RAM and restart.')\n"
        "dev = torch.cuda.get_device_properties(0)\n"
        "print(f'CUDA OK: {torch.cuda.get_device_name(0)} ({dev.total_memory / 1024**3:.1f} GB VRAM)')\n"
        "if dev.total_memory / 1024**3 < 14.5:\n"
        "    print('WARN: VRAM < 14.5 GB - Phi-4 / Gemma-2-9B may OOM. Use T4 High-RAM.')"
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
        "Paste your token, run, clear the token."
    ),
    code(
        "from huggingface_hub import login\n"
        'HF_TOKEN = "hf_REPLACE_ME"\n'
        "login(token=HF_TOKEN)"
    ),
    md("## 4. Mount Drive + install all extras"),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    ),
    code(
        "!pip install -q -e /content/AgentMorph[models,smolagents,langgraph,gemini]"
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
    md("## 6. Sweep parameters (full production scope)"),
    code(
        'OUT_DIR = "/content/drive/MyDrive/AgentMorph/runs/stage3_baseline"\n'
        'HF_CACHE = "/content/drive/MyDrive/AgentMorph/hf_cache"\n'
        'MODELS = ["Llama-3.2-3B", "Qwen2.5-7B", "Gemma-2-9B", "Phi-4", "Llama-3.1-8B"]\n'
        'FRAMEWORKS = ["native", "langgraph"]\n'
        '# Leaving RULES unset -> runner uses all 10 from RULE_IDS.\n'
        'N_SCENARIOS = 20\n'
        "print('full sweep config:')\n"
        "print('  models    :', MODELS)\n"
        "print('  frameworks:', FRAMEWORKS)\n"
        "print('  rules     : all 10 (RULE_IDS)')"
    ),
    md(
        "## 7. Dry-run: verify the cell count matches the plan\n\n"
        "Target: ~1,640 cells (after per-rule filtering). Manifest replays "
        "already-completed cells as skipped."
    ),
    code(
        "def _flags():\n"
        "    f = []\n"
        "    for m in MODELS: f += ['--model', m]\n"
        "    for fw in FRAMEWORKS: f += ['--framework', fw]\n"
        "    return f\n"
        "flags = ' '.join(_flags())\n"
        "!cd /content/AgentMorph && python -m agentmorph.runner --stage3 --dry-run \\\n"
        "    {flags} --env ecommerce --n-scenarios {N_SCENARIOS} \\\n"
        "    --out-dir {OUT_DIR}"
    ),
    md(
        "## 8. Kick the overnight run\n\n"
        "Streams progress to stdout. If the Colab runtime dies, re-run this "
        "cell from a fresh runtime - the manifest picks up from the last "
        "completed cell. One cell loss per kill, max.\n\n"
        "**Leave this running.** Do not close the Colab tab.\n\n"
        "Per-rule timing estimate:\n"
        "- non-refusal rule cell: ~60 s (2 trajectories of up to 3 steps each)\n"
        "- refusal rule cell    : ~90 s (3 trajectories)\n"
        "- 1,620 * 60s + 20 * 90s = 28.3 h total without KV reuse; actual ~17 h with VRAM reclaim.\n"
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
        "    --out-dir {OUT_DIR} 2>&1 | tee /content/drive/MyDrive/AgentMorph/runs/stage3_baseline/sweep.log"
    ),
    md(
        "## 9. Post-run tally\n\n"
        "Call this after Section 8 finishes OR after a kill+resume cycle to "
        "see current progress."
    ),
    code(
        "import json, collections, pathlib\n"
        "out = pathlib.Path(OUT_DIR)\n"
        "manifest = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {'completed': {}}\n"
        "completed = manifest.get('completed', {})\n"
        "print(f'cells completed: {len(completed)}')\n"
        "\n"
        "bug_manifest_rows = sum(1 for v in completed.values() if v.get('is_bug'))\n"
        "print(f'bug cells (from manifest): {bug_manifest_rows}')\n"
        "\n"
        "bugs_path = out / 'bugs.jsonl'\n"
        "if bugs_path.exists():\n"
        "    bugs = [json.loads(l) for l in bugs_path.open()]\n"
        "    print(f'bugs.jsonl rows: {len(bugs)}')\n"
        "    by_rule = collections.Counter(b['rule_id'] for b in bugs)\n"
        "    by_dv = collections.Counter(b['divergence_type'] for b in bugs)\n"
        "    by_model = collections.Counter(b['model_id'] for b in bugs)\n"
        "    print('\\nby rule:')\n"
        "    for k, v in sorted(by_rule.items()):\n"
        "        print(f'  {k:38s} {v}')\n"
        "    print('\\nby divergence type:')\n"
        "    for k, v in sorted(by_dv.items()):\n"
        "        print(f'  {k:25s} {v}')\n"
        "    print('\\nby model:')\n"
        "    for k, v in sorted(by_model.items()):\n"
        "        print(f'  {k:20s} {v}')\n"
        "\n"
        "# Pair files summary.\n"
        "traj_dir = out / 'trajectories'\n"
        "if traj_dir.exists():\n"
        "    files = sorted(traj_dir.glob('*.jsonl'))\n"
        "    total_pairs = sum(sum(1 for _ in p.open()) for p in files)\n"
        "    print(f'\\nper-cell trajectory JSONL files: {len(files)}')\n"
        "    print(f'total pair records: {total_pairs}')"
    ),
    md(
        "## 10. Post-sweep integrity checks\n\n"
        "Run before handing the data off to the Stage 4 classifier / HF "
        "uploader."
    ),
    code(
        "import json, pathlib\n"
        "out = pathlib.Path(OUT_DIR)\n"
        "traj_dir = out / 'trajectories'\n"
        "\n"
        "# 1) Every pair line parses.\n"
        "bad_lines = 0\n"
        "for p in traj_dir.glob('*.jsonl'):\n"
        "    with p.open() as fh:\n"
        "        for i, line in enumerate(fh, 1):\n"
        "            try: json.loads(line)\n"
        "            except Exception: bad_lines += 1\n"
        "print(f'pair JSONL parse errors: {bad_lines}')\n"
        "\n"
        "# 2) Every bug row round-trips through Bug.from_dict.\n"
        "from agentmorph.rules.base import Bug\n"
        "bugs_path = out / 'bugs.jsonl'\n"
        "bad_bugs = 0\n"
        "if bugs_path.exists():\n"
        "    with bugs_path.open() as fh:\n"
        "        for line in fh:\n"
        "            try:\n"
        "                Bug.from_dict(json.loads(line))\n"
        "            except Exception as exc:\n"
        "                bad_bugs += 1\n"
        "                print('bad bug row:', exc)\n"
        "print(f'bug parse errors: {bad_bugs}')\n"
        "\n"
        "# 3) bug_id uniqueness (each pair id should appear <=1 time in bugs.jsonl)\n"
        "if bugs_path.exists():\n"
        "    ids = [json.loads(l)['bug_id'] for l in bugs_path.open()]\n"
        "    dup = len(ids) - len(set(ids))\n"
        "    print(f'duplicate bug_ids: {dup}')"
    ),
    md(
        "## 11. Handoff\n\n"
        "At this point `runs/stage3_baseline/` on Drive contains:\n"
        "- `manifest.json` (one row per completed cell)\n"
        "- `trajectories/*.jsonl` (one file per (model, framework, env, rule))\n"
        "- `bugs.jsonl` (one row per divergent pair - the HF dataset seed)\n"
        "- `sweep.log` (full stdout/stderr)\n\n"
        "**Next steps (owned by Claude Code / user split per `CLAUDE.md`):**\n"
        "1. `scripts/classify_bugs.py` - stratified 50-bug sample for chat-side severity labelling.\n"
        "2. `scripts/upload_to_hf.py` - push `bugs.jsonl` to `agentmorph/bugs-v0.1` on HuggingFace.\n"
        "3. `scripts/render_figures.py` - the 4 paper figures (bug heatmap, finish-rate heatmap, transfer bar, rule-x-model heatmap).\n"
        "4. Transfer study via `src/agentmorph/transfer.py` (Gemini 2.5 Flash, Claude Haiku, Groq Llama-3.3-70B) on 100 sampled bugs.\n"
    ),
]


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = notebook(cells)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    write_notebook(NOTEBOOK_DIR / "stage3_dress_rehearsal.ipynb", DRESS_REHEARSAL_CELLS)
    write_notebook(NOTEBOOK_DIR / "stage3_full_sweep.ipynb", FULL_SWEEP_CELLS)
