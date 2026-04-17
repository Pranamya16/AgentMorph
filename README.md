# AgentMorph

Metamorphic testing for LLM agents. Mutate agent scenarios under semantics-preserving rules, flag behavior divergence as bugs.

Status: **Stage 1 — Foundation.** See [CLAUDE.md](CLAUDE.md) and `AgentMorph_Execution_Plan.docx` for the execution plan.

## Install (development)

```bash
pip install -e ".[all]"
```

## Colab quickstart

Open `notebooks/colab_setup.py` in a Colab cell, run it, then:

```python
from agentmorph.runner import run_baseline
run_baseline(model="Llama-3.2-3B", framework="smolagents", environment="ecommerce", n_scenarios=5)
```

## Layout

- `src/agentmorph/models.py` — 4-bit HF model loader + registry.
- `src/agentmorph/tools/` — unified tool abstraction + synthetic e-commerce suite.
- `src/agentmorph/agents/` — smolagents + LangGraph wrappers sharing one loaded model.
- `src/agentmorph/environments/` — AgentDojo + synthetic e-commerce env adapters.
- `src/agentmorph/trajectories.py` — JSONL trajectory schema + logger.
- `src/agentmorph/runner.py` — resumable baseline runner.
