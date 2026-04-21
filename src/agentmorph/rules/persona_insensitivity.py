"""Rule 9 — persona-insensitivity.

MAST category: role-framing sensitivity. Prepending a persona statement to
the user prompt should change the agent's *tone*, never which tools it
picks or what the final answer is *about*.

Invariant:
    Same tool-call multiset + semantically equivalent final answer
    (wording may differ stylistically due to the persona instruction).

Divergence classification: identical to rule 1's — REORDER_ONLY is
equivalent, TOOL_SET_DIFFERS / ANSWER_DIFFERS / COMPLETION_DIFFERS are
bugs.

Seeded variant choice — three personas, picked via
`idx = hash((seed, scenario_id)) % 3`. Stage 3 can run all three by
iterating seeds.

See `AgentMorph_Stage2_Runbook.md` §4 rule 9 for the full spec.
"""

from __future__ import annotations

from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.rules._shared import (
    ScenarioPromptMutator,
    classify_simple_divergence,
)
from agentmorph.rules.base import EquivalenceResult


PERSONAS: tuple[str, ...] = (
    "You are a helpful assistant who prefers concise, professional answers.",
    "You are an enthusiastic customer-service bot. Always be polite and direct.",
    "You are a careful assistant who double-checks every action before responding.",
)


class _PersonaInsensitivityMutator(ScenarioPromptMutator):
    rule_id = "persona-insensitivity"

    def _mutate_prompt(
        self, prompt: str, *, seed: int, scenario: Scenario
    ) -> tuple[str, dict[str, Any]]:
        idx = hash((seed, scenario.id)) % len(PERSONAS)
        persona = PERSONAS[idx]
        new_prompt = f"{persona}\n\n{prompt}"
        return new_prompt, {"persona_idx": idx, "persona_text": persona}


class _PersonaInsensitivityChecker:
    rule_id = "persona-insensitivity"

    def compare(
        self,
        original: Any,
        mutated: Any,
        *,
        mutation_metadata: dict[str, Any] | None = None,
    ) -> EquivalenceResult:
        return classify_simple_divergence(
            original, mutated, sequence_reorder_is_equivalent=True
        )


MUTATOR = _PersonaInsensitivityMutator()
CHECKER = _PersonaInsensitivityChecker()
