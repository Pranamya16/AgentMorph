"""Rule 10 — distractor-text-insensitivity.

MAST category: irrelevant-context filtering. The agent should ignore
off-topic small-talk in the user prompt.

Invariant:
    Prepending an unrelated sentence (off-topic, no command) followed by
    "Anyway, " before the real request should not change tool-call
    sequence or final answer.

Divergence classification: same decision tree as rule 9 — REORDER_ONLY
equivalent, TOOL_SET_DIFFERS / ANSWER_DIFFERS / COMPLETION_DIFFERS bugs.

Seeded variant choice — three distractors, picked via
`idx = hash((seed, scenario_id)) % 3`.

See `AgentMorph_Stage2_Runbook.md` §4 rule 10 for the full spec.
"""

from __future__ import annotations

from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.rules._shared import (
    ScenarioPromptMutator,
    classify_simple_divergence,
)
from agentmorph.rules.base import EquivalenceResult


DISTRACTORS: tuple[str, ...] = (
    "I love sunny days and going for long walks.",
    "The weather today has been unusually warm for this time of year.",
    "By the way, I just finished reading a fascinating book about rainforests.",
)


class _DistractorTextInsensitivityMutator(ScenarioPromptMutator):
    rule_id = "distractor-text-insensitivity"

    def _mutate_prompt(
        self, prompt: str, *, seed: int, scenario: Scenario
    ) -> tuple[str, dict[str, Any]]:
        idx = hash((seed, scenario.id)) % len(DISTRACTORS)
        distractor = DISTRACTORS[idx]
        # "Anyway, " is the pivot that returns the user to the real request.
        new_prompt = f"{distractor}\n\nAnyway, {prompt}"
        return new_prompt, {
            "distractor_idx": idx,
            "distractor_text": distractor,
        }


class _DistractorTextInsensitivityChecker:
    rule_id = "distractor-text-insensitivity"

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


MUTATOR = _DistractorTextInsensitivityMutator()
CHECKER = _DistractorTextInsensitivityChecker()
