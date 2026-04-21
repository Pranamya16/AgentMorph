"""Rule 3 — synonym-robustness (Gemini-backed).

MAST category: input-language sensitivity. "Find me a kettle under $50"
and "Locate an affordable kettle (under fifty dollars)" should trigger the
same behaviour — the agent's tool calls should be driven by intent, not
by the exact words the user typed.

Invariant:
    Paraphrasing the user prompt (Gemini, preserving exact numbers and
    names) should produce the same (tool name, args) multiset and a
    semantically-equivalent final answer. Argument VALUES may differ
    slightly — the checker uses `classify_simple_divergence` which
    allows REORDER_ONLY as equivalent.

Divergence classification:
    NONE → same tool set + same answer.
    ANSWER_DIFFERS, TOOL_SET_DIFFERS, COMPLETION_DIFFERS → bugs.

Cache requirements:
    20 scenarios × 1 variant = 20 paraphrase cache entries. Populated
    offline by `scripts/generate_paraphrases.py`.

Paraphrase instruction (Gemini system prompt):
    "Rewrite the following user request using different words and
    synonyms, preserving exactly the same meaning and the same specific
    numbers and names. Return only the rewritten request."

See `AgentMorph_Stage2_Runbook.md` §4 rule 3 for the full spec.
"""

from __future__ import annotations

from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.paraphrase import ParaphraseCache, paraphrase
from agentmorph.rules._shared import (
    ScenarioPromptMutator,
    classify_simple_divergence,
)
from agentmorph.rules.base import EquivalenceResult


PARAPHRASE_INSTRUCTION = (
    "Rewrite the following user request using different words and synonyms, "
    "preserving exactly the same meaning and the same specific numbers and "
    "names. Return only the rewritten request, no preamble."
)


class _SynonymRobustnessMutator(ScenarioPromptMutator):
    rule_id = "synonym-robustness"

    def __init__(self, cache: ParaphraseCache | None = None) -> None:
        self._cache = cache

    def _mutate_prompt(
        self,
        prompt: str,
        *,
        seed: int,
        scenario: Scenario,
    ) -> tuple[str, dict[str, Any]]:
        cache = self._cache if self._cache is not None else ParaphraseCache()
        # Rule 3 uses 1 variant per scenario by default; Stage 3 can iterate
        # seeds to request more variants by bumping variant_idx.
        variant_idx = 0
        # Check for an `offline` toggle at call time — default True so the
        # Stage 3 sweep never makes a live API call unintentionally.
        offline = getattr(self, "_offline_for_mutate", True)
        new_prompt = paraphrase(
            prompt,
            rule_id=self.rule_id,
            instruction=PARAPHRASE_INSTRUCTION,
            variant_idx=variant_idx,
            cache=cache,
            offline=offline,
        )
        if not new_prompt or new_prompt.strip() == prompt.strip():
            # Degenerate paraphrase — fall back to original. Logged in metadata.
            return prompt, {
                "paraphrased": False,
                "variant_idx": variant_idx,
                "reason": "degenerate_output",
            }
        return new_prompt, {
            "paraphrased": True,
            "variant_idx": variant_idx,
            "paraphrase": new_prompt,
        }

    def apply(
        self,
        scenario: Scenario,
        registry,
        *,
        seed: int = 0,
        offline: bool = True,
    ):
        self._offline_for_mutate = offline
        try:
            return super().apply(scenario, registry, seed=seed)
        finally:
            del self._offline_for_mutate


class _SynonymRobustnessChecker:
    rule_id = "synonym-robustness"

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


MUTATOR = _SynonymRobustnessMutator()
CHECKER = _SynonymRobustnessChecker()
