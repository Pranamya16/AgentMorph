"""Rule 1 — tool-order-invariance.

MAST category: prompt sensitivity / ordering effects. An agent should choose
tools by *capability*, not by the position of a tool's description in the
system prompt. If shuffling the tool list produces a different tool-call
set, that's a robustness bug.

Invariant:
    Permuting the order of tools in the registry should not change the
    multiset of tool calls the agent makes, or its final answer.

Divergence classification (via `classify_simple_divergence`):
    REORDER_ONLY (same tool multiset, different sequence) → equivalent;
        logged as a stats-only divergence, not a bug.
    TOOL_SET_DIFFERS, ANSWER_DIFFERS, COMPLETION_DIFFERS → bugs.

Scenario filter:
    Only meaningful for scenarios that produced ≥ 2 tool calls in Stage 1
    baselines. Single-tool scenarios have nothing to re-order, so we skip.
    The Stage 3 runner applies this filter — the rule itself is permissive.

See `AgentMorph_Stage2_Runbook.md` §4 rule 1 for the full spec.
"""

from __future__ import annotations

from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.rules._shared import (
    classify_simple_divergence,
    reorder_registry,
)
from agentmorph.rules.base import EquivalenceResult, MutationResult, Mutator
from agentmorph.tools.base import ToolRegistry


class _ToolOrderInvarianceMutator(Mutator):
    rule_id = "tool-order-invariance"

    def apply(
        self,
        scenario: Scenario,
        registry: ToolRegistry,
        *,
        seed: int = 0,
    ) -> MutationResult:
        new_registry, permutation = reorder_registry(
            registry, seed=seed, scenario_id=scenario.id
        )
        return MutationResult(
            scenario=scenario,
            registry=new_registry,
            metadata={
                "rule_id": self.rule_id,
                "seed": seed,
                "original_order": [t.name for t in registry],
                "permutation": permutation,
            },
        )


class _ToolOrderInvarianceChecker:
    rule_id = "tool-order-invariance"

    def compare(
        self,
        original: Any,
        mutated: Any,
        *,
        mutation_metadata: dict[str, Any] | None = None,
    ) -> EquivalenceResult:
        # Reorder-only differences are not bugs for this rule — they're
        # exactly what we're inviting by shuffling the tool list. Log as
        # REORDER_ONLY but treat as equivalent.
        return classify_simple_divergence(
            original,
            mutated,
            sequence_reorder_is_equivalent=True,
        )


MUTATOR = _ToolOrderInvarianceMutator()
CHECKER = _ToolOrderInvarianceChecker()
