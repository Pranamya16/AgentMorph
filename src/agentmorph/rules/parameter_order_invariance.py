"""Rule 7 — parameter-order-invariance.

MAST category: schema-ordering sensitivity. When a tool has multiple
parameters, the order in which they appear in the JSON-Schema
`properties` dict should not influence the model's argument-value
assignment — JSON dict semantics don't depend on key order.

Invariant:
    Reordering the keys of each tool's `parameters.properties` dict
    should not change the (name, args) multiset of tool calls the agent
    emits. Argument VALUES (not positions) are what matters.

Divergence classification (via `classify_simple_divergence`):
    NONE → agent called the same tool with the same VALUES despite the
        shuffled schema.
    TOOL_SET_DIFFERS / ANSWER_DIFFERS / COMPLETION_DIFFERS → bugs.

Scenario filter: any scenario where the agent emits tool calls with ≥2
args qualifies. Single-arg scenarios (e.g. `check_stock(product_id)`)
are no-ops for this rule; the Stage 3 runner logs but doesn't drop them.

See `AgentMorph_Stage2_Runbook.md` §4 rule 7 for the full spec.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.rules._shared import (
    SystemPromptMutator,
    classify_simple_divergence,
)
from agentmorph.rules.base import EquivalenceResult
from agentmorph.tools.base import Tool, ToolRegistry


def _shuffle_properties(
    params: dict[str, Any], *, seed: int, tool_name: str
) -> dict[str, Any]:
    """Return a deep-copied params dict with `properties` keys in a permuted
    order. Deterministic under (seed, tool_name)."""
    props = params.get("properties", {})
    if len(props) < 2:
        return params  # nothing meaningful to shuffle
    rng = random.Random(hash((seed, tool_name)) & 0xFFFFFFFF)
    keys = list(props.keys())
    permuted = rng.sample(keys, len(keys))
    new_props = {k: props[k] for k in permuted}
    new_params = {**params, "properties": new_props}
    return new_params


class _ParameterOrderInvarianceMutator(SystemPromptMutator):
    rule_id = "parameter-order-invariance"

    def _transform_tool(self, tool: Tool) -> Tool:
        # Seed via the tool name so different tools get independent permutations
        # but each tool is deterministic for a given seed.
        new_params = _shuffle_properties(
            tool.parameters, seed=self._seed_for_mutate, tool_name=tool.name
        )
        if new_params is tool.parameters:
            return tool
        return dataclasses.replace(tool, parameters=new_params)

    def apply(
        self,
        scenario: Scenario,
        registry: ToolRegistry,
        *,
        seed: int = 0,
    ):
        # SystemPromptMutator.apply calls _transform_tool for each tool;
        # stash the seed so the transform can read it (SystemPromptMutator
        # doesn't pass seed through).
        self._seed_for_mutate = seed
        try:
            return super().apply(scenario, registry, seed=seed)
        finally:
            # Clean up — keep the mutator reusable across scenarios.
            del self._seed_for_mutate

    def _metadata(
        self,
        *,
        registry: ToolRegistry,
        new_registry: ToolRegistry,
        seed: int,
        scenario: Scenario,
    ) -> dict[str, Any]:
        base = super()._metadata(
            registry=registry, new_registry=new_registry, seed=seed, scenario=scenario
        )
        # Record the original key order for each tool so the checker can
        # confirm that the emitted args use the right VALUES (dict equality
        # ignores order anyway, but the metadata is useful for diagnostics).
        base["original_property_orders"] = {
            t.name: list(t.parameters.get("properties", {}).keys()) for t in registry
        }
        base["mutated_property_orders"] = {
            t.name: list(t.parameters.get("properties", {}).keys()) for t in new_registry
        }
        base["tools_shuffled"] = [
            t.name
            for t in new_registry
            if base["original_property_orders"].get(t.name)
            != base["mutated_property_orders"].get(t.name)
        ]
        return base


class _ParameterOrderInvarianceChecker:
    rule_id = "parameter-order-invariance"

    def compare(
        self,
        original: Any,
        mutated: Any,
        *,
        mutation_metadata: dict[str, Any] | None = None,
    ) -> EquivalenceResult:
        # tool_call_set_equal already uses frozenset(args.items()), so dict
        # key order doesn't matter. Regular classify_simple_divergence is
        # exactly what we want.
        return classify_simple_divergence(
            original, mutated, sequence_reorder_is_equivalent=True
        )


MUTATOR = _ParameterOrderInvarianceMutator()
CHECKER = _ParameterOrderInvarianceChecker()
