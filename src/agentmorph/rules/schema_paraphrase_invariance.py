"""Rule 2 — schema-paraphrase-invariance (Gemini-backed).

MAST category: schema sensitivity. Tool selection should depend on a
tool's *capability* (name + parameters + semantics), not on the specific
English wording of its `description` field.

Invariant:
    Paraphrasing every tool's description (one Gemini call per tool,
    name + parameters + func unchanged) should not change which tool the
    agent picks or what arguments it emits.

Divergence classification (via `classify_simple_divergence`):
    NONE → same (tool name, args) multiset + same final answer.
    TOOL_SET_DIFFERS → the agent picked a different tool, or emitted
        different argument VALUES (the bug we want to catch).
    ANSWER_DIFFERS, COMPLETION_DIFFERS → bugs.

Cache requirements:
    30 tools × 1 variant = 30 paraphrase cache entries. Populated offline
    by `scripts/generate_paraphrases.py`. The mutator calls
    `paraphrase(..., offline=True)` and raises `ParaphraseCacheMiss`
    if the cache is empty — do NOT let the Stage 3 sweep make live API
    calls.

Paraphrase instruction (Gemini system prompt):
    "Rewrite the tool description below in different words but
    preserving exactly the same meaning. Keep it under 200 characters.
    Do not add or remove information. Return only the rewritten
    description, no preamble."

See `AgentMorph_Stage2_Runbook.md` §4 rule 2 for the full spec.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.paraphrase import ParaphraseCache, paraphrase
from agentmorph.rules._shared import (
    SystemPromptMutator,
    classify_simple_divergence,
)
from agentmorph.rules.base import EquivalenceResult
from agentmorph.tools.base import Tool, ToolRegistry


PARAPHRASE_INSTRUCTION = (
    "Rewrite the tool description below in different words but preserving "
    "exactly the same meaning. Keep it under 200 characters. Do not add or "
    "remove information. Return only the rewritten description, no preamble."
)


class _SchemaParaphraseInvarianceMutator(SystemPromptMutator):
    rule_id = "schema-paraphrase-invariance"

    def __init__(self, cache: ParaphraseCache | None = None) -> None:
        # Lazily construct the default cache — keeps import path dependency-free.
        self._cache = cache

    def _transform_tool(self, tool: Tool) -> Tool:
        cache = self._cache if self._cache is not None else ParaphraseCache()
        new_description = paraphrase(
            tool.description,
            rule_id=self.rule_id,
            instruction=PARAPHRASE_INSTRUCTION,
            variant_idx=self._variant_idx_for_mutate,
            cache=cache,
            offline=self._offline_for_mutate,
        )
        # Guard against Gemini returning an empty or identical string —
        # fall back to the original description rather than emit a
        # degenerate mutation.
        if not new_description or new_description.strip() == tool.description.strip():
            return tool
        return dataclasses.replace(tool, description=new_description)

    def apply(
        self,
        scenario: Scenario,
        registry: ToolRegistry,
        *,
        seed: int = 0,
        offline: bool = True,
    ):
        # Stash per-call knobs so _transform_tool can read them. The
        # SystemPromptMutator base doesn't forward these kwargs.
        self._variant_idx_for_mutate = 0  # rule 2 uses exactly 1 variant per tool
        self._offline_for_mutate = offline
        try:
            return super().apply(scenario, registry, seed=seed)
        finally:
            del self._variant_idx_for_mutate
            del self._offline_for_mutate

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
        base["paraphrases"] = {
            t.name: {
                "original": registry.get(t.name).description,
                "paraphrased": t.description,
                "changed": t.description != registry.get(t.name).description,
            }
            for t in new_registry
            if t.name in registry.tools
        }
        return base


class _SchemaParaphraseInvarianceChecker:
    rule_id = "schema-paraphrase-invariance"

    def compare(
        self,
        original: Any,
        mutated: Any,
        *,
        mutation_metadata: dict[str, Any] | None = None,
    ) -> EquivalenceResult:
        # Rule 2 cares about exact arg-value match — frozenset(args.items())
        # already captures that. `classify_simple_divergence` uses it.
        return classify_simple_divergence(
            original, mutated, sequence_reorder_is_equivalent=True
        )


MUTATOR = _SchemaParaphraseInvarianceMutator()
CHECKER = _SchemaParaphraseInvarianceChecker()
