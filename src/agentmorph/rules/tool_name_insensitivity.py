"""Rule 6 — tool-name-insensitivity.

MAST category: symbolic arbitrariness. A tool's NAME is a surface form; the
agent should pick tools by capability (description + parameters), not by
the specific token sequence in the name.

Invariant:
    Renaming every tool in the registry (1-to-1, semantic-preserving map)
    should not change which tool the agent picks. After applying the
    inverse rename to the mutated trajectory, the tool-call multisets of
    original and mutated runs should match.

Divergence classification (via `classify_simple_divergence` with
`name_mapping` set to the inverse rename map):
    NONE → the mutated trajectory's renamed tool calls reduce to the same
        multiset as the original.
    TOOL_SET_DIFFERS → the agent picked a different tool even after inverse
        rename (the bug we want to catch).
    ANSWER_DIFFERS, COMPLETION_DIFFERS → bugs.

Scenario filter: any scenario with ≥1 tool call qualifies. The Stage 3
runner logs scenarios where the mutated agent never emitted ANY of the
renamed tool names (which would be "the model refused to use the
renamed tools at all" — interesting signal).

See `AgentMorph_Stage2_Runbook.md` §4 rule 6 for the full spec.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.rules._name_map import INVERSE_NAME_MAP, NAME_MAP
from agentmorph.rules._shared import (
    SystemPromptMutator,
    classify_simple_divergence,
)
from agentmorph.rules.base import EquivalenceResult
from agentmorph.tools.base import Tool, ToolRegistry


class _ToolNameInsensitivityMutator(SystemPromptMutator):
    rule_id = "tool-name-insensitivity"

    def _transform_tool(self, tool: Tool) -> Tool:
        new_name = NAME_MAP.get(tool.name, tool.name)
        if new_name == tool.name:
            return tool
        return dataclasses.replace(tool, name=new_name)

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
        base["name_map"] = dict(NAME_MAP)
        base["inverse_name_map"] = dict(INVERSE_NAME_MAP)
        return base


class _ToolNameInsensitivityChecker:
    rule_id = "tool-name-insensitivity"

    def compare(
        self,
        original: Any,
        mutated: Any,
        *,
        mutation_metadata: dict[str, Any] | None = None,
    ) -> EquivalenceResult:
        # The mutated trajectory has renamed tool-call events — map them
        # back to original names before comparing multisets.
        name_mapping = (
            (mutation_metadata or {}).get("inverse_name_map") or INVERSE_NAME_MAP
        )
        # We need to apply name_mapping only to the MUTATED trajectory.
        # classify_simple_divergence applies it uniformly to both sides —
        # which is fine here because original names pass through unchanged
        # (they're not in INVERSE_NAME_MAP keys).
        return classify_simple_divergence(
            original,
            mutated,
            name_mapping=name_mapping,
            sequence_reorder_is_equivalent=True,
        )


MUTATOR = _ToolNameInsensitivityMutator()
CHECKER = _ToolNameInsensitivityChecker()
