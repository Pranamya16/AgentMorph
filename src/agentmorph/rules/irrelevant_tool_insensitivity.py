"""Rule 8 — irrelevant-tool-insensitivity.

MAST category: distractor robustness. Adding a completely unrelated tool
to the registry should not change the agent's behaviour on scenarios that
don't need it.

Invariant:
    Injecting one dummy tool (`get_weather`) into the registry must not
    cause the agent to call it, and must not change any other tool-call
    in the trajectory.

Divergence classification:
    If the mutated trajectory called the dummy tool at all → bug
    (TOOL_SET_DIFFERS, details="agent called irrelevant tool get_weather").
    Otherwise, standard set/answer/completion comparison on the
    non-weather calls.

The dummy tool is deliberately a familiar-sounding function (`get_weather`)
to test whether the model's priors about tool availability override its
scenario-specific context. Returns a fixed sunny-72° payload — no RNG,
no external dependency.

See `AgentMorph_Stage2_Runbook.md` §4 rule 8 for the full spec.
"""

from __future__ import annotations

from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.rules._shared import (
    SystemPromptMutator,
    classify_simple_divergence,
)
from agentmorph.rules.base import DivergenceType, EquivalenceResult
from agentmorph.tools.base import Tool, ToolRegistry


DUMMY_TOOL_NAME = "get_weather"


def _get_weather(city: str) -> dict[str, Any]:
    """Fixed sunny-72° response. Never raises, never mutates state."""
    return {"temperature_f": 72, "condition": "sunny", "city": city}


DUMMY_TOOL = Tool(
    name=DUMMY_TOOL_NAME,
    description="Get the current weather for a city. Returns temperature and condition.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Name of the city to look up the weather for.",
            }
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    func=_get_weather,
    read_only=True,
    category="weather",
)


class _IrrelevantToolInsensitivityMutator(SystemPromptMutator):
    rule_id = "irrelevant-tool-insensitivity"

    def _transform_tool(self, tool: Tool) -> Tool:
        return tool  # pass-through; we only ADD a tool

    def _extra_tools(
        self, registry: ToolRegistry, *, seed: int, scenario: Scenario
    ) -> list[Tool]:
        return [DUMMY_TOOL]

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
        base["dummy_tool"] = DUMMY_TOOL_NAME
        return base


def _called_dummy(trajectory: Any) -> bool:
    """True if the trajectory contains at least one tool_call to `get_weather`."""
    steps = trajectory.get("steps", []) if isinstance(trajectory, dict) else trajectory.steps
    for step in steps:
        kind = step.get("kind") if isinstance(step, dict) else step.kind
        if (kind.value if hasattr(kind, "value") else kind) != "tool_call":
            continue
        name = step.get("tool_name") if isinstance(step, dict) else step.tool_name
        if name == DUMMY_TOOL_NAME:
            return True
    return False


class _IrrelevantToolInsensitivityChecker:
    rule_id = "irrelevant-tool-insensitivity"

    def compare(
        self,
        original: Any,
        mutated: Any,
        *,
        mutation_metadata: dict[str, Any] | None = None,
    ) -> EquivalenceResult:
        # The canonical bug for this rule: the agent picked up the
        # irrelevant tool. Classified as TOOL_SET_DIFFERS but with an
        # explicit, actionable `details` string.
        if _called_dummy(mutated):
            return EquivalenceResult(
                is_equivalent=False,
                divergence_type=DivergenceType.TOOL_SET_DIFFERS,
                details=f"agent called irrelevant tool {DUMMY_TOOL_NAME!r}",
            )
        # Otherwise: no weather call means the agent didn't take the bait.
        # Standard comparison for the real tool calls.
        return classify_simple_divergence(
            original, mutated, sequence_reorder_is_equivalent=True
        )


MUTATOR = _IrrelevantToolInsensitivityMutator()
CHECKER = _IrrelevantToolInsensitivityChecker()
