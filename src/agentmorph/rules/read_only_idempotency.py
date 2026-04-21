"""Rule 4 — read-only-idempotency.

MAST category: side-effect discipline. Read-only operations should never
mutate state. Nudging the agent to use *more* read-only tools (e.g.
`view_cart`, `list_categories`) mid-trajectory must not change the final
state — cart items, orders placed, addresses added, etc.

Invariant:
    For a given scenario, running it twice — once as-is, once with a
    prompt nudge that asks the agent to also call a couple of read-only
    tools for context — should produce IDENTICAL post-trajectory state.

Divergence classification:
    `SIDE_EFFECTS_DIFFER` → the state snapshots disagree. Bug.
    `NONE` → state snapshots match. (The tool-call sequences usually
             differ because the nudged run makes extra read-only calls,
             and that's exactly what we're testing is harmless.)

Runner requirement (this is the only rule that needs a runner extension):
    The Stage 3 runner must call `snapshot_state(bundle)` before AND
    after each trajectory, then stash the result in
    `trajectory.metadata["state_delta"] = {"pre": ..., "post": ...}`.
    Without this, the checker has no state to compare and will emit a
    warning + treat the pair as equivalent. The feature is gated by
    the runner's `--capture-state` flag; see §6 of the runbook.

Scenario filter:
    The runner should skip scenarios where no read-only tool is in the
    scenario's expected-tools list — e.g. pure profile-update scenarios
    where there's nothing for the nudge to exploit. Stratification is
    the runner's responsibility; the mutator itself is permissive.

See `AgentMorph_Stage2_Runbook.md` §4 rule 4 for the full spec.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agentmorph.environments.base import Scenario
from agentmorph.rules._shared import ScenarioPromptMutator, state_delta
from agentmorph.rules.base import DivergenceType, EquivalenceResult
from agentmorph.tools.base import ToolRegistry


# The read-only tools we nudge the agent toward. All present in the
# 30-tool ecommerce suite and all marked `read_only=True`.
READ_ONLY_NUDGE_TOOLS: tuple[str, ...] = ("view_cart", "list_categories")

# Appended to the user prompt to encourage read-only tool use without
# otherwise changing the task.
NUDGE_SUFFIX: str = (
    "\n\nBefore giving your final answer, also call `view_cart` and "
    "`list_categories` once to confirm the context. These are read-only "
    "and won't affect anything else."
)


class _ReadOnlyIdempotencyMutator(ScenarioPromptMutator):
    rule_id = "read-only-idempotency"

    def _mutate_prompt(
        self, prompt: str, *, seed: int, scenario: Scenario
    ) -> tuple[str, dict[str, Any]]:
        new_prompt = prompt + NUDGE_SUFFIX
        return new_prompt, {
            "injected_reads": list(READ_ONLY_NUDGE_TOOLS),
            "nudge_style": "append_suffix",
        }


class _ReadOnlyIdempotencyChecker:
    rule_id = "read-only-idempotency"

    def compare(
        self,
        original: Any,
        mutated: Any,
        *,
        mutation_metadata: dict[str, Any] | None = None,
    ) -> EquivalenceResult:
        orig_state = _extract_state(original)
        mut_state = _extract_state(mutated)

        if orig_state is None or mut_state is None:
            # Runner didn't capture state — can't judge. Return equivalent
            # with an explicit note so the Stage 3 aggregator can flag
            # this pair as "unclassified" rather than a real bug.
            return EquivalenceResult(
                is_equivalent=True,
                divergence_type=DivergenceType.NONE,
                details=(
                    "state not captured — run with --capture-state for "
                    "meaningful rule-4 evaluation"
                ),
            )

        delta = state_delta(orig_state, mut_state)
        if not delta:
            return EquivalenceResult(
                is_equivalent=True,
                divergence_type=DivergenceType.NONE,
                details="identical post-trajectory state",
            )

        # Which fields diverged? Summarise up to 3 of them for readability.
        summary_fields = sorted(delta.keys())[:3]
        details = "state differs on: " + ", ".join(summary_fields)
        if len(delta) > 3:
            details += f" (+{len(delta) - 3} more)"
        return EquivalenceResult(
            is_equivalent=False,
            divergence_type=DivergenceType.SIDE_EFFECTS_DIFFER,
            details=details,
        )


def _extract_state(trajectory: Any) -> dict[str, Any] | None:
    """Pull the `post` state snapshot out of a trajectory's metadata.

    Returns None if the runner didn't capture state — which is a clear
    signal to the checker, not an error.
    """
    metadata = (
        trajectory.get("metadata")
        if isinstance(trajectory, dict)
        else getattr(trajectory, "metadata", None)
    )
    if not metadata:
        return None
    delta = metadata.get("state_delta") if isinstance(metadata, dict) else None
    if not delta:
        return None
    return delta.get("post")


MUTATOR = _ReadOnlyIdempotencyMutator()
CHECKER = _ReadOnlyIdempotencyChecker()
