"""Base protocols + types for Stage 2 metamorphic rules.

A rule is two paired objects:

  * `Mutator.apply(scenario, registry) -> MutationResult`
      Produces a new scenario (and a possibly-different tool registry) whose
      agent behaviour should be equivalent to the original under the rule's
      stated invariant. The mutator is pure: same seed + same inputs → same
      output.

  * `EquivalenceChecker.compare(original, mutated) -> EquivalenceResult`
      Given the two resulting trajectories, decides whether the pair is
      equivalent or diverged. A divergence with `divergence_type != None` is
      a bug candidate; Stage 3 aggregates those into the HF dataset.

The `Bug` dataclass is the HF-serialisable record that ships in the final
dataset. Fields are aligned with the project intro doc's `{scenario,
mutation, diverged_trajectories}` schema.

Design invariants:

  * Pure Python, no torch, no framework dependencies. Importable without
    the `[models]` extra, consistent with the rest of the package.
  * `MutationResult.metadata` is the spot to stash rule-specific data that
    the equivalence checker needs downstream (e.g. the expected
    read-only tool set, or the original tool-name mapping). Nothing else
    reads `metadata` — keep it per-rule.
  * `Trajectory` is the existing `agentmorph.trajectories.Trajectory`;
    we import lazily only inside `compare()` to keep this module
    torch-free on module load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover — typing only
    from agentmorph.environments.base import Scenario
    from agentmorph.tools.base import ToolRegistry
    from agentmorph.trajectories import Trajectory


# -- Divergence taxonomy -----------------------------------------------------


class DivergenceType(str, Enum):
    """How the mutated trajectory differs from the original, in rising severity.

    Kept deliberately small — `chat` extends MAST as needed for Stage 4.
    Values chosen so string sort = severity-adjacent grouping.
    """

    # No meaningful difference; pair is equivalent under the rule's invariant.
    NONE = "none"

    # Same final answer + same tool-call set but in a different order.
    # Usually benign; still recorded so Stage 3 can compute ordering-fragility
    # statistics per rule/model.
    REORDER_ONLY = "reorder_only"

    # Different tool-call set (different tool names or parameters), but the
    # final answer is still consistent with the user intent.
    TOOL_SET_DIFFERS = "tool_set_differs"

    # Final answer differs semantically.
    ANSWER_DIFFERS = "answer_differs"

    # Side effects (cart state, order placed, etc.) diverge between the two
    # runs — the most severe divergence type.
    SIDE_EFFECTS_DIFFER = "side_effects_differ"

    # One trajectory finished, the other errored or hit max_steps.
    COMPLETION_DIFFERS = "completion_differs"

    # One trajectory refused the user request; the other complied.
    # Central to rule 5 (refusal-consistency).
    REFUSAL_DIFFERS = "refusal_differs"


class Severity(str, Enum):
    """Bug severity for classification in Stage 3.

    Chat assigns these on a 50-bug stratified sample; the aggregator then
    propagates via per-rule heuristics. Severity is orthogonal to
    `DivergenceType` — a `REORDER_ONLY` can still be `HIGH` in a cart flow.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNCLASSIFIED = "unclassified"


# -- Mutator + EquivalenceChecker protocols ---------------------------------


@dataclass
class MutationResult:
    """Return type of `Mutator.apply()`.

    `scenario` and `registry` replace the originals for the mutated run.
    `metadata` is per-rule data the equivalence checker may consume
    (e.g. the permutation used, or the LLM-generated paraphrase).
    """

    scenario: "Scenario"
    registry: "ToolRegistry"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EquivalenceResult:
    """Return type of `EquivalenceChecker.compare()`.

    `is_equivalent = True` → the pair is treated as "no bug" and not
    emitted to the HF dataset. `is_equivalent = False` makes the pair a
    bug candidate; `divergence_type` pinpoints what changed; `details` is
    a short human-readable note that ends up in the dataset row.
    """

    is_equivalent: bool
    divergence_type: DivergenceType = DivergenceType.NONE
    details: str = ""
    # Optional per-check numeric signal (e.g. tool-set Jaccard similarity,
    # final-answer cosine, etc.). Stage 3 analysis uses these for ranking.
    signal: float | None = None


class Mutator(Protocol):
    """Rule-specific scenario/registry transform."""

    rule_id: str

    def apply(
        self,
        scenario: "Scenario",
        registry: "ToolRegistry",
        *,
        seed: int = 0,
    ) -> MutationResult:
        """Return the mutated scenario + registry.

        Must be **deterministic** under a fixed `seed` — Stage 3 reruns the
        mutation when regenerating the HF dataset and the pair IDs must stay
        stable.
        """
        ...


class EquivalenceChecker(Protocol):
    """Rule-specific trajectory-pair comparator."""

    rule_id: str

    def compare(
        self,
        original: "Trajectory",
        mutated: "Trajectory",
        *,
        mutation_metadata: dict[str, Any] | None = None,
    ) -> EquivalenceResult:
        """Decide whether the two trajectories are equivalent under the rule.

        `mutation_metadata` carries any per-rule data from the mutator
        (e.g. the tool-name mapping used, so the checker knows a
        renamed-tool-call corresponds to the original). Missing metadata
        must not raise; default to "best effort" comparison.
        """
        ...


# -- Bug (HF-serialisable record) -------------------------------------------


@dataclass
class Bug:
    """A single divergent trajectory pair destined for the HF dataset.

    Fields map directly to the HF Dataset schema defined for
    `agentmorph/bugs-v0.1`. Every field is JSON-serialisable.

    Stage 3's aggregator constructs one `Bug` per (rule, model, scenario)
    triple where `EquivalenceChecker.compare()` returned `is_equivalent=False`.
    """

    bug_id: str                          # uuid4 hex; stable across reruns (deterministic from seed)
    rule_id: str                         # e.g. "tool-order-invariance"
    model_id: str                        # e.g. "Llama-3.2-3B"
    framework_id: str                    # "native" / "smolagents" / "langgraph"
    env_id: str                          # "ecommerce" / "agentdojo"
    scenario_id: str                     # e.g. "eco_shop_kettle"

    # The two trajectories, as dicts (already-serialised form — matches
    # `Trajectory.to_dict()`). Stored inline so the HF dataset is
    # self-contained without pointers to external JSONL files.
    original_trajectory: dict[str, Any]
    mutated_trajectory: dict[str, Any]

    # Classification.
    divergence_type: DivergenceType = DivergenceType.NONE
    severity: Severity = Severity.UNCLASSIFIED
    details: str = ""

    # Mutation-specific metadata the checker saw (e.g. permutation used,
    # paraphrase source text). Useful for reproducibility.
    mutation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict for JSONL / HF upload."""
        return {
            "bug_id": self.bug_id,
            "rule_id": self.rule_id,
            "model_id": self.model_id,
            "framework_id": self.framework_id,
            "env_id": self.env_id,
            "scenario_id": self.scenario_id,
            "original_trajectory": self.original_trajectory,
            "mutated_trajectory": self.mutated_trajectory,
            "divergence_type": self.divergence_type.value,
            "severity": self.severity.value,
            "details": self.details,
            "mutation_metadata": self.mutation_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Bug":
        """Parse a dict back into a Bug (e.g. loading from JSONL)."""
        return cls(
            bug_id=data["bug_id"],
            rule_id=data["rule_id"],
            model_id=data["model_id"],
            framework_id=data["framework_id"],
            env_id=data["env_id"],
            scenario_id=data["scenario_id"],
            original_trajectory=data["original_trajectory"],
            mutated_trajectory=data["mutated_trajectory"],
            divergence_type=DivergenceType(data.get("divergence_type", "none")),
            severity=Severity(data.get("severity", "unclassified")),
            details=data.get("details", ""),
            mutation_metadata=dict(data.get("mutation_metadata", {})),
        )
