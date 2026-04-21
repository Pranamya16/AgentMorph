"""Metamorphic rules (Stage 2).

Each rule defines:
  * a `Mutator` — takes a `Scenario` + a scenario's bound `ToolRegistry` and
    returns a mutated (Scenario, ToolRegistry) tuple whose agent behaviour
    should be *equivalent* to the original under the rule's invariant;
  * an `EquivalenceChecker` — given the original and mutated trajectories,
    classifies the pair as equivalent or as a concrete divergence type.

Ten rules land here across Apr 22-27 per `AgentMorph_Realistic_Plan.md` §4:

    1. tool-order-invariance             (Apr 22)
    2. schema-paraphrase-invariance      (Apr 23, Gemini-backed)
    3. synonym-robustness                (Apr 23, Gemini-backed)
    4. read-only-idempotency             (Apr 27, hardest — last)
    5. refusal-consistency               (Apr 24, Gemini-backed)
    6. tool-name-insensitivity           (Apr 24)
    7. parameter-order-invariance        (Apr 25)
    8. irrelevant-tool-insensitivity     (Apr 25)
    9. persona-insensitivity             (Apr 26)
   10. distractor-text-insensitivity     (Apr 26)

Pure Python — no GPU required to build or test a mutator.
"""

from __future__ import annotations

from agentmorph.rules.base import (
    Bug,
    DivergenceType,
    EquivalenceChecker,
    EquivalenceResult,
    MutationResult,
    Mutator,
)
from agentmorph.rules.registry import (
    RULE_IDS,
    available_rules,
    make_equivalence_checker,
    make_mutator,
)

__all__ = [
    "Bug",
    "DivergenceType",
    "EquivalenceChecker",
    "EquivalenceResult",
    "MutationResult",
    "Mutator",
    "RULE_IDS",
    "available_rules",
    "make_equivalence_checker",
    "make_mutator",
]
