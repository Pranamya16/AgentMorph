"""Registry of the 10 Stage-2 metamorphic rules.

`RULE_IDS` is the authoritative ordered tuple of rule identifiers.
`make_mutator(rule_id)` and `make_equivalence_checker(rule_id)` are the
factory entry points the Stage 3 runner calls — they do lazy imports so
unfinished rules don't break imports of the rest of the package.

During Apr 22-27, each rule lands as `rules/<rule_id>.py` exposing a
`MUTATOR` and `CHECKER` module-level singleton. The factory looks up
`RULE_IDS` and imports on demand.

Keeping this explicit (vs. import *) catches typos and makes it trivial
to ship a partial rule set if the schedule tightens.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing only
    from agentmorph.rules.base import EquivalenceChecker, Mutator


#: Ordered tuple of the 10 rule identifiers. Source of truth for Stage 3
#: sweep matrix (models × rules × scenarios). Do NOT reorder without
#: bumping the HF dataset version.
RULE_IDS: tuple[str, ...] = (
    "tool-order-invariance",
    "schema-paraphrase-invariance",
    "synonym-robustness",
    "read-only-idempotency",
    "refusal-consistency",
    "tool-name-insensitivity",
    "parameter-order-invariance",
    "irrelevant-tool-insensitivity",
    "persona-insensitivity",
    "distractor-text-insensitivity",
)


def _module_name(rule_id: str) -> str:
    """`tool-order-invariance` -> `agentmorph.rules.tool_order_invariance`."""
    return f"agentmorph.rules.{rule_id.replace('-', '_')}"


def make_mutator(rule_id: str) -> "Mutator":
    """Return the mutator singleton for `rule_id`, or raise a clear error."""
    if rule_id not in RULE_IDS:
        raise KeyError(
            f"unknown rule {rule_id!r}; known rules: {sorted(RULE_IDS)}"
        )
    try:
        module = importlib.import_module(_module_name(rule_id))
    except ImportError as exc:
        raise NotImplementedError(
            f"rule {rule_id!r} is not yet implemented (module "
            f"{_module_name(rule_id)} missing). "
            f"See AgentMorph_Realistic_Plan.md §4 for the ship schedule."
        ) from exc
    mut = getattr(module, "MUTATOR", None)
    if mut is None:
        raise NotImplementedError(
            f"rule module {_module_name(rule_id)} must expose a "
            f"MUTATOR module-level singleton."
        )
    return mut


def make_equivalence_checker(rule_id: str) -> "EquivalenceChecker":
    """Return the equivalence checker singleton for `rule_id`."""
    if rule_id not in RULE_IDS:
        raise KeyError(
            f"unknown rule {rule_id!r}; known rules: {sorted(RULE_IDS)}"
        )
    try:
        module = importlib.import_module(_module_name(rule_id))
    except ImportError as exc:
        raise NotImplementedError(
            f"rule {rule_id!r} is not yet implemented (module "
            f"{_module_name(rule_id)} missing)."
        ) from exc
    chk = getattr(module, "CHECKER", None)
    if chk is None:
        raise NotImplementedError(
            f"rule module {_module_name(rule_id)} must expose a "
            f"CHECKER module-level singleton."
        )
    return chk


def available_rules() -> list[str]:
    """Rules that have a shipped module on disk (subset of RULE_IDS).

    Useful for partial-sweep runs — `for rule_id in available_rules(): ...`
    processes only what's ready.
    """
    ready: list[str] = []
    for rule_id in RULE_IDS:
        try:
            importlib.import_module(_module_name(rule_id))
            ready.append(rule_id)
        except ImportError:
            continue
    return ready
