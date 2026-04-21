"""Shared utilities for Stage-2 rule implementations.

Every rule in `agentmorph.rules.*` consumes at least one helper here. Keeping
them in `_shared.py` means:
  * rule modules are short (≤ 100 lines each) and easy to audit
  * cross-rule inconsistencies (e.g. two rules defining their own
    tool-call-set-equal) are eliminated by construction
  * the equivalence-checker vocabulary stays uniform so the HF dataset's
    `divergence_type` values are directly comparable across rules

Pure Python — no torch, no framework imports. The package convention is
that `agentmorph.rules` stays importable without the `[models]` extra.
"""

from __future__ import annotations

import copy
import dataclasses
import random
import re
from typing import TYPE_CHECKING, Any, Callable, Hashable

from agentmorph.rules.base import (
    DivergenceType,
    EquivalenceResult,
    MutationResult,
    Mutator,
)
from agentmorph.tools.base import Tool, ToolRegistry

if TYPE_CHECKING:  # pragma: no cover — typing only
    from agentmorph.environments.base import Scenario, ScenarioBundle
    from agentmorph.trajectories import Trajectory


# -- Registry cloning --------------------------------------------------------


def clone_registry(
    registry: ToolRegistry,
    transform: Callable[[Tool], Tool | None] = lambda t: t,
    *,
    extra_tools: list[Tool] | None = None,
) -> ToolRegistry:
    """Return a new `ToolRegistry` with `transform` applied to each tool.

    Parameters
    ----------
    registry:
        Source registry. Not mutated.
    transform:
        Called on each tool. Return a new `Tool` to keep it, or return
        `None` to drop it from the cloned registry.
    extra_tools:
        Optional extra tools appended after the transformed set. Used by
        rules that inject distractor or read-only tools.

    Used by rules 1, 2, 6, 7, 8 — any rule that mutates the tool surface
    returns a cloned registry via this helper so the original is safe to
    reuse for the baseline run.
    """
    cloned = ToolRegistry()
    for tool in registry:
        new = transform(tool)
        if new is None:
            continue
        cloned.register(new)
    if extra_tools:
        for tool in extra_tools:
            cloned.register(tool)
    return cloned


def reorder_registry(
    registry: ToolRegistry,
    *,
    seed: int = 0,
    scenario_id: str = "",
) -> tuple[ToolRegistry, list[str]]:
    """Deterministic permutation of a registry's tool ordering.

    Returns `(new_registry, permutation_names)` where `permutation_names`
    is the sequence of tool names in the new order — useful for rule 1's
    metadata.
    """
    tool_list = list(registry)
    rng = random.Random(hash((seed, scenario_id)) & 0xFFFFFFFF)
    permuted = rng.sample(tool_list, len(tool_list))
    new = ToolRegistry()
    for tool in permuted:
        new.register(tool)
    return new, [t.name for t in permuted]


# -- SystemPromptMutator base -----------------------------------------------


class SystemPromptMutator(Mutator):
    """Base class for rules that mutate the *tool surface* of the system prompt.

    Subclass responsibilities:
      * set `rule_id`
      * implement `_transform_tool(tool) -> Tool | None`
      * override `_extra_tools(registry, seed, scenario)` if injecting
        additional tools (rule 8)
      * override `_metadata(registry, new_registry, seed, scenario)` to
        surface rule-specific metadata

    The scenario itself is not mutated here — only the bound ToolRegistry.
    Rules that mutate the scenario prompt (rules 3, 5, 9, 10) should
    subclass `ScenarioPromptMutator` below instead.
    """

    rule_id: str = "abstract"

    def _transform_tool(self, tool: Tool) -> Tool | None:  # pragma: no cover — abstract
        return tool

    def _extra_tools(
        self, registry: ToolRegistry, *, seed: int, scenario: "Scenario"
    ) -> list[Tool] | None:
        return None

    def _metadata(
        self,
        *,
        registry: ToolRegistry,
        new_registry: ToolRegistry,
        seed: int,
        scenario: "Scenario",
    ) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "seed": seed,
            "original_tool_names": [t.name for t in registry],
            "mutated_tool_names": [t.name for t in new_registry],
        }

    def apply(
        self,
        scenario: "Scenario",
        registry: ToolRegistry,
        *,
        seed: int = 0,
    ) -> MutationResult:
        new_registry = clone_registry(
            registry,
            self._transform_tool,
            extra_tools=self._extra_tools(registry, seed=seed, scenario=scenario),
        )
        return MutationResult(
            scenario=scenario,
            registry=new_registry,
            metadata=self._metadata(
                registry=registry,
                new_registry=new_registry,
                seed=seed,
                scenario=scenario,
            ),
        )


# -- ScenarioPromptMutator base ---------------------------------------------


class ScenarioPromptMutator(Mutator):
    """Base for rules that mutate the scenario's user prompt (rules 3, 5, 9, 10).

    Subclasses implement `_mutate_prompt(prompt, seed, scenario)`.
    """

    rule_id: str = "abstract"

    def _mutate_prompt(
        self, prompt: str, *, seed: int, scenario: "Scenario"
    ) -> tuple[str, dict[str, Any]]:
        """Return `(new_prompt, extra_metadata)`."""
        raise NotImplementedError

    def apply(
        self,
        scenario: "Scenario",
        registry: ToolRegistry,
        *,
        seed: int = 0,
    ) -> MutationResult:
        new_prompt, extra = self._mutate_prompt(
            scenario.prompt, seed=seed, scenario=scenario
        )
        # Preserve scenario id + metadata so downstream comparisons line up.
        new_metadata = {
            **scenario.metadata,
            "original_prompt": scenario.prompt,
            "mutator_extra": extra,
        }
        new_scenario = dataclasses.replace(
            scenario,
            prompt=new_prompt,
            metadata=new_metadata,
        )
        return MutationResult(
            scenario=new_scenario,
            registry=registry,
            metadata={
                "rule_id": self.rule_id,
                "seed": seed,
                "original_prompt": scenario.prompt,
                "new_prompt": new_prompt,
                **extra,
            },
        )


# -- State snapshot (rule 4) ------------------------------------------------


def snapshot_state(bundle: "ScenarioBundle") -> dict[str, Any]:
    """Deterministic JSON-friendly snapshot of mutable ShopState fields.

    The product catalog is excluded (immutable across a scenario).

    Returns an empty dict if the bundle's state is not a `ShopState` — this
    keeps rule 4 from crashing on AgentDojo scenarios. Stage 2 only runs
    ecommerce scenarios, so AgentDojo isn't a real concern for now.
    """
    state = getattr(bundle, "state", None)
    if state is None:
        return {}

    # Local import so `rules._shared` stays importable without the
    # ecommerce module being on the path.
    try:
        from agentmorph.tools.ecommerce.state import ShopState
    except Exception:  # pragma: no cover — defensive
        return {}

    if not isinstance(state, ShopState):
        return {}

    def _addr_view(a):
        return a.view() if hasattr(a, "view") else {}

    return {
        "cart_items": [
            {"product_id": item.product_id, "quantity": item.quantity}
            for item in state.cart.items.values()
        ],
        "cart_promo_code": state.cart.promo_code,
        "orders": [
            {
                "id": o.id,
                "status": o.status,
                "total": o.total,
                "address_id": o.address_id,
                "payment_method_id": o.payment_method_id,
                "return_reason": o.return_reason,
                "tracking_number": o.tracking_number,
            }
            for o in state.orders.values()
        ],
        "addresses": {aid: _addr_view(a) for aid, a in state.addresses.items()},
        "payment_methods": {pid: pm.view() for pid, pm in state.payment_methods.items()},
        "reviews": [
            {"product_id": r.product_id, "rating": r.rating, "author": r.author}
            for r in state.reviews
        ],
        "tickets": {tid: t.__dict__.copy() for tid, t in state.tickets.items()},
        "user_profile": state.user.__dict__.copy(),
        "product_stock": {pid: p.stock for pid, p in state.products.items()},
    }


def state_delta(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    """Top-level diff between two state snapshots.

    Returns `{field: {"before": ..., "after": ...}}` only for fields that
    actually changed. Empty dict = no side effects.
    """
    delta: dict[str, Any] = {}
    for key in set(pre) | set(post):
        before = pre.get(key)
        after = post.get(key)
        if before != after:
            delta[key] = {"before": before, "after": after}
    return delta


# -- Tool-call comparison primitives ----------------------------------------


def tool_calls_of(
    trajectory: "Trajectory" | dict[str, Any],
    *,
    name_mapping: dict[str, str] | None = None,
) -> list[tuple[str, frozenset[tuple[str, Hashable]]]]:
    """Normalise a trajectory's tool_call events into `(name, frozen-args)` tuples.

    Accepts either a `Trajectory` object or a `to_dict()`-style dict — the
    HF-dataset path stores dicts, the live-runner path holds objects.

    `name_mapping` optionally translates tool names (used by rule 6's
    tool-name-insensitivity checker to compare across renames).
    """
    steps = _steps_of(trajectory)

    out: list[tuple[str, frozenset[tuple[str, Hashable]]]] = []
    for step in steps:
        kind = step.get("kind") if isinstance(step, dict) else step.kind
        if (kind.value if hasattr(kind, "value") else kind) != "tool_call":
            continue
        name = step.get("tool_name") if isinstance(step, dict) else step.tool_name
        args = step.get("tool_args") if isinstance(step, dict) else step.tool_args
        args = args or {}
        if name_mapping and name in name_mapping:
            name = name_mapping[name]
        # Freeze the args dict; coerce unhashable values to their repr.
        frozen_args = frozenset(
            (k, _hashable(v)) for k, v in args.items()
        )
        out.append((name, frozen_args))
    return out


def _hashable(value: Any) -> Hashable:
    """Best-effort hashable coercion for equivalence comparison."""
    try:
        hash(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return tuple(sorted(((k, _hashable(v)) for k, v in value.items())))
        if isinstance(value, (list, tuple, set)):
            return tuple(_hashable(v) for v in value)
        return repr(value)


def _steps_of(trajectory: "Trajectory" | dict[str, Any]) -> list[Any]:
    if isinstance(trajectory, dict):
        return trajectory.get("steps", [])
    return trajectory.steps


def tool_call_set_equal(
    t1: "Trajectory" | dict[str, Any],
    t2: "Trajectory" | dict[str, Any],
    *,
    name_mapping: dict[str, str] | None = None,
) -> bool:
    """True if `t1` and `t2` made the same multiset of tool calls (order-insensitive)."""
    return sorted(tool_calls_of(t1, name_mapping=name_mapping)) == sorted(
        tool_calls_of(t2, name_mapping=name_mapping)
    )


def tool_call_sequence_equal(
    t1: "Trajectory" | dict[str, Any],
    t2: "Trajectory" | dict[str, Any],
    *,
    name_mapping: dict[str, str] | None = None,
) -> bool:
    """True if `t1` and `t2` made the same tool calls in the same order."""
    return tool_calls_of(t1, name_mapping=name_mapping) == tool_calls_of(
        t2, name_mapping=name_mapping
    )


# -- Answer-equivalence primitive -------------------------------------------


_WS_RE = re.compile(r"\s+")


def normalise_answer(text: str | None) -> str:
    """Lowercase, collapse whitespace, strip. Idempotent + deterministic."""
    if text is None:
        return ""
    return _WS_RE.sub(" ", text).strip().lower()


def final_answer_semantically_equal(
    t1: "Trajectory" | dict[str, Any],
    t2: "Trajectory" | dict[str, Any],
) -> bool:
    """Best-effort answer-equality check.

    Not an LLM judge — Stage 2 keeps comparisons deterministic and fast.
    Two answers are equal if their normalised forms are equal OR one is a
    prefix of the other (models often truncate vs. elaborate the same
    underlying answer).
    """
    a = normalise_answer(_final_answer_of(t1))
    b = normalise_answer(_final_answer_of(t2))
    if not a and not b:
        return True  # both no-answer; equivalent at this axis (completion check elsewhere)
    if a == b:
        return True
    if a and b and (a.startswith(b) or b.startswith(a)):
        return True
    return False


def _final_answer_of(trajectory: "Trajectory" | dict[str, Any]) -> str | None:
    if isinstance(trajectory, dict):
        return trajectory.get("final_answer")
    return trajectory.final_answer


# -- Completion-status comparison -------------------------------------------


def completed(trajectory: "Trajectory" | dict[str, Any]) -> bool:
    """Did the trajectory reach a final answer (vs. max_steps / adapter error)?"""
    fa = _final_answer_of(trajectory)
    return fa is not None and str(fa).strip() != ""


# -- Ready-made equivalence-checker templates -------------------------------
#
# Rules assemble these into their own CHECKER singletons. Keeps per-rule
# checkers ≤ 20 lines.


def classify_simple_divergence(
    original: "Trajectory" | dict[str, Any],
    mutated: "Trajectory" | dict[str, Any],
    *,
    name_mapping: dict[str, str] | None = None,
    sequence_reorder_is_equivalent: bool = True,
) -> EquivalenceResult:
    """Common decision tree used by rules 1, 6, 9, 10 (and rule 8's post-
    distractor check).

    Returns `NONE` + equivalent if both trajectories agree on tool set +
    answer + completion. Classifies divergences by the most specific
    matching type.

    `sequence_reorder_is_equivalent`: if True (rule 1 default), a
    difference that is *only* tool-order (same set) counts as equivalent
    but with `divergence_type=REORDER_ONLY` recorded for stats.
    """
    if completed(original) != completed(mutated):
        return EquivalenceResult(
            is_equivalent=False,
            divergence_type=DivergenceType.COMPLETION_DIFFERS,
            details=f"completion differs: original={completed(original)}, mutated={completed(mutated)}",
        )

    same_set = tool_call_set_equal(original, mutated, name_mapping=name_mapping)
    same_seq = tool_call_sequence_equal(original, mutated, name_mapping=name_mapping)
    same_answer = final_answer_semantically_equal(original, mutated)

    if same_set and same_seq and same_answer:
        return EquivalenceResult(is_equivalent=True, divergence_type=DivergenceType.NONE)

    if same_set and not same_seq:
        # Tool set identical, order differs.
        if sequence_reorder_is_equivalent:
            return EquivalenceResult(
                is_equivalent=True,
                divergence_type=DivergenceType.REORDER_ONLY,
                details="same tool set, different order",
            )
        return EquivalenceResult(
            is_equivalent=False,
            divergence_type=DivergenceType.REORDER_ONLY,
            details="same tool set, different order",
        )

    if not same_set:
        return EquivalenceResult(
            is_equivalent=False,
            divergence_type=DivergenceType.TOOL_SET_DIFFERS,
            details="tool sets differ",
        )

    # Same tools + order but different answer.
    return EquivalenceResult(
        is_equivalent=False,
        divergence_type=DivergenceType.ANSWER_DIFFERS,
        details="final answers differ after normalisation",
    )
