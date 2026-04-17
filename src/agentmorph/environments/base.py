"""Scenario / Environment protocol shared by AgentDojo + ecommerce adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from agentmorph.tools.base import ToolRegistry


@dataclass
class Scenario:
    """A single task for an agent to run.

    Kept deliberately tiny: id, user prompt, pointer back to the environment
    that built it, plus freeform metadata. Mutators in Stage 2 will derive
    new scenarios by copying + edit.
    """

    id: str
    env_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioBundle:
    """A scenario paired with a freshly-reset tool registry."""

    scenario: Scenario
    registry: ToolRegistry
    # Opaque per-env state handle (e.g., ShopState) so tests / judges can peek.
    state: Any | None = None


class Environment(Protocol):
    """Environment adapter protocol."""

    env_id: str

    def scenarios(self) -> Iterator[Scenario]: ...
    def reset(self, scenario: Scenario) -> ScenarioBundle: ...
