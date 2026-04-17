"""Environment adapters.

Each environment produces `Scenario` objects and builds a fresh `ToolRegistry`
for a given scenario. Stage 2 mutators operate on (scenario, registry) pairs,
so keeping the interface identical across environments is load-bearing.
"""

from __future__ import annotations

from agentmorph.environments.base import (
    Environment,
    Scenario,
    ScenarioBundle,
)
from agentmorph.environments.registry import ENVIRONMENT_IDS, load_environment

__all__ = [
    "ENVIRONMENT_IDS",
    "Environment",
    "Scenario",
    "ScenarioBundle",
    "load_environment",
]
