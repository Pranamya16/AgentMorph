"""Registry + factory for environments."""

from __future__ import annotations

from typing import Any

from agentmorph.environments.base import Environment


ENVIRONMENT_IDS: tuple[str, ...] = ("ecommerce", "agentdojo")


def load_environment(env_id: str, **kwargs: Any) -> Environment:
    if env_id == "ecommerce":
        from agentmorph.environments.ecommerce_env import EcommerceEnvironment
        return EcommerceEnvironment(**kwargs)
    if env_id == "agentdojo":
        from agentmorph.environments.agentdojo_env import AgentDojoEnvironment
        return AgentDojoEnvironment(**kwargs)
    raise ValueError(
        f"unknown environment {env_id!r}; expected one of {ENVIRONMENT_IDS}"
    )
