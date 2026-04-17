"""Framework registry + `make_agent(framework_id, model, tools, config)`.

The runner calls `make_agent` so the code path is uniform across frameworks.
"""

from __future__ import annotations

from typing import Any

from agentmorph.agents.base import AgentConfig, AgentRunner, NativeAgent
from agentmorph.tools.base import ToolRegistry


FRAMEWORK_IDS: tuple[str, ...] = ("native", "smolagents", "langgraph")


def make_agent(
    framework_id: str,
    *,
    loaded_model: Any,
    tools: ToolRegistry,
    config: AgentConfig,
) -> AgentRunner:
    """Build the agent runner for the given framework."""
    if framework_id == "native":
        return NativeAgent(loaded_model=loaded_model, tools=tools, config=config)
    if framework_id == "smolagents":
        # Lazy import keeps the package usable without the extra installed.
        from agentmorph.agents.smolagents_agent import SmolagentsAgent
        return SmolagentsAgent(loaded_model=loaded_model, tools=tools, config=config)
    if framework_id == "langgraph":
        from agentmorph.agents.langgraph_agent import LangGraphAgent
        return LangGraphAgent(loaded_model=loaded_model, tools=tools, config=config)
    raise ValueError(
        f"unknown framework {framework_id!r}; expected one of {FRAMEWORK_IDS}"
    )
