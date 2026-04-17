"""Agent framework adapters.

Each adapter turns a (LoadedModel, ToolRegistry, prompt) tuple into a
`Trajectory`. The adapters are lazy — importing this package does not import
smolagents or LangGraph; that only happens when you call the framework-
specific factory.
"""

from __future__ import annotations

from agentmorph.agents.base import AgentConfig, AgentRunner, NativeAgent, run_agent
from agentmorph.agents.registry import FRAMEWORK_IDS, make_agent

__all__ = [
    "AgentConfig",
    "AgentRunner",
    "FRAMEWORK_IDS",
    "NativeAgent",
    "make_agent",
    "run_agent",
]
