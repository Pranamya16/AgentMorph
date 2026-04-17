"""Tests for the NativeAgent ReAct loop.

Uses a `FakeModel` that replays a scripted sequence of assistant turns — no
torch / transformers needed. The goal is to exercise the adapter contract
and the parser, not model quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentmorph.agents.base import AgentConfig, NativeAgent
from agentmorph.tools.ecommerce import build_ecommerce_registry


@dataclass
class FakeModel:
    """Stand-in for `LoadedModel` that returns canned outputs."""

    outputs: list[str] = field(default_factory=list)
    calls: list[list[dict[str, str]]] = field(default_factory=list)
    spec: Any = None

    def __post_init__(self) -> None:
        # A tiny stub spec object — NativeAgent doesn't touch .spec, but the
        # Trajectory records model_id from AgentConfig, so we can leave this.
        pass

    def chat(self, messages, **kwargs) -> str:
        self.calls.append(list(messages))
        if not self.outputs:
            return "FINAL: out of canned outputs"
        return self.outputs.pop(0)


def test_native_agent_runs_tool_and_returns_final() -> None:
    _state, registry = build_ecommerce_registry()
    model = FakeModel(
        outputs=[
            '```json\n{"tool": "list_categories", "arguments": {}}\n```',
            "FINAL: Categories listed.",
        ]
    )
    agent = NativeAgent(
        loaded_model=model,
        tools=registry,
        config=AgentConfig(model_id="fake-model", framework_id="native", max_steps=4),
    )
    t = agent.run(prompt="what categories do you have?", scenario_id="sx", env_id="ecommerce")

    assert t.final_answer == "Categories listed."
    kinds = [s.kind.value for s in t.steps]
    assert "tool_call" in kinds and "tool_result" in kinds and "final_answer" in kinds
    # Tool result must carry the real output.
    tool_result = next(s for s in t.steps if s.kind.value == "tool_result")
    assert isinstance(tool_result.tool_output, list)


def test_native_agent_marks_max_steps_exhausted() -> None:
    _state, registry = build_ecommerce_registry()
    # No final answer ever — keep issuing a valid tool call.
    model = FakeModel(
        outputs=['```json\n{"tool": "list_categories", "arguments": {}}\n```'] * 20
    )
    agent = NativeAgent(
        loaded_model=model,
        tools=registry,
        config=AgentConfig(model_id="fake-model", framework_id="native", max_steps=2),
    )
    t = agent.run(prompt="noop", scenario_id="sy", env_id="ecommerce")
    assert any(s.kind.value == "error" for s in t.steps)
    assert t.final_answer is None


def test_native_agent_handles_unparseable_reply() -> None:
    _state, registry = build_ecommerce_registry()
    model = FakeModel(
        outputs=[
            "i don't know what i'm doing",
            "FINAL: giving up",
        ]
    )
    agent = NativeAgent(
        loaded_model=model,
        tools=registry,
        config=AgentConfig(model_id="fake-model", framework_id="native", max_steps=4),
    )
    t = agent.run(prompt="?", scenario_id="sz", env_id="ecommerce")
    assert t.final_answer == "giving up"
    assert any(s.kind.value == "error" for s in t.steps)


def test_native_agent_propagates_tool_errors_without_crashing() -> None:
    _state, registry = build_ecommerce_registry()
    model = FakeModel(
        outputs=[
            '```json\n{"tool": "get_product", "arguments": {"product_id": "DOES_NOT_EXIST"}}\n```',
            "FINAL: sorry",
        ]
    )
    agent = NativeAgent(
        loaded_model=model,
        tools=registry,
        config=AgentConfig(model_id="fake-model", framework_id="native", max_steps=4),
    )
    t = agent.run(prompt="?", scenario_id="sw", env_id="ecommerce")
    tr = next(s for s in t.steps if s.kind.value == "tool_result")
    assert tr.tool_error is not None and "DOES_NOT_EXIST" in tr.tool_error
