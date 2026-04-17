"""Agent runner protocol + a minimal-dependency `NativeAgent` fallback.

`NativeAgent` implements a small ReAct loop directly against our
`LoadedModel` + `ToolRegistry`, producing a canonical `Trajectory`. It is:

  * the only adapter that works without the `[smolagents]` or `[langgraph]`
    extras — valuable for CI, for unit tests, and as a circuit-breaker if a
    framework version breaks mid-Stage-3;
  * the reference that the smolagents / LangGraph adapters normalize
    themselves against (same trajectory shape, same tool invocations).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from agentmorph.tools.base import ToolRegistry
from agentmorph.trajectories import Trajectory


@dataclass
class AgentConfig:
    """Knobs shared by all adapters."""

    model_id: str                       # "Llama-3.2-3B" etc — matches models.MODEL_REGISTRY
    framework_id: str                   # "native" / "smolagents" / "langgraph"
    max_steps: int = 8
    max_new_tokens: int = 384
    temperature: float = 0.0
    system_prompt: str | None = None


class AgentRunner(Protocol):
    """Thin protocol. Keeps the runner decoupled from any framework."""

    def run(
        self,
        *,
        prompt: str,
        scenario_id: str,
        env_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Trajectory: ...


# -- Prompt scaffolding used by the native fallback -------------------------

_NATIVE_SYSTEM = """You are a careful tool-using agent. You have access to the tools listed below.

On every turn, reply with EITHER a single JSON object in a fenced code block:

```json
{{"tool": "<tool_name>", "arguments": {{"<arg>": <value>, ...}}}}
```

OR, when you are ready to give the final answer, a line starting with `FINAL:`.

Available tools:
{tool_docs}

Rules:
- Always wrap tool calls in a ```json fenced block.
- Only call one tool per turn.
- Once you have enough information, respond with `FINAL: <answer>` and stop.
"""


_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_FINAL_RE = re.compile(r"FINAL:\s*(.*)", re.DOTALL)


def _render_tool_docs(registry: ToolRegistry) -> str:
    lines = []
    for tool in registry:
        required = tool.parameters.get("required", [])
        props = tool.parameters.get("properties", {})
        arg_sig = ", ".join(
            f"{name}{'' if name in required else '?'}: {spec.get('type', 'any')}"
            for name, spec in props.items()
        )
        lines.append(f"- {tool.name}({arg_sig}) — {tool.description}")
    return "\n".join(lines)


def _parse_step(text: str) -> tuple[str, dict[str, Any] | str]:
    """Return either ('tool', {...}) or ('final', str) or ('noop', raw)."""
    final_match = _FINAL_RE.search(text)
    json_match = _JSON_BLOCK_RE.search(text)

    # Prefer whichever appears first in the output.
    if final_match and (not json_match or final_match.start() < json_match.start()):
        return "final", final_match.group(1).strip()
    if json_match:
        try:
            obj = json.loads(json_match.group(1))
        except json.JSONDecodeError as exc:
            return "noop", f"JSON parse error: {exc}"
        if "tool" not in obj or "arguments" not in obj:
            return "noop", "Tool call missing `tool` or `arguments` field."
        return "tool", obj
    return "noop", text.strip()


@dataclass
class NativeAgent:
    """Manual ReAct loop over our LoadedModel + ToolRegistry."""

    loaded_model: Any                   # agentmorph.models.LoadedModel
    tools: ToolRegistry
    config: AgentConfig

    def run(
        self,
        *,
        prompt: str,
        scenario_id: str,
        env_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Trajectory:
        trajectory = Trajectory.new(
            scenario_id=scenario_id,
            env_id=env_id,
            model_id=self.config.model_id,
            framework_id=self.config.framework_id,
            prompt=prompt,
            metadata=dict(metadata or {}),
        )

        system = self.config.system_prompt or _NATIVE_SYSTEM.format(
            tool_docs=_render_tool_docs(self.tools)
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        try:
            for _ in range(self.config.max_steps):
                reply = self.loaded_model.chat(
                    messages,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
                trajectory.add_thought(reply)
                messages.append({"role": "assistant", "content": reply})

                kind, payload = _parse_step(reply)
                if kind == "final":
                    trajectory.add_final_answer(payload)  # type: ignore[arg-type]
                    break
                if kind == "tool":
                    assert isinstance(payload, dict)
                    name = payload["tool"]
                    args = payload["arguments"] or {}
                    trajectory.add_tool_call(name, args)
                    result = self.tools.call(name, args)
                    trajectory.add_tool_result(
                        name, args, output=result.output, error=result.error
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool `{name}` returned:\n"
                                f"{json.dumps(result.output, default=str) if result.ok else 'ERROR: ' + str(result.error)}"
                            ),
                        }
                    )
                    continue
                # Malformed output — nudge the model once, then stop.
                trajectory.add_error(f"unparseable step: {payload}")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last message was not a valid tool call or FINAL answer. "
                            "Respond with a ```json {...}``` block or `FINAL: ...`."
                        ),
                    }
                )
            else:
                trajectory.add_error("max_steps exhausted without a FINAL answer")
        except Exception as exc:  # pragma: no cover — defensive
            trajectory.add_error(f"{type(exc).__name__}: {exc}")

        trajectory.finish()
        return trajectory


def run_agent(
    runner: AgentRunner,
    *,
    prompt: str,
    scenario_id: str,
    env_id: str,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    """Thin convenience wrapper so callers don't repeat kwargs."""
    return runner.run(
        prompt=prompt, scenario_id=scenario_id, env_id=env_id, metadata=metadata
    )
