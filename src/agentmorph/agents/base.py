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
    """Knobs shared by all adapters.

    Defaults are sized for Colab T4 + small open models. The framework
    adapters (smolagents / LangGraph) both append prior turns to each new
    prompt, so `max_steps=4` and `max_new_tokens=256` keep peak context
    under ~10K tokens — well inside T4's VRAM headroom at 4-bit.
    """

    model_id: str                       # "Llama-3.2-3B" etc — matches models.MODEL_REGISTRY
    framework_id: str                   # "native" / "smolagents" / "langgraph"
    max_steps: int = 3
    max_new_tokens: int = 192
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


# Fenced code block in any of the common markers the small open models like
# to emit: ```json (the one we ask for), ```python, plain ``` with no lang,
# and even ```tool_code (some Gemma variants). All capture the JSON body.
_JSON_BLOCK_RE = re.compile(
    r"```(?:json|python|tool_code|tool)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

# Last-resort: a bare JSON object in the output with no fence. Python's `re`
# can't balance braces, so we do a short linear scan from each candidate `{`
# that's followed by a `"tool"` or `"name"` key.
_BARE_JSON_START = re.compile(r'\{\s*"(?:tool|name)"\s*:', re.DOTALL)


def _find_bare_json(text: str) -> tuple[int, str] | None:
    """Return (start_index, substring) of a balanced JSON object in `text`
    that begins with `{"tool":` or `{"name":`, or None if not found."""
    for m in _BARE_JSON_START.finditer(text):
        start = m.start()
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return start, text[start : i + 1]
    return None

# Final-answer triggers, in decreasing strictness. Each captures the tail
# of the line/block as the answer.
_FINAL_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*FINAL\s*:\s*(.*)$", re.DOTALL | re.MULTILINE),
    re.compile(r"^\s*FINAL ANSWER\s*:\s*(.*)$", re.DOTALL | re.MULTILINE | re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:The final answer is|My final answer is|Answer)\s*:\s*(.*)$", re.DOTALL | re.IGNORECASE),
)


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


def _first_final_match(text: str) -> re.Match[str] | None:
    best: re.Match[str] | None = None
    for rx in _FINAL_REGEXES:
        m = rx.search(text)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best


def _normalize_tool_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Accept {tool, arguments} OR {name, arguments} OR {name, args} shapes."""
    name = obj.get("tool") or obj.get("name")
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    if not name or not isinstance(args, dict):
        return None
    return {"tool": name, "arguments": args}


def _parse_step(text: str) -> tuple[str, dict[str, Any] | str]:
    """Return either ('tool', {...}) or ('final', str) or ('noop', raw).

    Loose on purpose: small open models diverge from whatever format you
    asked for. We accept several fenced and unfenced tool-call shapes and
    several final-answer prefixes.
    """
    final_match = _first_final_match(text)

    fenced = _JSON_BLOCK_RE.search(text)
    if fenced is not None:
        json_start = fenced.start()
        json_body = fenced.group(1)
    else:
        bare = _find_bare_json(text)
        if bare is None:
            json_start = None
            json_body = None
        else:
            json_start, json_body = bare

    # Prefer whichever appears first in the output — if the model emits a
    # tool call and THEN declares a final answer in the same turn, run the
    # tool first (the final answer would have been premature anyway).
    if final_match and (json_start is None or final_match.start() < json_start):
        return "final", final_match.group(1).strip()
    if json_body is not None:
        try:
            obj = json.loads(json_body)
        except json.JSONDecodeError as exc:
            return "noop", f"JSON parse error: {exc}"
        normalized = _normalize_tool_obj(obj)
        if normalized is None:
            return "noop", "Tool call missing a recognizable name/arguments pair."
        return "tool", normalized
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
