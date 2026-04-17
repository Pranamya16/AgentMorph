"""smolagents adapter.

Wraps our `LoadedModel` as a smolagents `TransformersModel`-compatible callable
and converts each of our `Tool` objects into a smolagents `Tool`. The run log
is normalized back into our `Trajectory` schema so Stage 2 mutators see the
same structure regardless of framework.

Requires the `[smolagents]` extra; imports are deferred so the package is
importable without smolagents installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentmorph.agents.base import AgentConfig
from agentmorph.tools.base import Tool, ToolRegistry
from agentmorph.trajectories import Trajectory


# -- JSON Schema → smolagents `inputs` dict ---------------------------------

def _params_to_smolagents_inputs(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    required = set(schema.get("required", []))
    out: dict[str, dict[str, Any]] = {}
    for name, spec in schema.get("properties", {}).items():
        out[name] = {
            "type": spec.get("type", "any"),
            "description": spec.get("description", ""),
            "nullable": name not in required,
        }
    return out


def _wrap_tool(tool: Tool) -> Any:
    """Return a smolagents.Tool instance that proxies to our Tool."""
    from smolagents import Tool as SmolaTool  # lazy import

    class _Wrapped(SmolaTool):
        name = tool.name
        description = tool.description
        inputs = _params_to_smolagents_inputs(tool.parameters)
        output_type = "any"

        def forward(self, **kwargs: Any) -> Any:  # noqa: D401
            # smolagents passes through kwargs verbatim; our Tool validates them.
            return tool.invoke(kwargs)

    _Wrapped.__name__ = f"Smola_{tool.name}"
    return _Wrapped()


# -- Model wrapper ----------------------------------------------------------

def _wrap_model(loaded_model: Any) -> Any:
    """Return a callable conforming to the smolagents model protocol."""
    from smolagents import TransformersModel  # type: ignore

    # TransformersModel can take an already-loaded model + tokenizer by
    # passing `model_id=None` and setting attrs directly. The smolagents API
    # has shifted across versions; the safest bet is to construct via the
    # class and overwrite fields.
    class _Wrapped(TransformersModel):
        def __init__(self) -> None:
            # Skip the parent constructor — we already hold the weights.
            self.model_id = loaded_model.spec.hf_repo
            self.model = loaded_model.model
            self.tokenizer = loaded_model.tokenizer
            self.device = loaded_model.device
            self.kwargs = {}
            self.last_input_token_count = 0
            self.last_output_token_count = 0

        def __call__(self, messages, stop_sequences=None, **kwargs):  # type: ignore[override]
            text = loaded_model.chat(
                [{"role": m["role"], "content": m["content"]} for m in messages],
                max_new_tokens=kwargs.get("max_new_tokens", 384),
                temperature=kwargs.get("temperature", 0.0),
                stop=list(stop_sequences) if stop_sequences else None,
            )
            # smolagents expects a ChatMessage-like object; most versions accept
            # a plain string with a .content attribute. Use a tiny shim.
            class _Msg:
                role = "assistant"
                def __init__(self, content: str) -> None:
                    self.content = content
                    self.tool_calls = None
            return _Msg(text)

    return _Wrapped()


# -- Adapter ----------------------------------------------------------------

@dataclass
class SmolagentsAgent:
    """smolagents.ToolCallingAgent wired up to our model + tools."""

    loaded_model: Any
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
        from smolagents import ToolCallingAgent  # lazy import

        trajectory = Trajectory.new(
            scenario_id=scenario_id,
            env_id=env_id,
            model_id=self.config.model_id,
            framework_id=self.config.framework_id,
            prompt=prompt,
            metadata=dict(metadata or {}),
        )

        try:
            wrapped_tools = [_wrap_tool(t) for t in self.tools]
            model = _wrap_model(self.loaded_model)
            agent = ToolCallingAgent(
                tools=wrapped_tools,
                model=model,
                max_steps=self.config.max_steps,
            )
            final = agent.run(prompt)

            # Normalize smolagents' step log into our Trajectory format.
            logs = getattr(agent, "memory", None)
            steps = getattr(logs, "steps", []) if logs is not None else []
            for step in steps:
                # Each step in smolagents is an ActionStep-like object with
                # fields for the LLM output, tool call, and observation.
                llm_out = getattr(step, "model_output", None) or getattr(step, "llm_output", None)
                if llm_out:
                    trajectory.add_thought(str(llm_out))
                tool_calls = getattr(step, "tool_calls", None) or []
                for tc in tool_calls:
                    name = getattr(tc, "name", None) or getattr(tc, "tool_name", None)
                    args = getattr(tc, "arguments", None) or getattr(tc, "tool_args", None) or {}
                    if name is None:
                        continue
                    trajectory.add_tool_call(name, args if isinstance(args, dict) else {"value": args})
                obs = getattr(step, "observation", None) or getattr(step, "observations", None)
                err = getattr(step, "error", None)
                if obs is not None or err is not None:
                    tc = (tool_calls or [None])[0]
                    name = getattr(tc, "name", None) if tc is not None else None
                    trajectory.add_tool_result(
                        name or "unknown",
                        {},
                        output=obs,
                        error=str(err) if err else None,
                    )

            trajectory.add_final_answer(str(final) if final is not None else "")
        except Exception as exc:  # pragma: no cover — framework churn guard
            trajectory.add_error(f"smolagents adapter failed: {type(exc).__name__}: {exc}")

        trajectory.finish()
        return trajectory
