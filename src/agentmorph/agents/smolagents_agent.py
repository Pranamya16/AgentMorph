"""smolagents adapter.

Wraps our `LoadedModel` as a smolagents `TransformersModel`-compatible callable
and converts each of our `Tool` objects into a smolagents `Tool`. The run log
is normalized back into our `Trajectory` schema so Stage 2 mutators see the
same structure regardless of framework.

Requires the `[smolagents]` extra; imports are deferred so the package is
importable without smolagents installed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from agentmorph.agents.base import AgentConfig
from agentmorph.tools.base import Tool, ToolRegistry
from agentmorph.trajectories import Trajectory


# -- JSON Schema → smolagents `inputs` dict ---------------------------------

# smolagents restricts input types to a small enum; anything else raises at
# tool-class creation. Map JSON-Schema types into that enum.
_SMOLA_TYPES = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


def _params_to_smolagents_inputs(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    required = set(schema.get("required", []))
    out: dict[str, dict[str, Any]] = {}
    for name, spec in schema.get("properties", {}).items():
        t = _SMOLA_TYPES.get(spec.get("type", "string"), "string")
        entry: dict[str, Any] = {
            "type": t,
            "description": spec.get("description", "") or name,
        }
        if name not in required:
            entry["nullable"] = True
        out[name] = entry
    return out


def _wrap_tool(tool: Tool) -> Any:
    """Return a smolagents.Tool instance that proxies to our Tool.

    smolagents validates that `forward`'s signature matches the declared
    `inputs` dict (a `**kwargs` forward fails with "'forward' method
    parameters were {...}"), so we build a forward with the exact signature
    the tool expects.
    """
    from smolagents import Tool as SmolaTool  # lazy import

    properties = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))
    param_names = list(properties.keys())

    inputs_dict = _params_to_smolagents_inputs(tool.parameters)

    # Build an explicit signature for `forward(self, <named params>...)`.
    fwd_params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for name in param_names:
        default = inspect.Parameter.empty if name in required else None
        fwd_params.append(
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default
            )
        )
    fwd_sig = inspect.Signature(fwd_params)

    def forward(self, **kwargs: Any) -> Any:
        # Drop Nones so optional args don't mask our JSON-Schema defaults.
        payload = {k: v for k, v in kwargs.items() if v is not None}
        return tool.invoke(payload)

    forward.__signature__ = fwd_sig  # type: ignore[attr-defined]
    forward.__name__ = "forward"

    cls = type(
        f"Smola_{tool.name}",
        (SmolaTool,),
        {
            "name": tool.name,
            "description": tool.description,
            "inputs": inputs_dict,
            # smolagents' AUTHORIZED_TYPES doesn't include "any"; "object" is
            # the broadest safe choice for tools that return dicts/lists.
            "output_type": "object",
            "forward": forward,
        },
    )
    return cls()


# -- Model wrapper ----------------------------------------------------------

def _wrap_model(loaded_model: Any) -> Any:
    """Return a callable conforming to the smolagents model protocol."""
    from smolagents import TransformersModel  # type: ignore

    # TransformersModel can take an already-loaded model + tokenizer by
    # passing `model_id=None` and setting attrs directly. The smolagents API
    # has shifted across versions; the safest bet is to construct via the
    # class and overwrite fields.
    class _Wrapped(TransformersModel):
        # Class-level defaults for every attribute smolagents' `Model` base
        # class normally sets during `__init__`. We skip that `__init__`
        # (otherwise the parent tries to re-download the weights we already
        # hold), so any attribute it would have set must be pre-declared
        # here or smolagents hits AttributeError on first call.
        #
        # The __getattr__ fallback below breaks the whack-a-mole cycle where
        # each new smolagents release adds another *_kwargs attr we haven't
        # listed — any unknown attribute ending in `_kwargs` defaults to
        # an empty dict, which is what the base class would have set anyway.
        flatten_messages_as_text = False          # we drive apply_chat_template ourselves
        _custom_role_conversions: dict | None = None
        tool_name_key = "name"
        tool_arguments_key = "arguments"
        model_kwargs: dict = {}
        apply_chat_template_kwargs: dict = {}
        tokenizer_kwargs: dict = {}
        generation_kwargs: dict = {}
        stream = False
        structured_generation_provider = False

        def __getattr__(self, name: str) -> Any:
            # Only called when normal attribute lookup fails. Provide safe
            # defaults for the `*_kwargs` pattern so smolagents' internals
            # don't crash on attrs we haven't explicitly listed.
            if name.startswith("__"):
                raise AttributeError(name)
            if name.endswith("_kwargs"):
                return {}
            # Anything else: propagate AttributeError so smolagents falls
            # back to its own defaults rather than getting a silent None.
            raise AttributeError(name)

        def __init__(self) -> None:
            # Deliberately skip TransformersModel.__init__ (it would reload
            # the weights). The class attrs above cover Model.__init__'s
            # side effects.
            self.model_id = loaded_model.spec.hf_repo
            self.model = loaded_model.model
            self.tokenizer = loaded_model.tokenizer
            self.device = loaded_model.device
            self.kwargs = {}
            self.last_input_token_count = 0
            self.last_output_token_count = 0

        def __call__(self, messages, stop_sequences=None, **kwargs):  # type: ignore[override]
            # Flatten any structured content chunks (smolagents stores assistant
            # outputs as `[{"type": "text", "text": "..."}]` internally; when
            # those get echoed back on subsequent turns we see a list here).
            conv: list[dict[str, str]] = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for chunk in content:
                        if isinstance(chunk, dict):
                            parts.append(str(chunk.get("text") or chunk.get("content") or ""))
                        else:
                            parts.append(str(chunk))
                    content = "\n".join(parts)
                conv.append({"role": role, "content": str(content)})

            text = loaded_model.chat(
                conv,
                max_new_tokens=kwargs.get("max_new_tokens", 256),
                temperature=kwargs.get("temperature", 0.0),
                stop=list(stop_sequences) if stop_sequences else None,
            )

            # Return a real smolagents ChatMessage so round-tripping through
            # the agent's memory preserves `.content` as a plain string.
            # Falling back to a minimal shim keeps the adapter importable on
            # older smolagents versions that don't export ChatMessage.
            try:
                from smolagents.models import ChatMessage  # type: ignore
                return ChatMessage(role="assistant", content=text, tool_calls=None)
            except Exception:
                class _Msg:
                    role = "assistant"
                    def __init__(self, content: str) -> None:
                        self.content = content
                        self.tool_calls = None
                    def __str__(self) -> str:
                        return self.content
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
