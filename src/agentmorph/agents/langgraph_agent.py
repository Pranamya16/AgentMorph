"""LangGraph adapter.

Builds a `create_react_agent` graph using a HuggingFacePipeline chat model
backed by our `LoadedModel`, with our `Tool` objects converted to
LangChain `StructuredTool`s. Streams graph events and normalizes them into
our `Trajectory` schema.

Requires the `[langgraph]` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentmorph.agents.base import AgentConfig
from agentmorph.tools.base import Tool, ToolRegistry
from agentmorph.trajectories import Trajectory


# -- JSON Schema → pydantic for StructuredTool args_schema -------------------

_NULLISH = {"null", "none", "nil", "", "undefined", "n/a"}


def _coerce_arg(value: Any) -> Any:
    """Coerce string-encoded primitives from tool-calling LLMs.

    Small open models routinely emit JSON args like:
      {"category": "null", "min_price": "0", "limit": "1"}
    i.e. strings instead of proper JSON null/int. Treat common string spellings
    of null as None so pydantic drops them back to the field default, and let
    pydantic itself handle "0" -> 0 / "50" -> 50.0 via its default coercion.
    """
    if isinstance(value, str):
        if value.strip().lower() in _NULLISH:
            return None
    return value


def _tool_to_structured(tool: Tool) -> Any:
    """Wrap our Tool as a LangChain StructuredTool."""
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model  # pydantic v2

    props = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))

    py_type = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    fields: dict[str, tuple[Any, Any]] = {}
    for name, spec in props.items():
        t = py_type.get(spec.get("type", "string"), str)
        if name in required:
            fields[name] = (t, Field(..., description=spec.get("description", "")))
        else:
            default = spec.get("default", None)
            fields[name] = (t | None, Field(default, description=spec.get("description", "")))

    model_cls = create_model(f"{tool.name}_Args", **fields)  # type: ignore[arg-type]

    def _call(**kwargs: Any) -> Any:
        cleaned = {k: _coerce_arg(v) for k, v in kwargs.items()}
        return tool.invoke({k: v for k, v in cleaned.items() if v is not None})

    return StructuredTool.from_function(
        func=_call,
        name=tool.name,
        description=tool.description,
        args_schema=model_cls,
    )


# -- Chat-model wrapper around our LoadedModel ------------------------------

def _wrap_chat_model(loaded_model: Any, *, temperature: float, max_new_tokens: int) -> Any:
    """Return a `BaseChatModel`-compatible wrapper using transformers directly.

    Implements:
      * `_generate` that delegates to `LoadedModel.chat`
      * `bind_tools` that returns a tool-aware copy of the wrapper
        (`create_react_agent` calls this — without it LangGraph raises
        `NotImplementedError` and we get zero finished trajectories)
      * text-level tool-call parsing: when tools are bound, we inject a
        system message teaching the model our JSON-fenced tool-call format,
        then parse any matching block out of the generated text and surface
        it as a structured `AIMessage.tool_calls` entry. This is what
        `create_react_agent`'s ReAct loop expects to dispatch tools.
    """
    import json as _json
    import re as _re
    import uuid as _uuid

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    # Matches the START of a JSON tool-call object inside any fenced or
    # unfenced region. The actual end of the object is found via balanced-
    # brace scan (Python re can't balance `{}`), so nested `arguments: {...}`
    # and multi-call `;`-separated blobs both resolve correctly.
    _JSON_OBJ_START = _re.compile(r'\{\s*"(?:name|tool)"\s*:', _re.DOTALL)

    def _scan_balanced_json(text: str) -> str | None:
        """Return the first balanced `{...}` object starting with a `name`
        or `tool` key, or None if not found."""
        for m in _JSON_OBJ_START.finditer(text):
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
                        return text[start : i + 1]
        return None

    def _tool_docs(tools: list) -> str:
        lines = []
        for t in tools:
            try:
                arg_names = list(getattr(t, "args_schema").__fields__.keys())  # pydantic v1
            except Exception:
                try:
                    arg_names = list(getattr(t, "args_schema").model_fields.keys())  # pydantic v2
                except Exception:
                    arg_names = []
            args = ", ".join(arg_names)
            lines.append(f"- {t.name}({args}) — {t.description}")
        return "\n".join(lines)

    def _tool_system_prompt(tools: list) -> str:
        return (
            "You have access to these tools:\n"
            f"{_tool_docs(tools)}\n\n"
            "To call a tool, respond with EXACTLY one JSON object in a fenced block:\n"
            "```json\n"
            '{"name": "<tool_name>", "arguments": {"<arg>": <value>, ...}}\n'
            "```\n"
            "When you have the final answer, respond in plain text (no JSON)."
        )

    def _parse_tool_call(text: str, allowed_names: set[str]) -> dict | None:
        """Return {name, args} for the FIRST tool-call JSON object in `text`.

        Finds the object via balanced-brace scan (so nested `arguments`
        dicts don't confuse it), then validates the name is one of the
        bound tools. Returns None if no valid tool call is found.
        """
        raw = _scan_balanced_json(text)
        if raw is None:
            return None
        try:
            obj = _json.loads(raw)
        except _json.JSONDecodeError:
            return None
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        if not name or name not in allowed_names or not isinstance(args, dict):
            return None
        return {"name": name, "args": args}

    class _LocalChat(BaseChatModel):
        # Make these regular fields so pydantic v2 is happy on BaseChatModel.
        bound_tools: list = []

        @property
        def _llm_type(self) -> str:
            return f"agentmorph-{loaded_model.spec.id}"

        def bind_tools(self, tools, **_kwargs):  # type: ignore[override]
            # `create_react_agent` calls this once to register tools. Return
            # a copy so the shared instance stays untools-aware.
            new = _LocalChat()
            new.bound_tools = list(tools)
            return new

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            conv: list[dict[str, str]] = []

            if self.bound_tools:
                conv.append({"role": "system", "content": _tool_system_prompt(self.bound_tools)})

            for m in messages:
                role = {
                    "human": "user",
                    "ai": "assistant",
                    "system": "system",
                    "tool": "tool",
                }.get(m.type, "user")
                # LangGraph's tool messages carry name + content; fold them into
                # a user-visible observation so the next turn can see the result.
                if m.type == "tool":
                    tool_name = getattr(m, "name", "tool")
                    conv.append({
                        "role": "user",
                        "content": f"Tool `{tool_name}` returned:\n{m.content}",
                    })
                    continue
                conv.append({
                    "role": role,
                    "content": m.content if isinstance(m.content, str) else str(m.content),
                })

            text = loaded_model.chat(
                conv,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
            )

            allowed = {t.name for t in self.bound_tools} if self.bound_tools else set()
            parsed = _parse_tool_call(text, allowed)
            if parsed is not None:
                ai = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": parsed["name"],
                        "args": parsed["args"],
                        "id": f"call_{_uuid.uuid4().hex[:12]}",
                        "type": "tool_call",
                    }],
                )
            else:
                ai = AIMessage(content=text)
            return ChatResult(generations=[ChatGeneration(message=ai)])

    return _LocalChat()


# -- Adapter ----------------------------------------------------------------

@dataclass
class LangGraphAgent:
    """LangGraph ReAct agent wired up to our model + tools."""

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
        from langgraph.prebuilt import create_react_agent  # lazy
        from langchain_core.messages import HumanMessage, SystemMessage

        trajectory = Trajectory.new(
            scenario_id=scenario_id,
            env_id=env_id,
            model_id=self.config.model_id,
            framework_id=self.config.framework_id,
            prompt=prompt,
            metadata=dict(metadata or {}),
        )

        try:
            chat = _wrap_chat_model(
                self.loaded_model,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
            )
            lc_tools = [_tool_to_structured(t) for t in self.tools]

            sys_msg = self.config.system_prompt or (
                "You are a careful tool-using agent. Call tools when useful and stop "
                "once you have the answer."
            )

            graph = create_react_agent(chat, tools=lc_tools)

            # Invoke synchronously and normalize the resulting message list.
            result = graph.invoke(
                {"messages": [SystemMessage(content=sys_msg), HumanMessage(content=prompt)]},
                config={"recursion_limit": max(4, self.config.max_steps * 2)},
            )

            messages = result.get("messages", [])
            final_answer = None
            for msg in messages:
                msg_type = getattr(msg, "type", None)
                if msg_type == "ai":
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if content:
                        trajectory.add_thought(content)
                    for tc in getattr(msg, "tool_calls", []) or []:
                        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                        if name:
                            trajectory.add_tool_call(name, args or {})
                    final_answer = content
                elif msg_type == "tool":
                    trajectory.add_tool_result(
                        getattr(msg, "name", "unknown") or "unknown",
                        {},
                        output=msg.content if isinstance(msg.content, str) else str(msg.content),
                    )

            if final_answer is not None:
                trajectory.add_final_answer(final_answer)
            else:
                trajectory.add_error("langgraph produced no final AIMessage")
        except Exception as exc:  # pragma: no cover — framework churn guard
            trajectory.add_error(f"langgraph adapter failed: {type(exc).__name__}: {exc}")

        trajectory.finish()
        return trajectory
