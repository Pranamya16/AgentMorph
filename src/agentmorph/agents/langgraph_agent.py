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
        return tool.invoke({k: v for k, v in kwargs.items() if v is not None})

    return StructuredTool.from_function(
        func=_call,
        name=tool.name,
        description=tool.description,
        args_schema=model_cls,
    )


# -- Chat-model wrapper around our LoadedModel ------------------------------

def _wrap_chat_model(loaded_model: Any, *, temperature: float, max_new_tokens: int) -> Any:
    """Return a `BaseChatModel`-compatible wrapper using transformers directly.

    Avoids the version churn of `langchain-huggingface` by subclassing
    `BaseChatModel` with a minimal `_generate` that delegates to
    `LoadedModel.chat`.
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _LocalChat(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return f"agentmorph-{loaded_model.spec.id}"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            conv = []
            for m in messages:
                role = {
                    "human": "user",
                    "ai": "assistant",
                    "system": "system",
                    "tool": "tool",
                }.get(m.type, "user")
                conv.append({"role": role, "content": m.content if isinstance(m.content, str) else str(m.content)})
            text = loaded_model.chat(
                conv,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

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
