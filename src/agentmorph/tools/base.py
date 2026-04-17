"""Unified tool abstraction.

One `Tool` definition feeds both smolagents and LangGraph adapters so Stage 2
mutators operate on a single canonical schema regardless of framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import jsonschema


JsonSchema = dict[str, Any]


@dataclass(frozen=True)
class Tool:
    """A tool with a JSON-schema signature and a Python implementation.

    `parameters` follows JSON Schema (same shape the OpenAI / Anthropic function
    calling APIs use): a top-level object with `properties` and `required`.
    """

    name: str
    description: str
    parameters: JsonSchema
    func: Callable[..., Any]
    read_only: bool = False
    category: str = "misc"

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"Tool name must be a valid identifier: {self.name!r}")
        if self.parameters.get("type") != "object":
            raise ValueError(f"Tool {self.name} parameters must be a JSON-Schema object")

    def validate_args(self, args: dict[str, Any]) -> None:
        jsonschema.validate(args, self.parameters)

    def invoke(self, args: dict[str, Any]) -> Any:
        self.validate_args(args)
        return self.func(**args)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    name: str
    arguments: dict[str, Any]
    output: Any
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ToolRegistry:
    """Named collection of tools.

    Order is preserved — mutators such as tool-order invariance can read
    `names()` to produce permutations.
    """

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.tools[tool.name] = tool

    def extend(self, tools: Iterable[Tool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool:
        return self.tools[name]

    def names(self) -> list[str]:
        return list(self.tools.keys())

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools.values())

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self.tools:
            return ToolResult(name, arguments, output=None, error=f"unknown tool: {name}")
        try:
            out = self.tools[name].invoke(arguments)
            return ToolResult(name, arguments, output=out)
        except Exception as exc:
            return ToolResult(name, arguments, output=None, error=f"{type(exc).__name__}: {exc}")

    def openai_schema(self) -> list[dict[str, Any]]:
        """Function-calling schema (OpenAI / Anthropic shape)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools.values()
        ]
