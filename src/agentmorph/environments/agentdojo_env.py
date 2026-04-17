"""AgentDojo environment adapter.

AgentDojo exposes multiple task suites (workspace, travel, banking, slack).
We expose each user task as one `Scenario`, and we build a fresh
`ToolRegistry` that proxies to the suite's python tools using our unified
`Tool` type.

Notes:
  * Requires the `[agentdojo]` extra; imports are lazy.
  * AgentDojo's tool signatures are Python callables, not JSON Schema, so we
    reflect on annotations to derive a minimal schema. Mutators that need a
    tight schema (e.g. schema-paraphrase) should skip AgentDojo tools whose
    signatures we couldn't introspect.
  * For Stage 1 we run the *user tasks only* (no injection tasks). The
    injection side comes in later stages once we're studying prompt-injection
    transfer, not agent robustness.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Iterator

from agentmorph.environments.base import Environment, Scenario, ScenarioBundle
from agentmorph.tools.base import Tool, ToolRegistry


# -- Python type → JSON Schema -----------------------------------------------

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_jsonschema(annotation: Any) -> dict[str, Any]:
    origin = getattr(annotation, "__origin__", None)
    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}
    if origin in (list, tuple, set):
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _callable_to_tool(fn: Any, *, env_state: Any, category: str) -> Tool:
    """Wrap an AgentDojo tool function into our `Tool`."""
    sig = inspect.signature(fn)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        # Skip the environment / state params AgentDojo injects on the fly.
        if name in {"env", "state", "context"}:
            continue
        schema = _annotation_to_jsonschema(param.annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)
        properties[name] = schema

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    def _call(**kwargs: Any) -> Any:
        # AgentDojo tools take either `env=` or bare args depending on version;
        # we probe and pass whatever the signature accepts.
        sig_local = inspect.signature(fn)
        if "env" in sig_local.parameters:
            return fn(env=env_state, **kwargs)
        return fn(**kwargs)

    return Tool(
        name=getattr(fn, "__name__", "tool"),
        description=(inspect.getdoc(fn) or "").strip().splitlines()[0][:500],
        parameters=parameters,
        func=_call,
        read_only=False,       # AgentDojo doesn't surface this; leave conservative.
        category=category,
    )


# -- Environment -------------------------------------------------------------


@dataclass
class AgentDojoEnvironment(Environment):
    """Adapter over a single AgentDojo task suite.

    Use `suite="workspace"` by default — it's the smallest suite and exercises
    a good spread of tool types (email, calendar, drive).
    """

    env_id: str = "agentdojo"
    suite: str = "workspace"
    max_tasks: int | None = None
    _suite_obj: Any | None = field(default=None, init=False, repr=False)

    # -- Lazy-loaded suite ----------------------------------------------------

    def _get_suite(self) -> Any:
        if self._suite_obj is not None:
            return self._suite_obj
        try:
            from agentdojo.task_suite import get_suites  # type: ignore
            suites = get_suites()
        except ImportError as exc:  # pragma: no cover — extra not installed
            raise ImportError(
                "AgentDojo is not installed. `pip install agentmorph[agentdojo]`."
            ) from exc
        if self.suite not in suites:
            raise KeyError(f"unknown agentdojo suite {self.suite!r}; have {list(suites)}")
        self._suite_obj = suites[self.suite]
        return self._suite_obj

    # -- Environment protocol -------------------------------------------------

    def scenarios(self) -> Iterator[Scenario]:
        suite = self._get_suite()
        user_tasks = getattr(suite, "user_tasks", {})
        items = list(user_tasks.items())
        if self.max_tasks is not None:
            items = items[: self.max_tasks]
        for task_id, task in items:
            prompt = getattr(task, "PROMPT", None) or getattr(task, "prompt", None) or ""
            yield Scenario(
                id=f"agentdojo/{self.suite}/{task_id}",
                env_id=self.env_id,
                prompt=prompt,
                metadata={"suite": self.suite, "task_id": task_id},
            )

    def reset(self, scenario: Scenario) -> ScenarioBundle:
        suite = self._get_suite()
        # AgentDojo environments are constructed via `suite.load_and_inject_default_environment(...)`
        # on recent versions. We call whichever initializer is available.
        env_state: Any
        if hasattr(suite, "load_and_inject_default_environment"):
            env_state = suite.load_and_inject_default_environment({})
        elif hasattr(suite, "environment"):
            env_state = suite.environment()
        else:
            env_state = None

        tools = getattr(suite, "tools", None) or []
        registry = ToolRegistry()
        for fn in tools:
            try:
                registry.register(
                    _callable_to_tool(fn, env_state=env_state, category=self.suite)
                )
            except Exception:
                # Skip tools that don't reflect cleanly — keeps the registry
                # usable even when AgentDojo adds new tool shapes.
                continue

        return ScenarioBundle(scenario=scenario, registry=registry, state=env_state)
