"""AgentDojo environment adapter.

AgentDojo exposes multiple task suites (workspace, slack, travel, banking).
We expose each user task as one `Scenario` and build a fresh `ToolRegistry`
that proxies to the suite's python tools using our unified `Tool` type.

Notes:
  * Requires the `[agentdojo]` extra; imports are lazy.
  * AgentDojo has rewritten its public API several times. This adapter probes
    three import surfaces in order, so both older and newer releases work
    without code edits: `agentdojo.task_suite.get_suites`, `agentdojo.task_suite.task_suite.SUITES`,
    and the top-level `agentdojo.task_suites`. If none resolve, the adapter
    raises a clear error telling the user which version to install.
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
        # probe at call time so we don't bind to a single API shape.
        sig_local = inspect.signature(fn)
        if "env" in sig_local.parameters and env_state is not None:
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


# -- AgentDojo API discovery -------------------------------------------------


def _discover_suites() -> dict[str, Any]:
    """Return a mapping of {suite_name -> suite_obj} from whichever AgentDojo
    API shape is installed. Raises ImportError with a clear hint if none.

    Probes, in order:
      1. `agentdojo.task_suite.get_suites()` (older releases)
      2. `agentdojo.task_suite.task_suite.SUITES` (mid releases)
      3. `agentdojo.task_suites` module with one suite per submodule
    """
    try:
        import agentdojo  # noqa: F401  — just verifying the package is installed
    except ImportError as exc:
        raise ImportError(
            "AgentDojo is not installed. On Colab: "
            "`pip install -q agentdojo`  (or the `[agentdojo]` extra of agentmorph)."
        ) from exc

    # The official suites are registered as a side effect of importing
    # `agentdojo.default_suites`. Without this, `get_suites()` returns {}.
    try:
        import agentdojo.default_suites  # noqa: F401  — side-effect import
    except Exception:  # pragma: no cover
        pass

    # Shape 1 (most modern, post default_suites import): get_suites() returns
    # a populated dict of {name -> TaskSuite}.
    try:
        from agentdojo.task_suite import get_suites  # type: ignore
        suites = get_suites()
        if suites:
            return dict(suites)
    except Exception:  # pragma: no cover
        pass

    # Shape 1b: get_suite(name) for each of the four standard suites.
    try:
        from agentdojo.task_suite import get_suite  # type: ignore
        out: dict[str, Any] = {}
        for name in ("workspace", "slack", "travel", "banking"):
            try:
                s = get_suite(name)
                if s is not None:
                    out[name] = s
            except Exception:
                continue
        if out:
            return out
    except Exception:  # pragma: no cover
        pass

    # Shape 2: SUITES dict at agentdojo.task_suite.task_suite.
    try:
        from agentdojo.task_suite.task_suite import SUITES  # type: ignore
        if SUITES:
            return dict(SUITES)
    except Exception:  # pragma: no cover
        pass

    # Shape 3: agentdojo.task_suites.<name> modules each exposing a `task_suite`.
    try:
        import importlib
        import pkgutil
        import agentdojo.task_suites as _ts  # type: ignore
        out = {}
        for mod_info in pkgutil.iter_modules(_ts.__path__):
            name = mod_info.name
            try:
                m = importlib.import_module(f"agentdojo.task_suites.{name}")
                obj = getattr(m, "task_suite", None) or getattr(m, "SUITE", None)
                if obj is not None:
                    out[name] = obj
            except Exception:
                continue
        if out:
            return out
    except Exception:  # pragma: no cover
        pass

    raise ImportError(
        "AgentDojo is installed but no recognized suite API was found. "
        "Supported shapes: `agentdojo.task_suite.get_suites()`, "
        "`agentdojo.task_suite.task_suite.SUITES`, or `agentdojo.task_suites.<name>.task_suite`. "
        "If you see this, paste your agentdojo version so the adapter can be updated."
    )


def _init_environment(suite: Any) -> Any:
    """Best-effort environment initializer across AgentDojo versions."""
    for method_name in (
        "load_and_inject_default_environment",
        "load_environment",
        "environment",
        "init_environment",
    ):
        method = getattr(suite, method_name, None)
        if callable(method):
            try:
                # Some take an injection dict, some take nothing.
                sig = inspect.signature(method)
                if len(sig.parameters) >= 1:
                    return method({})
                return method()
            except TypeError:
                try:
                    return method()
                except Exception:
                    continue
            except Exception:
                continue
    return None


def _suite_tools(suite: Any) -> list[Any]:
    """AgentDojo exposes tools as `.tools` (list) or `.tools.values()` (dict)."""
    t = getattr(suite, "tools", None)
    if t is None:
        return []
    if isinstance(t, dict):
        return list(t.values())
    return list(t)


def _suite_user_tasks(suite: Any) -> dict[str, Any]:
    """AgentDojo user-task container name varies across releases."""
    for attr in ("user_tasks", "USER_TASKS", "tasks"):
        tasks = getattr(suite, attr, None)
        if tasks:
            if isinstance(tasks, dict):
                return dict(tasks)
            if isinstance(tasks, (list, tuple)):
                return {str(i): t for i, t in enumerate(tasks)}
    return {}


# -- Environment -------------------------------------------------------------


@dataclass
class AgentDojoEnvironment(Environment):
    """Adapter over a single AgentDojo task suite.

    Default `suite="workspace"` — it's the smallest suite and exercises a good
    spread of tool types (email, calendar, drive).
    """

    env_id: str = "agentdojo"
    suite: str = "workspace"
    max_tasks: int | None = None
    _suite_obj: Any | None = field(default=None, init=False, repr=False)
    _suite_registry: dict[str, Any] | None = field(default=None, init=False, repr=False)

    # -- Lazy-loaded suite ----------------------------------------------------

    def _get_suite(self) -> Any:
        if self._suite_obj is not None:
            return self._suite_obj
        if self._suite_registry is None:
            self._suite_registry = _discover_suites()
        if self.suite not in self._suite_registry:
            raise KeyError(
                f"unknown agentdojo suite {self.suite!r}; "
                f"available: {sorted(self._suite_registry)}"
            )
        self._suite_obj = self._suite_registry[self.suite]
        return self._suite_obj

    def available_suites(self) -> list[str]:
        """For debugging — list the suites the installed AgentDojo exposes."""
        if self._suite_registry is None:
            self._suite_registry = _discover_suites()
        return sorted(self._suite_registry)

    # -- Environment protocol -------------------------------------------------

    def scenarios(self) -> Iterator[Scenario]:
        suite = self._get_suite()
        user_tasks = _suite_user_tasks(suite)
        items = list(user_tasks.items())
        if self.max_tasks is not None:
            items = items[: self.max_tasks]
        for task_id, task in items:
            prompt = (
                getattr(task, "PROMPT", None)
                or getattr(task, "prompt", None)
                or getattr(task, "user_message", None)
                or ""
            )
            yield Scenario(
                id=f"agentdojo/{self.suite}/{task_id}",
                env_id=self.env_id,
                prompt=str(prompt),
                metadata={"suite": self.suite, "task_id": str(task_id)},
            )

    def reset(self, scenario: Scenario) -> ScenarioBundle:
        suite = self._get_suite()
        env_state = _init_environment(suite)

        registry = ToolRegistry()
        for fn in _suite_tools(suite):
            try:
                registry.register(
                    _callable_to_tool(fn, env_state=env_state, category=self.suite)
                )
            except Exception:
                # Skip tools that don't reflect cleanly — keeps the registry
                # usable even when AgentDojo adds new tool shapes.
                continue

        return ScenarioBundle(scenario=scenario, registry=registry, state=env_state)
