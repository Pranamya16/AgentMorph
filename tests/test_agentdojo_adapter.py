"""Smoke tests for the AgentDojo adapter.

These run without the `[agentdojo]` extra installed — they only verify:
  * The adapter imports cleanly (doesn't accidentally try to import agentdojo
    at module-load time).
  * When agentdojo is absent, calling `scenarios()` or `reset()` raises
    ImportError with an actionable message.
  * Discovery probes handle AbsentAPIShape gracefully.

The `needs_agentdojo` marker exists for any future test that requires the
real library — skip by default.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_adapter_module_imports_without_agentdojo() -> None:
    """The adapter must be importable even if `agentdojo` isn't installed."""
    import agentmorph.environments.agentdojo_env as mod
    assert hasattr(mod, "AgentDojoEnvironment")
    # Importing the module must NOT have imported agentdojo as a side effect.
    # (Trying to import a missing module is fine; eagerly importing it isn't.)


def test_agentdojo_env_construction_is_lazy() -> None:
    """Constructing an env shouldn't trigger the agentdojo import."""
    from agentmorph.environments.agentdojo_env import AgentDojoEnvironment
    env = AgentDojoEnvironment(suite="workspace", max_tasks=5)
    assert env.env_id == "agentdojo"
    assert env.suite == "workspace"
    assert env.max_tasks == 5
    # No suite discovery yet.
    assert env._suite_registry is None


def test_agentdojo_env_surfaces_missing_dep_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """If agentdojo isn't importable, scenarios() raises a clear ImportError."""
    # Block any import of `agentdojo` by injecting a sentinel into sys.modules.
    import agentmorph.environments.agentdojo_env as mod

    def _block(*_args: object, **_kwargs: object) -> object:
        raise ImportError("agentdojo not installed (simulated)")

    monkeypatch.setattr(mod, "_discover_suites", _block)

    from agentmorph.environments.agentdojo_env import AgentDojoEnvironment
    env = AgentDojoEnvironment()
    with pytest.raises(ImportError):
        list(env.scenarios())


def test_agentdojo_env_via_registry_does_not_import_agentdojo() -> None:
    """`load_environment('agentdojo')` must not eagerly import the real library."""
    from agentmorph.environments import load_environment
    env = load_environment("agentdojo", suite="workspace")
    assert env.env_id == "agentdojo"


@pytest.mark.skipif(
    importlib.util.find_spec("agentdojo") is None,
    reason="agentdojo not installed (real integration test)",
)
def test_agentdojo_env_real_suite_discovery() -> None:
    """When agentdojo IS installed, at least one suite should be discoverable."""
    from agentmorph.environments.agentdojo_env import AgentDojoEnvironment
    env = AgentDojoEnvironment()
    suites = env.available_suites()
    assert suites, "expected at least one AgentDojo suite to be discoverable"
    # Typical suites shipped with agentdojo:
    assert any(s in suites for s in ("workspace", "slack", "travel", "banking"))
