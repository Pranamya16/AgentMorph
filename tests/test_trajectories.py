"""Tests for the Trajectory schema + JSONL writer."""

from __future__ import annotations

import json
from pathlib import Path

from agentmorph.trajectories import (
    StepKind,
    Trajectory,
    TrajectoryWriter,
    completed_scenario_ids,
    iter_jsonl,
    read_trajectories,
)


def _sample_trajectory(scenario_id: str = "s1") -> Trajectory:
    t = Trajectory.new(
        scenario_id=scenario_id,
        env_id="ecommerce",
        model_id="Llama-3.2-3B",
        framework_id="native",
        prompt="hello",
    )
    t.add_thought("thinking...")
    t.add_tool_call("search_products", {"query": "kettle"})
    t.add_tool_result("search_products", {"query": "kettle"}, output=[{"id": "P1"}])
    t.add_final_answer("done")
    t.finish()
    return t


def test_trajectory_collects_steps_in_order() -> None:
    t = _sample_trajectory()
    kinds = [s.kind for s in t.steps]
    assert kinds == [
        StepKind.THOUGHT,
        StepKind.TOOL_CALL,
        StepKind.TOOL_RESULT,
        StepKind.FINAL_ANSWER,
    ]
    assert t.final_answer == "done"
    assert t.wall_seconds is not None and t.wall_seconds >= 0


def test_trajectory_serializes_to_dict_roundtrip() -> None:
    t = _sample_trajectory()
    d = t.to_dict()
    # Must be json-serializable.
    serialized = json.dumps(d)
    data = json.loads(serialized)
    assert data["scenario_id"] == "s1"
    assert data["steps"][0]["kind"] == "thought"
    assert data["steps"][2]["tool_output"] == [{"id": "P1"}]


def test_trajectory_writer_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"
    with TrajectoryWriter(path, fsync=False) as w:
        w.write(_sample_trajectory("s1"))
        w.write(_sample_trajectory("s2"))

    assert path.exists()
    rows = read_trajectories(path)
    assert [r["scenario_id"] for r in rows] == ["s1", "s2"]


def test_completed_scenario_ids_across_files(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    with TrajectoryWriter(a, fsync=False) as w:
        w.write(_sample_trajectory("s1"))
    with TrajectoryWriter(b, fsync=False) as w:
        w.write(_sample_trajectory("s2"))
        w.write(_sample_trajectory("s3"))

    assert completed_scenario_ids([a, b]) == {"s1", "s2", "s3"}


def test_unserializable_tool_output_falls_back_to_repr() -> None:
    t = Trajectory.new(
        scenario_id="s", env_id="e", model_id="m", framework_id="f", prompt="p"
    )

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    t.add_tool_result("x", {}, output=Opaque())
    # Must not raise; must be JSON-serializable.
    serialized = json.dumps(t.to_dict())
    assert "<opaque>" in serialized


def test_iter_jsonl_missing_file_returns_empty(tmp_path: Path) -> None:
    assert list(iter_jsonl(tmp_path / "nope.jsonl")) == []
