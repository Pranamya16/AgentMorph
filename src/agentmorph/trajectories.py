"""Trajectory schema + JSONL writer.

A `Trajectory` is the canonical record of one agent run on one scenario.
Stage 2 mutators compare trajectory pairs, so the schema is intentionally
framework-agnostic: anything a smolagents or LangGraph run produces collapses
into the same shape.

Design notes:
  * JSONL on disk, one trajectory per line. Cheap to append, cheap to resume.
  * No pydantic at serialization time — stdlib `json` + `dataclasses.asdict`
    keeps the writer safe to call inside a signal handler during Colab kills.
  * `tool_output` is stored as-is if JSON-serializable, else `repr()`.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator


class StepKind(str, Enum):
    """Kinds of events recorded in a trajectory.

    Kept minimal on purpose. Mutation rules in Stage 2 key on these.
    """

    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class TrajectoryStep:
    index: int
    kind: StepKind
    # For THOUGHT / FINAL_ANSWER / ERROR.
    content: str | None = None
    # For TOOL_CALL / TOOL_RESULT.
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output: Any = None
    tool_error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        # Replace unserializable outputs with their repr — never fail the log.
        try:
            json.dumps(d["tool_output"])
        except TypeError:
            d["tool_output"] = repr(d["tool_output"])
        return d


@dataclass
class Trajectory:
    trajectory_id: str
    scenario_id: str
    env_id: str
    model_id: str
    framework_id: str
    prompt: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        scenario_id: str,
        env_id: str,
        model_id: str,
        framework_id: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> "Trajectory":
        return cls(
            trajectory_id=uuid.uuid4().hex,
            scenario_id=scenario_id,
            env_id=env_id,
            model_id=model_id,
            framework_id=framework_id,
            prompt=prompt,
            metadata=dict(metadata or {}),
        )

    # Convenience appenders — keep agent-adapter code tiny.
    def add_thought(self, content: str) -> TrajectoryStep:
        return self._append(StepKind.THOUGHT, content=content)

    def add_tool_call(self, name: str, args: dict[str, Any]) -> TrajectoryStep:
        return self._append(StepKind.TOOL_CALL, tool_name=name, tool_args=dict(args))

    def add_tool_result(
        self, name: str, args: dict[str, Any], output: Any, error: str | None = None
    ) -> TrajectoryStep:
        return self._append(
            StepKind.TOOL_RESULT,
            tool_name=name,
            tool_args=dict(args),
            tool_output=output,
            tool_error=error,
        )

    def add_final_answer(self, content: str) -> TrajectoryStep:
        self.final_answer = content
        return self._append(StepKind.FINAL_ANSWER, content=content)

    def add_error(self, content: str) -> TrajectoryStep:
        return self._append(StepKind.ERROR, content=content)

    def finish(self) -> None:
        self.finished_at = time.time()

    @property
    def wall_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "scenario_id": self.scenario_id,
            "env_id": self.env_id,
            "model_id": self.model_id,
            "framework_id": self.framework_id,
            "prompt": self.prompt,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_seconds": self.wall_seconds,
            "metadata": self.metadata,
        }

    # Internal ---------------------------------------------------------------
    def _append(self, kind: StepKind, **fields: Any) -> TrajectoryStep:
        step = TrajectoryStep(index=len(self.steps), kind=kind, **fields)
        self.steps.append(step)
        return step


class TrajectoryWriter:
    """Appends trajectories to a JSONL file, flushing aggressively.

    Colab sessions die without warning. Every `write()` flushes and fsyncs so
    a killed session leaves a valid-up-to-last-trajectory file on disk.
    """

    def __init__(self, path: os.PathLike[str] | str, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, trajectory: Trajectory) -> None:
        self._fh.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()
        if self._fsync:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def iter_jsonl(path: os.PathLike[str] | str) -> Iterator[dict[str, Any]]:
    """Read back trajectories from disk (for resume / analysis)."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_trajectories(path: os.PathLike[str] | str) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def completed_scenario_ids(paths: Iterable[os.PathLike[str] | str]) -> set[str]:
    """Scenario IDs already recorded across one or more JSONL files.

    Used by the baseline runner to skip cells that are already done.
    """
    seen: set[str] = set()
    for p in paths:
        for row in iter_jsonl(p):
            sid = row.get("scenario_id")
            if sid:
                seen.add(sid)
    return seen
