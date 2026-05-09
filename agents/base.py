"""Shared agent infrastructure.

Two pieces:

    GymClient   — thin HTTP client that talks to the FastAPI server.
                  Knows how to reset, fetch the DOM summary, apply UI actions,
                  and run the verifier.

    Trajectory  — dataclass that records every step of an episode so the
                  whole run is replayable, inspectable, and (later) usable
                  as training data.

The point of this module is to make any new agent (Claude tool-use, Qwen
DOM-action, Computer Use, hand-coded oracle, ...) trivial to write — just
implement `run(client) -> Trajectory` and use these helpers.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #

@dataclass
class Step:
    step_idx: int
    action_kind: str                     # "dispatch", "send_message", "click", ...
    action_args: dict[str, Any]
    server_result: dict[str, Any]        # what /api/apply returned
    state_after: dict[str, Any]          # snapshot after the action
    reasoning: str | None = None         # the agent's private rationale
    raw_model_output: str | None = None  # full text the model emitted


@dataclass
class Trajectory:
    """Full record of one episode for one agent on one (task, seed)."""

    episode_id: str
    task_id: str
    seed: int
    agent_name: str
    started_at: float
    finished_at: float | None = None
    initial_state: dict[str, Any] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    verifier_result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None              # set if the agent itself crashed

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id, "seed": self.seed,
            "agent_name": self.agent_name,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "steps": [asdict(s) for s in self.steps],
            "verifier_result": self.verifier_result,
            "error": self.error,
        }

    @property
    def success(self) -> bool:
        return bool(self.verifier_result.get("success", False))

    @property
    def score(self) -> float:
        return float(self.verifier_result.get("score", 0.0))

    @property
    def error_category(self) -> str:
        return str(self.verifier_result.get("error_category", "unknown"))


# --------------------------------------------------------------------------- #
# GymClient — HTTP wrapper around the FastAPI server
# --------------------------------------------------------------------------- #

class GymClient:
    """Stateful client. Use one instance per episode; .reset() starts a new one."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ----- lifecycle -----
    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        r = self._http.post("/api/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def state(self) -> dict[str, Any]:
        r = self._http.get("/api/state")
        r.raise_for_status()
        return r.json()

    def initial(self) -> dict[str, Any]:
        r = self._http.get("/api/initial")
        r.raise_for_status()
        return r.json()

    def dom_summary(self) -> dict[str, Any]:
        r = self._http.get("/api/dom_summary")
        r.raise_for_status()
        return r.json()

    # ----- actions -----
    def apply(self, kind: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        r = self._http.post("/api/apply", json={"kind": kind, "args": args or {}})
        r.raise_for_status()
        return r.json()

    def finish(self) -> dict[str, Any]:
        return self.apply("finish", {})

    def verify(self) -> dict[str, Any]:
        r = self._http.post("/api/verify")
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------------- #
# Helpers for agents to record actions
# --------------------------------------------------------------------------- #

def new_trajectory(task_id: str, seed: int, agent_name: str) -> Trajectory:
    return Trajectory(
        episode_id=str(uuid.uuid4())[:8],
        task_id=task_id,
        seed=seed,
        agent_name=agent_name,
        started_at=time.time(),
    )


def save_trajectory(traj: Trajectory, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_task = traj.task_id.replace("/", "_")
    path = out_dir / f"{safe_task}__{traj.seed}__{traj.episode_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        json.dump(traj.to_json(), f, indent=2)
    return path
