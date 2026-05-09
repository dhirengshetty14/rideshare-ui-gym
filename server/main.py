"""FastAPI app — the gym's serving layer.

Endpoints:

    GET  /                             serves the single-page UI
    POST /api/reset                    reset state for (task_id, seed)
    GET  /api/state                    current state snapshot (JSON)
    GET  /api/initial                  the initial-state snapshot (for verifier)
    POST /api/apply                    apply one mutation (UI events route here)
    POST /api/verify                   run the verifier, return result
    GET  /api/tasks                    list registered task ids
    GET  /api/dom_summary              structured DOM-style summary for agents

The two views into state — `/api/state` for the UI to render, and
`/api/dom_summary` for non-vision agents — are kept separate on purpose:
the UI version is verbose; the agent version is compact and emphasizes
data-test-id targets the agent can refer to.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server import state as state_mod
from server.state import GymState
from server.tasks import TASKS, make_task
from server.verifiers import verify


# --------------------------------------------------------------------------- #
# In-memory session — single-tenant; this is a dev/demo gym, not a service.
# --------------------------------------------------------------------------- #

class Session:
    """Holds the current world plus a snapshot of its initial state so the
    verifier can compare. Reset between episodes."""
    initial: GymState | None = None
    current: GymState | None = None


SESSION = Session()


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class ResetRequest(BaseModel):
    task_id: str
    seed: int = 0


class ApplyRequest(BaseModel):
    """One UI-level action. The `kind` field dispatches to the right
    state-mutator. Each kind has its own typed args, validated below."""
    kind: Literal[
        "dispatch", "send_message", "block_account", "set_surge", "finish",
    ]
    args: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "ui"

app = FastAPI(title="Rideshare UI Gym", version="0.1.0")
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


def _require_session() -> GymState:
    if SESSION.current is None:
        raise HTTPException(
            status_code=409,
            detail="No active episode. POST /api/reset first.",
        )
    return SESSION.current


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


# --------------------------------------------------------------------------- #
# Episode lifecycle
# --------------------------------------------------------------------------- #

@app.get("/api/tasks")
def list_tasks() -> dict[str, Any]:
    return {"tasks": list(TASKS.keys())}


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    if req.task_id not in TASKS:
        raise HTTPException(404, f"unknown task '{req.task_id}'")
    fresh = make_task(req.task_id, req.seed)
    SESSION.initial = copy.deepcopy(fresh)
    SESSION.current = fresh
    return {
        "ok": True, "task_id": req.task_id, "seed": req.seed,
        "state": fresh.to_json(),
    }


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return _require_session().to_json()


@app.get("/api/initial")
def get_initial() -> dict[str, Any]:
    if SESSION.initial is None:
        raise HTTPException(409, "no episode")
    return SESSION.initial.to_json()


# --------------------------------------------------------------------------- #
# Mutations
# --------------------------------------------------------------------------- #

@app.post("/api/apply")
def apply(req: ApplyRequest) -> dict[str, Any]:
    s = _require_session()
    if s.finished:
        raise HTTPException(409, "episode already finished")

    args = req.args
    s.step += 1               # one UI event = one logical step
    if req.kind == "dispatch":
        result = state_mod.dispatch(
            s, driver_id=str(args.get("driver_id", "")),
            rider_id=str(args.get("rider_id", "")),
        )
    elif req.kind == "send_message":
        result = state_mod.send_message(
            s,
            recipient_id=str(args.get("recipient_id", "")),
            recipient_kind=args.get("recipient_kind", "rider"),
            body=str(args.get("body", "")),
        )
    elif req.kind == "block_account":
        try:
            account_id = int(args.get("account_id", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "block_account requires int account_id")
        result = state_mod.block_account(
            s, account_id=account_id, reason=str(args.get("reason", "")),
        )
    elif req.kind == "set_surge":
        try:
            multiplier = float(args.get("multiplier", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "set_surge requires float multiplier")
        result = state_mod.set_surge(
            s, zone=str(args.get("zone", "")), multiplier=multiplier,
        )
    elif req.kind == "finish":
        result = state_mod.finish(s)
    else:
        raise HTTPException(400, f"unknown action kind {req.kind}")

    return {
        "ok": result.get("ok", True),
        "result": result,
        "state": s.to_json(),
    }


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #

@app.post("/api/verify")
def run_verifier() -> dict[str, Any]:
    if SESSION.initial is None or SESSION.current is None:
        raise HTTPException(409, "no episode")
    result = verify(SESSION.current.task_id, SESSION.initial, SESSION.current)
    return {"verifier": result, "task_id": SESSION.current.task_id}


# --------------------------------------------------------------------------- #
# Agent-friendly compact view
# --------------------------------------------------------------------------- #

@app.get("/api/dom_summary")
def dom_summary() -> dict[str, Any]:
    """Compact, structured snapshot the agent can consume directly. Each
    interactable element is listed with its data-test-id and current value /
    label so the agent can reason about what to click without parsing HTML.
    Mirrors what a real DOM tree exposes but with all the noise stripped."""
    s = _require_session()

    elements: list[dict[str, Any]] = []

    # Task banner
    elements.append({
        "id": "task-banner",
        "kind": "label",
        "text": f"[{s.task_difficulty.upper()}] {s.task_id}",
        "brief": s.task_brief,
        "step": s.step,
        "finished": s.finished,
    })

    # Driver rows
    for d in sorted(s.drivers.values(), key=lambda d: d.id):
        elements.append({
            "id": f"driver-row-{d.id}",
            "kind": "driver_row",
            "driver_id": d.id, "name": d.name,
            "location": list(d.location), "status": d.status,
            "rating": d.rating,
            "actions": [
                {"id": f"btn-dispatch-{d.id}", "kind": "button",
                 "label": "Dispatch", "enabled": d.status == "idle"},
                {"id": f"btn-msg-driver-{d.id}", "kind": "button",
                 "label": "Send message", "enabled": True},
            ],
        })

    # Rider rows
    for r in sorted(s.riders.values(), key=lambda r: r.id):
        elements.append({
            "id": f"rider-row-{r.id}",
            "kind": "rider_row",
            "rider_id": r.id,
            "pickup": list(r.pickup), "pickup_label": r.pickup_label,
            "waited_minutes": r.waited_minutes, "status": r.status,
            "priority": r.priority,
            "actions": [
                {"id": f"btn-msg-rider-{r.id}", "kind": "button",
                 "label": "Send message", "enabled": True},
            ],
        })

    # Fraud panel
    for fa in sorted(s.fraud_alerts.values(), key=lambda fa: fa.account_id):
        elements.append({
            "id": f"fraud-row-{fa.account_id}",
            "kind": "fraud_row",
            "account_id": fa.account_id,
            "severity": fa.severity, "pattern": fa.pattern,
            "evidence": fa.evidence,
            "actions": [
                {"id": f"btn-block-{fa.account_id}", "kind": "button",
                 "label": "Block", "enabled": True},
            ],
        })

    # Trips and messages — read-only
    for t in s.trips.values():
        elements.append({
            "id": f"trip-{t.id}", "kind": "trip",
            "trip_id": t.id, "driver_id": t.driver_id,
            "rider_id": t.rider_id, "eta_minutes": t.eta_minutes,
        })
    for m in s.messages:
        elements.append({
            "id": f"message-{m.id}", "kind": "message",
            "recipient_id": m.recipient_id, "recipient_kind": m.recipient_kind,
            "body": m.body, "step": m.sent_at_step,
        })

    # Submit button is always present
    elements.append({
        "id": "btn-submit-task", "kind": "button", "label": "Submit task",
        "enabled": not s.finished,
    })

    return {
        "task_id": s.task_id, "step": s.step, "finished": s.finished,
        "task_brief": s.task_brief, "task_difficulty": s.task_difficulty,
        "elements": elements,
    }
