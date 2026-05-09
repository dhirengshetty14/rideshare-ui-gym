"""In-memory state model for the rideshare dispatch UI gym.

The state is intentionally small and JSON-serializable — no database. The
backend mutates it in response to UI events, the frontend reads it on every
poll, and the verifier inspects the final snapshot to score the episode.

Key types:
    Driver       — one driver in the fleet (location, status, rating)
    Rider        — a rider currently waiting / queued
    Trip         — an active assignment of a driver to a rider
    FraudAlert   — a flagged account awaiting operator review
    Message      — a logged message sent to a rider or driver
    AuditLogEntry — a record of an operator action (block, escalate, etc.)
    GymState     — the whole world: a flat snapshot of everything above

Geography is a 10x10 toy grid. Distances are Manhattan (|dx|+|dy|). That's
enough fidelity to make "closest driver" non-trivial without dragging in a
mapping library.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Atomic entities
# --------------------------------------------------------------------------- #

DriverStatus = Literal["idle", "en_route", "offline", "blocked"]
RiderStatus = Literal["waiting", "assigned", "completed", "cancelled"]


@dataclass
class Driver:
    id: str
    name: str
    location: tuple[int, int]            # (x, y) on the 10x10 grid
    status: DriverStatus = "idle"
    rating: float = 4.5                  # 0–5
    completed_trips_today: int = 0
    flagged: bool = False                # set by fraud-detection panel


@dataclass
class Rider:
    id: str
    pickup: tuple[int, int]
    pickup_label: str                    # human-readable, e.g. "JFK Terminal 4"
    waited_minutes: int = 0
    status: RiderStatus = "waiting"
    priority: bool = False               # e.g. accident / VIP


@dataclass
class Trip:
    id: str
    driver_id: str
    rider_id: str
    eta_minutes: int


@dataclass
class FraudAlert:
    account_id: int                      # the suspect driver ID, but as int
                                         # so the IDs in the fraud panel are
                                         # different from the dispatch panel.
    severity: Literal["low", "medium", "high"]
    pattern: str                         # "many_short_trips", "fake_pickups", ...
    evidence: list[str]                  # bullet points shown in the modal


@dataclass
class Message:
    id: str
    recipient_id: str                    # driver_id or rider_id
    recipient_kind: Literal["driver", "rider"]
    body: str
    sent_at_step: int                    # which env step this was logged at


@dataclass
class AuditLogEntry:
    step: int
    action: str                          # "block", "escalate", "dispatch", ...
    target_id: str                       # account_id or driver_id
    reason: str                          # operator-supplied or system reason
    operator: str = "agent"              # always "agent" in this gym


# --------------------------------------------------------------------------- #
# Whole-world state
# --------------------------------------------------------------------------- #

@dataclass
class GymState:
    """The whole world the agent is operating in. Everything verifiers ever
    need to score an episode is contained here."""

    task_id: str = ""
    seed: int = 0
    step: int = 0                        # number of agent actions applied
    finished: bool = False               # True after the agent submits

    # Top-level entities — kept as dicts keyed by id for O(1) lookup.
    drivers: dict[str, Driver] = field(default_factory=dict)
    riders: dict[str, Rider] = field(default_factory=dict)
    trips: dict[str, Trip] = field(default_factory=dict)
    fraud_alerts: dict[int, FraudAlert] = field(default_factory=dict)

    # Logs accumulate across the episode and are inspected by verifiers.
    messages: list[Message] = field(default_factory=list)
    audit_log: list[AuditLogEntry] = field(default_factory=list)

    # Surge zones — toy version: just a multiplier on a named region.
    surge: dict[str, float] = field(default_factory=dict)

    # The active task's brief, shown to the agent in the UI banner.
    task_brief: str = ""
    task_difficulty: Literal["easy", "medium", "hard"] = "easy"

    def to_json(self) -> dict[str, Any]:
        """Serialize to plain dicts for the FastAPI response. Tuples become
        lists automatically."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "step": self.step,
            "finished": self.finished,
            "task_brief": self.task_brief,
            "task_difficulty": self.task_difficulty,
            "drivers": {k: asdict(v) for k, v in self.drivers.items()},
            "riders": {k: asdict(v) for k, v in self.riders.items()},
            "trips": {k: asdict(v) for k, v in self.trips.items()},
            "fraud_alerts": {str(k): asdict(v) for k, v in self.fraud_alerts.items()},
            "messages": [asdict(m) for m in self.messages],
            "audit_log": [asdict(e) for e in self.audit_log],
            "surge": dict(self.surge),
        }


# --------------------------------------------------------------------------- #
# Mutations — these are the *only* legal ways to change state. The FastAPI
# /apply endpoint dispatches to one of these. Verifiers later read the
# resulting state to grade the episode.
# --------------------------------------------------------------------------- #

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dispatch(state: GymState, driver_id: str, rider_id: str) -> dict[str, Any]:
    """Assign a driver to a rider, creating a Trip and updating both sides."""
    if driver_id not in state.drivers:
        return {"ok": False, "error": f"unknown driver {driver_id}"}
    if rider_id not in state.riders:
        return {"ok": False, "error": f"unknown rider {rider_id}"}

    driver = state.drivers[driver_id]
    rider = state.riders[rider_id]

    if driver.status != "idle":
        return {"ok": False, "error": f"driver {driver_id} is {driver.status}, not idle"}
    if rider.status != "waiting":
        return {"ok": False, "error": f"rider {rider_id} is {rider.status}, not waiting"}

    eta = manhattan(driver.location, rider.pickup) * 2  # toy: 2 min per grid unit
    trip_id = f"T{len(state.trips) + 1:04d}"
    state.trips[trip_id] = Trip(
        id=trip_id, driver_id=driver_id, rider_id=rider_id, eta_minutes=eta,
    )
    driver.status = "en_route"
    rider.status = "assigned"

    state.audit_log.append(AuditLogEntry(
        step=state.step, action="dispatch",
        target_id=driver_id,
        reason=f"matched to {rider_id}, eta={eta}min",
    ))
    return {"ok": True, "trip_id": trip_id, "eta_minutes": eta}


def send_message(
    state: GymState,
    recipient_id: str,
    recipient_kind: Literal["driver", "rider"],
    body: str,
) -> dict[str, Any]:
    if recipient_kind == "driver" and recipient_id not in state.drivers:
        return {"ok": False, "error": f"unknown driver {recipient_id}"}
    if recipient_kind == "rider" and recipient_id not in state.riders:
        return {"ok": False, "error": f"unknown rider {recipient_id}"}
    if not body.strip():
        return {"ok": False, "error": "empty message"}

    msg_id = f"M{len(state.messages) + 1:04d}"
    state.messages.append(Message(
        id=msg_id,
        recipient_id=recipient_id,
        recipient_kind=recipient_kind,
        body=body.strip(),
        sent_at_step=state.step,
    ))
    return {"ok": True, "message_id": msg_id}


def block_account(state: GymState, account_id: int, reason: str) -> dict[str, Any]:
    if account_id not in state.fraud_alerts:
        return {"ok": False, "error": f"no alert for account {account_id}"}
    if not reason.strip():
        return {"ok": False, "error": "block requires a reason"}

    # Also flag the corresponding driver if there's one with a matching numeric ID.
    matching_driver_id = f"D{account_id}"
    if matching_driver_id in state.drivers:
        state.drivers[matching_driver_id].status = "blocked"
        state.drivers[matching_driver_id].flagged = True

    state.audit_log.append(AuditLogEntry(
        step=state.step, action="block",
        target_id=str(account_id), reason=reason.strip(),
    ))
    return {"ok": True}


def set_surge(state: GymState, zone: str, multiplier: float) -> dict[str, Any]:
    if multiplier < 1.0 or multiplier > 5.0:
        return {"ok": False, "error": f"surge {multiplier} out of range [1.0, 5.0]"}
    state.surge[zone] = float(multiplier)
    state.audit_log.append(AuditLogEntry(
        step=state.step, action="set_surge",
        target_id=zone, reason=f"multiplier={multiplier}",
    ))
    return {"ok": True}


def finish(state: GymState) -> dict[str, Any]:
    """Agent declares itself done. The verifier scores from here."""
    state.finished = True
    state.audit_log.append(AuditLogEntry(
        step=state.step, action="finish", target_id="-", reason="agent submitted",
    ))
    return {"ok": True}
