"""Task definitions — initial state factories for each scenario.

A task is a function `make(seed: int) -> GymState` that builds a fresh
starting world. Different seeds give different driver positions, rider
counts, etc., but the *goal* is constant per task (so the verifier knows
what to check).

Three tasks span the difficulty curve:

    easy/match_single_ride       1 single dispatch, 1 rider, 5 drivers
    medium/dispatch_with_message dispatch + send a custom rider message
    hard/fraud_review_and_block  navigate to fraud panel, review, block
                                 a specific account with a logged reason

Adding more tasks later is just: add a function here + a verifier in
verifiers.py + register it in TASKS.
"""

from __future__ import annotations

import random
from server.state import (
    Driver, FraudAlert, GymState, Rider,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

DRIVER_NAMES = [
    "Alice", "Bob", "Carlos", "Dana", "Elena", "Faisal", "Grace", "Hiro",
    "Ines", "Jamal", "Kira", "Luis", "Maya", "Nadia", "Omar", "Priya",
]

PICKUP_LABELS = [
    "JFK Terminal 4", "Penn Station", "Times Square", "Brooklyn Bridge Park",
    "Columbia University", "Lower East Side", "Williamsburg", "Soho",
    "Battery Park", "Madison Square Garden",
]


def _fresh_drivers(rng: random.Random, n: int) -> dict[str, Driver]:
    drivers: dict[str, Driver] = {}
    for i in range(n):
        did = f"D{1000 + i + 1}"          # D1001, D1002, ...
        drivers[did] = Driver(
            id=did,
            name=rng.choice(DRIVER_NAMES),
            location=(rng.randint(0, 9), rng.randint(0, 9)),
            status="idle",
            rating=round(rng.uniform(3.8, 4.95), 2),
            completed_trips_today=rng.randint(0, 6),
        )
    return drivers


def _fresh_riders(
    rng: random.Random, n: int, ids: list[str] | None = None,
) -> dict[str, Rider]:
    riders: dict[str, Rider] = {}
    for i in range(n):
        rid = (ids[i] if ids else f"R{i + 1}")
        riders[rid] = Rider(
            id=rid,
            pickup=(rng.randint(0, 9), rng.randint(0, 9)),
            pickup_label=rng.choice(PICKUP_LABELS),
            waited_minutes=rng.randint(1, 8),
            status="waiting",
        )
    return riders


# --------------------------------------------------------------------------- #
# Task 1: easy / match_single_ride
# --------------------------------------------------------------------------- #

EASY_BRIEF = (
    "Rider R1 is waiting. Match them to the closest IDLE driver. "
    "(Hint: each driver row shows their (x, y) location; rider R1's pickup is "
    "shown in the rider queue.) When you're done, click 'Submit task'."
)


def make_easy_match_single_ride(seed: int) -> GymState:
    rng = random.Random(seed)
    drivers = _fresh_drivers(rng, 5)

    # One of the drivers is offline so the agent has to skip them.
    offline_driver_id = rng.choice(list(drivers.keys()))
    drivers[offline_driver_id].status = "offline"

    riders = _fresh_riders(rng, 1, ids=["R1"])

    return GymState(
        task_id="easy/match_single_ride",
        seed=seed,
        task_brief=EASY_BRIEF,
        task_difficulty="easy",
        drivers=drivers,
        riders=riders,
    )


# --------------------------------------------------------------------------- #
# Task 2: medium / dispatch_with_message
# --------------------------------------------------------------------------- #

MEDIUM_BRIEF = (
    "Rider R3 is a priority passenger (medical urgency). "
    "Two things to do: (1) dispatch the closest IDLE driver to R3, and "
    "(2) send R3 a confirmation message that includes the assigned driver's "
    "name and the ETA in minutes. Then submit."
)


def make_medium_dispatch_with_message(seed: int) -> GymState:
    rng = random.Random(seed)
    drivers = _fresh_drivers(rng, 8)

    # Two drivers offline so the closest *idle* driver isn't always D1001.
    for did in rng.sample(list(drivers.keys()), 2):
        drivers[did].status = "offline"

    # 3 riders; R3 is the priority one.
    riders = _fresh_riders(rng, 3, ids=["R1", "R2", "R3"])
    riders["R3"].priority = True
    riders["R3"].pickup_label = "Mount Sinai Hospital — main entrance"
    riders["R3"].waited_minutes = max(riders["R3"].waited_minutes, 6)

    return GymState(
        task_id="medium/dispatch_with_message",
        seed=seed,
        task_brief=MEDIUM_BRIEF,
        task_difficulty="medium",
        drivers=drivers,
        riders=riders,
    )


# --------------------------------------------------------------------------- #
# Task 3: hard / fraud_review_and_block
# --------------------------------------------------------------------------- #

HARD_BRIEF = (
    "Three accounts are flagged on the Fraud Review panel. Account 1042 has "
    "a high-severity 'fake_pickups' pattern (5+ confirmed bogus rides in the "
    "last hour). Block account 1042 and log the reason as "
    "'fake_pickups_confirmed'. The other two flagged accounts are NOT yours "
    "to block — only 1042. Then submit."
)


def make_hard_fraud_review_and_block(seed: int) -> GymState:
    rng = random.Random(seed)

    # Need at least 4 drivers so that 1042, 1037, 1051 all exist as Driver rows.
    drivers: dict[str, Driver] = {}
    for did_num in (1037, 1042, 1051, 1063, 1078):
        d = Driver(
            id=f"D{did_num}",
            name=rng.choice(DRIVER_NAMES),
            location=(rng.randint(0, 9), rng.randint(0, 9)),
            status="idle",
            rating=round(rng.uniform(3.5, 4.8), 2),
            completed_trips_today=rng.randint(2, 12),
        )
        drivers[d.id] = d

    # One waiting rider (just for visual realism — not required for the task).
    riders = _fresh_riders(rng, 1, ids=["R1"])

    fraud_alerts = {
        1042: FraudAlert(
            account_id=1042, severity="high", pattern="fake_pickups",
            evidence=[
                "5 trips in last 60 min, all ended within 1 grid unit of pickup",
                "Same passenger phone (+1-555-FAKE) on 4 of 5 trips",
                "Driver self-rated all trips 5 stars within 5 seconds of completion",
            ],
        ),
        1037: FraudAlert(
            account_id=1037, severity="low", pattern="rapid_cancellations",
            evidence=[
                "3 cancellations in 20 min — well within tolerance",
                "Cancellation reason field present and looks legitimate",
            ],
        ),
        1051: FraudAlert(
            account_id=1051, severity="medium", pattern="rating_drop",
            evidence=[
                "Rolling 7-day rating fell from 4.7 → 4.2",
                "No customer complaints filed",
            ],
        ),
    }

    return GymState(
        task_id="hard/fraud_review_and_block",
        seed=seed,
        task_brief=HARD_BRIEF,
        task_difficulty="hard",
        drivers=drivers,
        riders=riders,
        fraud_alerts=fraud_alerts,
    )


# --------------------------------------------------------------------------- #
# Registry — the only thing the FastAPI server imports.
# --------------------------------------------------------------------------- #

TASKS = {
    "easy/match_single_ride":         make_easy_match_single_ride,
    "medium/dispatch_with_message":   make_medium_dispatch_with_message,
    "hard/fraud_review_and_block":    make_hard_fraud_review_and_block,
}


def make_task(task_id: str, seed: int) -> GymState:
    if task_id not in TASKS:
        raise KeyError(f"unknown task '{task_id}'. Known: {list(TASKS)}")
    return TASKS[task_id](seed)
