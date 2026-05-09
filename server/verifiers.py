"""Per-task verifiers — programmatic referees for each scenario.

Each verifier takes (initial_state, final_state) and returns a result dict:

    {
        "score":           float in [0, 1],
        "success":         bool,
        "error_category":  one of {"none", "goal_incomplete", "wrong_action",
                                   "wrong_args", "side_effect_missing"},
        "details":         human-readable bullets explaining the score,
    }

The structure mirrors the rideshare-gym verifier contract (and through it,
the standard pattern in TauBench / STARK-style gyms): check FINAL STATE,
not output text. Models can game text-pattern verifiers; they can't game
"the right database row got inserted".

Partial credit is granted when the agent did most-of-the-thing-right but
missed one specific assertion. That partial-credit signal is what makes
trajectories useful for downstream training (DPO, GRPO).
"""

from __future__ import annotations

from typing import Any

from server.state import GymState, manhattan


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ok() -> dict[str, Any]:
    return {"score": 1.0, "success": True, "error_category": "none", "details": []}


def _fail(category: str, score: float, *bullets: str) -> dict[str, Any]:
    return {
        "score": float(score),
        "success": False,
        "error_category": category,
        "details": list(bullets),
    }


def _closest_idle_driver(state: GymState, rider_id: str) -> str | None:
    """Returns driver_id of the idle driver with minimum Manhattan distance to
    the rider's pickup. None if no idle drivers exist."""
    rider = state.riders.get(rider_id)
    if rider is None:
        return None
    idle = [d for d in state.drivers.values() if d.status == "idle"]
    if not idle:
        return None
    closest = min(idle, key=lambda d: manhattan(d.location, rider.pickup))
    return closest.id


# --------------------------------------------------------------------------- #
# Task 1: easy / match_single_ride
# --------------------------------------------------------------------------- #

def verify_easy_match_single_ride(
    initial: GymState, final: GymState,
) -> dict[str, Any]:
    """Did the agent dispatch the correct (closest idle) driver to R1?"""
    expected_driver = _closest_idle_driver(initial, "R1")
    if expected_driver is None:
        # The task is malformed; no idle drivers in the initial state.
        return _fail("side_effect_missing", 0.0,
                     "Verifier bug: no idle driver to assign to R1.")

    # Did anything happen at all?
    rider_assigned = final.riders["R1"].status == "assigned"
    has_trip_for_r1 = any(t.rider_id == "R1" for t in final.trips.values())

    if not has_trip_for_r1:
        return _fail("goal_incomplete", 0.0,
                     "No trip exists for rider R1.",
                     f"Expected a trip with driver_id={expected_driver}.")

    # Find R1's trip and check the assigned driver
    trip = next(t for t in final.trips.values() if t.rider_id == "R1")
    assigned_driver = trip.driver_id

    if assigned_driver == expected_driver:
        return _ok()

    # Wrong driver picked. Partial credit: did they pick *some* idle driver?
    initial_status = initial.drivers[assigned_driver].status \
        if assigned_driver in initial.drivers else "unknown"
    if initial_status == "idle":
        return _fail("wrong_action", 0.5,
                     f"Dispatched {assigned_driver}, but the closest idle "
                     f"driver to R1 was {expected_driver}.")
    return _fail("wrong_action", 0.2,
                 f"Dispatched {assigned_driver}, who was {initial_status} (not idle).")


# --------------------------------------------------------------------------- #
# Task 2: medium / dispatch_with_message
# --------------------------------------------------------------------------- #

def verify_medium_dispatch_with_message(
    initial: GymState, final: GymState,
) -> dict[str, Any]:
    """
    Two requirements (each worth 0.5 of the score):

    A. Dispatched the closest idle driver to R3.
    B. Sent a message to R3 that mentions the assigned driver's NAME and the
       ETA in minutes (an integer, with 'min' or 'minute' nearby).

    Full credit only if both are satisfied. Each missing → 0.5 each.
    """
    score = 0.0
    bullets: list[str] = []

    # ----- A. Dispatch correct driver -----
    expected = _closest_idle_driver(initial, "R3")
    has_trip_for_r3 = any(t.rider_id == "R3" for t in final.trips.values())

    if not has_trip_for_r3:
        bullets.append("[A] No trip exists for R3 — dispatch missing.")
        dispatch_ok = False
        assigned = None
    else:
        trip = next(t for t in final.trips.values() if t.rider_id == "R3")
        assigned = trip.driver_id
        dispatch_ok = (assigned == expected)
        if dispatch_ok:
            score += 0.5
            bullets.append(f"[A] OK: dispatched {assigned} (closest idle).")
        else:
            score += 0.2  # picked someone, but wrong
            bullets.append(
                f"[A] PARTIAL: dispatched {assigned}, expected {expected} "
                f"(closest idle to R3).")

    # ----- B. Message sent with name + ETA -----
    msgs_to_r3 = [m for m in final.messages
                  if m.recipient_kind == "rider" and m.recipient_id == "R3"]
    if not msgs_to_r3:
        bullets.append("[B] No message sent to R3.")
    else:
        msg = msgs_to_r3[-1]                # last message wins
        body = msg.body.lower()

        driver_name_ok = False
        if assigned and assigned in final.drivers:
            driver_name = final.drivers[assigned].name.lower()
            driver_name_ok = driver_name in body

        # Look for an integer near "min" / "minute"
        import re
        eta_ok = bool(re.search(r"\b\d+\s*(min|minute)", body))

        if driver_name_ok and eta_ok:
            score += 0.5
            bullets.append("[B] OK: message includes driver name and ETA.")
        elif driver_name_ok or eta_ok:
            score += 0.25
            missing = "ETA" if not eta_ok else "driver name"
            bullets.append(f"[B] PARTIAL: message missing {missing}.")
        else:
            bullets.append(
                "[B] FAIL: message sent but missing both driver name and ETA. "
                f"Body was: {msg.body!r}")

    if score >= 1.0:
        return _ok()
    return _fail(
        "goal_incomplete" if score == 0.0 else "wrong_args",
        round(score, 3),
        *bullets,
    )


# --------------------------------------------------------------------------- #
# Task 3: hard / fraud_review_and_block
# --------------------------------------------------------------------------- #

def verify_hard_fraud_review_and_block(
    initial: GymState, final: GymState,
) -> dict[str, Any]:
    """
    Three requirements:

    A. Account 1042 was blocked (audit log entry with action='block',
       target_id='1042').
    B. The reason mentions 'fake_pickups' (case-insensitive substring).
    C. Did NOT block 1037 or 1051 (false positives are penalized harder than
       missing the true positive — overblocking real users is the worst
       outcome for an ops gym).

    Scoring: A=0.5, B=0.3, C=0.2. False-positive blocks zero out the score
    (over-blocking is a hard fail).
    """
    bullets: list[str] = []
    blocks = [e for e in final.audit_log if e.action == "block"]

    blocked_targets = {e.target_id for e in blocks}
    blocked_1042 = "1042" in blocked_targets
    blocked_other = blocked_targets - {"1042"}

    # Hard fail: blocked an innocent account
    if blocked_other:
        return _fail("wrong_action", 0.0,
                     f"Blocked innocent account(s): {sorted(blocked_other)}. "
                     "Only 1042 should be blocked. Over-blocking real users "
                     "is the worst possible outcome here.")

    score = 0.0

    # A. Did we block 1042?
    if blocked_1042:
        score += 0.5
        bullets.append("[A] OK: account 1042 was blocked.")
    else:
        bullets.append("[A] FAIL: account 1042 was NOT blocked.")

    # B. Did the reason mention the right pattern?
    if blocked_1042:
        block_entry = next(e for e in blocks if e.target_id == "1042")
        if "fake_pickups" in block_entry.reason.lower():
            score += 0.3
            bullets.append(f"[B] OK: reason mentions fake_pickups.")
        else:
            bullets.append(
                f"[B] PARTIAL: 1042 blocked but reason was {block_entry.reason!r} "
                "(should mention 'fake_pickups').")

    # C. No false positives (already established above)
    score += 0.2
    bullets.append("[C] OK: no innocent accounts were blocked.")

    if score >= 1.0:
        return _ok()
    return _fail(
        "goal_incomplete" if not blocked_1042 else "wrong_args",
        round(score, 3),
        *bullets,
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

VERIFIERS = {
    "easy/match_single_ride":         verify_easy_match_single_ride,
    "medium/dispatch_with_message":   verify_medium_dispatch_with_message,
    "hard/fraud_review_and_block":    verify_hard_fraud_review_and_block,
}


def verify(task_id: str, initial: GymState, final: GymState) -> dict[str, Any]:
    if task_id not in VERIFIERS:
        raise KeyError(f"no verifier for {task_id}. Known: {list(VERIFIERS)}")
    return VERIFIERS[task_id](initial, final)
