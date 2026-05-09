"""Hand-coded oracle agent.

Solves each of the 3 tasks deterministically and correctly. Used to:

    1. Validate verifiers (the oracle should always score 1.0 — if it
       doesn't, your verifier has a bug).
    2. Provide gold-trajectory reference data for downstream SFT.
    3. Anchor evaluation: the oracle is the ceiling.

Important: the oracle is NOT an LLM. It's hand-written Python that knows
each task's specific shape. That's fine — its purpose is verifier
validation and gold trajectories, not benchmarking model capability.
"""

from __future__ import annotations

from agents.base import (
    GymClient, Step, Trajectory, new_trajectory, save_trajectory,
)


def _record_step(traj: Trajectory, kind: str, args: dict, response: dict,
                 reasoning: str = "") -> None:
    traj.steps.append(Step(
        step_idx=len(traj.steps),
        action_kind=kind, action_args=args,
        server_result=response.get("result", {}),
        state_after=response.get("state", {}),
        reasoning=reasoning,
    ))


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --------------------------------------------------------------------------- #
# Task solvers
# --------------------------------------------------------------------------- #

def solve_easy_match_single_ride(client: GymClient, traj: Trajectory) -> None:
    s = client.state()
    rider = s["riders"]["R1"]
    pickup = tuple(rider["pickup"])

    # Find closest IDLE driver
    idle = [(d_id, d) for d_id, d in s["drivers"].items() if d["status"] == "idle"]
    if not idle:
        return                           # nothing to do; verifier will mark fail
    idle.sort(key=lambda kv: _manhattan(tuple(kv[1]["location"]), pickup))
    closest_id = idle[0][0]

    resp = client.apply("dispatch", {"driver_id": closest_id, "rider_id": "R1"})
    _record_step(traj, "dispatch",
                 {"driver_id": closest_id, "rider_id": "R1"}, resp,
                 reasoning=f"Closest idle driver to R1 is {closest_id}.")
    resp = client.finish()
    _record_step(traj, "finish", {}, resp, reasoning="Single-task done; submitting.")


def solve_medium_dispatch_with_message(client: GymClient, traj: Trajectory) -> None:
    s = client.state()
    rider = s["riders"]["R3"]
    pickup = tuple(rider["pickup"])

    idle = [(d_id, d) for d_id, d in s["drivers"].items() if d["status"] == "idle"]
    if not idle:
        return
    idle.sort(key=lambda kv: _manhattan(tuple(kv[1]["location"]), pickup))
    closest_id, closest = idle[0]
    eta = _manhattan(tuple(closest["location"]), pickup) * 2

    resp = client.apply("dispatch", {"driver_id": closest_id, "rider_id": "R3"})
    _record_step(traj, "dispatch",
                 {"driver_id": closest_id, "rider_id": "R3"}, resp,
                 reasoning=f"R3 is priority; closest idle is {closest_id}, eta {eta} min.")

    body = (f"Hi! Your driver {closest['name']} is on the way. "
            f"ETA approximately {eta} minutes. Please be ready at the pickup point.")
    resp = client.apply("send_message",
                        {"recipient_id": "R3", "recipient_kind": "rider", "body": body})
    _record_step(traj, "send_message",
                 {"recipient_id": "R3", "recipient_kind": "rider", "body": body}, resp,
                 reasoning="Confirmation message includes driver name and ETA.")

    resp = client.finish()
    _record_step(traj, "finish", {}, resp)


def solve_hard_fraud_review_and_block(client: GymClient, traj: Trajectory) -> None:
    s = client.state()

    # The brief tells us 1042 is the high-severity one. We confirm by reading
    # the alert's severity field — the agent should also do this.
    target_account = 1042
    alert = s["fraud_alerts"].get(str(target_account))
    if alert is None or alert["severity"] != "high":
        return                          # if conditions changed, abort

    reason = "fake_pickups_confirmed"
    resp = client.apply("block_account",
                        {"account_id": target_account, "reason": reason})
    _record_step(traj, "block_account",
                 {"account_id": target_account, "reason": reason}, resp,
                 reasoning="High-severity fake_pickups confirmed; blocking 1042 only.")

    resp = client.finish()
    _record_step(traj, "finish", {}, resp,
                 reasoning="Other alerts are not high-severity; do not block them.")


SOLVERS = {
    "easy/match_single_ride":        solve_easy_match_single_ride,
    "medium/dispatch_with_message":  solve_medium_dispatch_with_message,
    "hard/fraud_review_and_block":   solve_hard_fraud_review_and_block,
}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_oracle(client: GymClient, task_id: str, seed: int) -> Trajectory:
    """Reset gym, solve task, run verifier, return the recorded trajectory."""
    if task_id not in SOLVERS:
        raise KeyError(f"oracle has no solver for {task_id}")

    reset_resp = client.reset(task_id, seed)
    traj = new_trajectory(task_id=task_id, seed=seed, agent_name="oracle")
    traj.initial_state = reset_resp["state"]

    try:
        SOLVERS[task_id](client, traj)
    except Exception as e:
        traj.error = f"{type(e).__name__}: {e}"

    traj.final_state = client.state()
    traj.verifier_result = client.verify()["verifier"]
    import time
    traj.finished_at = time.time()
    return traj
