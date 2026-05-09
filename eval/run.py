"""Episode runner — the CLI you point at the gym to evaluate an agent.

Usage examples:

    # Run the hand-coded oracle on all 3 tasks, seed 0
    python -m eval.run --agent oracle --tasks all --seeds 0

    # Run the Anthropic DOM agent on the medium task, seeds 0-2
    python -m eval.run --agent dom --backend anthropic \
        --tasks medium/dispatch_with_message --seeds 0,1,2

    # Run a local Qwen via LiteLLM (e.g. served by vLLM at port 8001)
    python -m eval.run --agent dom --backend litellm \
        --model openai/Qwen/Qwen2.5-7B-Instruct \
        --tasks all --seeds 0

Outputs go to ./trajectories/<agent>/ as JSONL files (one per episode), and
a scorecard summary is printed at the end.

NOTE: the FastAPI server must be running first:
    uvicorn server.main:app --reload
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from agents.base import GymClient, Trajectory, save_trajectory
from agents.oracle_agent import run_oracle, SOLVERS as ORACLE_TASKS
from agents.dom_agent import DOMAgent


ALL_TASKS = list(ORACLE_TASKS.keys())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_seeds(spec: str) -> list[int]:
    """'0', '0,1,2', '0-4' -> list of ints."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _parse_tasks(spec: str) -> list[str]:
    if spec == "all":
        return ALL_TASKS
    return [t.strip() for t in spec.split(",") if t.strip()]


def _print_scorecard(rows: list[Trajectory]) -> None:
    if not rows:
        print("No episodes ran.")
        return

    print()
    print("=" * 78)
    print(f"{'task':<40} {'seed':>5} {'score':>7} {'success':>8} {'category':>20}")
    print("-" * 78)

    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in rows:
        by_task[t.task_id].append(t)
        print(f"{t.task_id:<40} {t.seed:>5} {t.score:>7.2f} "
              f"{str(t.success):>8} {t.error_category:>20}")

    print("-" * 78)
    overall_success = sum(1 for t in rows if t.success) / len(rows)
    overall_score = sum(t.score for t in rows) / len(rows)
    print(f"{'OVERALL':<40} {'':>5} {overall_score:>7.2f} "
          f"{overall_success * 100:>7.1f}%")

    print()
    print("Per-task:")
    for task_id, ts in sorted(by_task.items()):
        succ = sum(1 for t in ts if t.success) / len(ts)
        scor = sum(t.score for t in ts) / len(ts)
        print(f"  {task_id:<40}  n={len(ts)}  success={succ * 100:5.1f}%  "
              f"mean_score={scor:.3f}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", choices=["oracle", "dom"], required=True,
                    help="Which agent to run.")
    ap.add_argument("--backend", choices=["anthropic", "litellm"], default="anthropic",
                    help="For --agent dom: which LLM backend.")
    ap.add_argument("--model", default=None,
                    help="Override the model id (e.g. claude-sonnet-4-5-20250929 "
                         "or openai/gpt-4o-mini or openai/Qwen/Qwen2.5-7B-Instruct).")
    ap.add_argument("--tasks", default="all",
                    help="'all' or comma-separated task ids.")
    ap.add_argument("--seeds", default="0", help="'0', '0,1,2', or '0-4'.")
    ap.add_argument("--max-steps", type=int, default=12,
                    help="DOM agent: max action turns per episode.")
    ap.add_argument("--server", default="http://localhost:8000",
                    help="FastAPI gym base URL.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write trajectory files. "
                         "Defaults to ./trajectories/<agent>.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tasks = _parse_tasks(args.tasks)
    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir or f"trajectories/{args.agent}")

    client = GymClient(args.server)
    # Sanity: server up?
    try:
        client._http.get("/api/tasks")
    except Exception as e:
        print(f"ERROR: cannot reach gym at {args.server}: {e}\n"
              "Start the server first with: uvicorn server.main:app --reload",
              file=sys.stderr)
        sys.exit(2)

    dom_agent = None
    if args.agent == "dom":
        dom_agent = DOMAgent(backend=args.backend, model=args.model,
                             max_steps=args.max_steps, verbose=not args.quiet)

    trajectories: list[Trajectory] = []
    started = time.time()

    for task_id in tasks:
        for seed in seeds:
            if not args.quiet:
                print(f"\n>>> {args.agent} on {task_id} seed={seed}")
            if args.agent == "oracle":
                if task_id not in ORACLE_TASKS:
                    print(f"  (skip — oracle has no solver for {task_id})")
                    continue
                traj = run_oracle(client, task_id, seed)
            else:
                traj = dom_agent.run(client, task_id, seed)

            path = save_trajectory(traj, out_dir)
            if not args.quiet:
                print(f"  -> score={traj.score:.2f} success={traj.success} "
                      f"category={traj.error_category} -> {path}")
            trajectories.append(traj)

    elapsed = time.time() - started
    print(f"\nRan {len(trajectories)} episodes in {elapsed:.1f}s "
          f"(saved to {out_dir})")
    _print_scorecard(trajectories)

    # Also dump a one-line JSON summary for downstream parsing
    summary = {
        "agent": args.agent,
        "backend": args.backend if args.agent == "dom" else None,
        "model": dom_agent.model if dom_agent else None,
        "n_episodes": len(trajectories),
        "overall_success_rate": (
            sum(1 for t in trajectories if t.success) / len(trajectories)
            if trajectories else 0.0
        ),
        "mean_score": (
            sum(t.score for t in trajectories) / len(trajectories)
            if trajectories else 0.0
        ),
        "by_task": {
            task_id: {
                "n": sum(1 for t in trajectories if t.task_id == task_id),
                "success_rate": (
                    sum(1 for t in trajectories
                        if t.task_id == task_id and t.success)
                    / max(1, sum(1 for t in trajectories if t.task_id == task_id))
                ),
            }
            for task_id in tasks
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_scorecard.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"Scorecard written to {out_dir}/_scorecard.json")


if __name__ == "__main__":
    main()
