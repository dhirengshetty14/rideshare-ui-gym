"""DOM-action LLM agent.

The agent gets a compact JSON view of the current UI ("DOM summary") on
every step. It picks ONE action per turn — click, type, dispatch,
send_message, block_account, or finish — by emitting a tool call. The
runner translates the tool call into a /api/apply request to the gym.

Why not pixel-level Computer Use?
    * Faster to iterate on a weekend
    * Works with non-vision models (Qwen, Llama) out of the box
    * The "compact DOM" view is what production agentic systems actually
      tend to use under the hood — Computer Use is the showy version, but
      most real systems use accessibility trees / DOM dumps for reliability

The same architecture supports adding a screenshot-based path later: just
swap the dom_summary observation for a base64 screenshot and the model's
tool schema for click(x, y) / type(text). The trajectory recorder doesn't
care which.

Tested against:
    * Anthropic Claude (sonnet-4 family) via Anthropic SDK
    * Any LiteLLM-compatible endpoint (OpenAI, local Qwen via vLLM, etc.)

Pick the backend with `--backend anthropic` or `--backend litellm` on the
runner.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from agents.base import (
    GymClient, Step, Trajectory, new_trajectory,
)


# --------------------------------------------------------------------------- #
# Tool schemas — what actions the LLM can take
# --------------------------------------------------------------------------- #

UI_TOOLS_OPENAI: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "dispatch",
            "description": "Assign a driver to a rider. Both must exist; the "
                           "driver must be idle and the rider must be waiting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "driver_id": {"type": "string",
                                  "description": "ID like 'D1001'."},
                    "rider_id": {"type": "string",
                                 "description": "ID like 'R1'."},
                },
                "required": ["driver_id", "rider_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a rider or driver. Use this for "
                           "confirmations, ETAs, and operator-side comms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_id": {"type": "string"},
                    "recipient_kind": {"type": "string",
                                       "enum": ["rider", "driver"]},
                    "body": {"type": "string",
                             "description": "The message body (plain text)."},
                },
                "required": ["recipient_id", "recipient_kind", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "block_account",
            "description": "Block a flagged account in the fraud panel. "
                           "Provide a reason — it is permanently logged. Be "
                           "sure: blocking innocent accounts is a hard fail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["account_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the task as complete. The verifier runs "
                           "next and you cannot take any more actions after "
                           "calling finish. Only call this when you believe "
                           "you've fully completed the task brief.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# Anthropic uses a slightly different schema shape (no "function" wrapper).
UI_TOOLS_ANTHROPIC: list[dict[str, Any]] = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in UI_TOOLS_OPENAI
]


SYSTEM_PROMPT = """You are an experienced rideshare dispatch operator running a console.
Each turn you see a compact JSON snapshot of the dashboard plus a task brief.
Your job: read the brief, look at the state, and call ONE tool that makes
progress toward completing the task. After enough actions, call `finish` to
submit.

Rules:
- Read the task brief carefully. Different tasks need different actions.
- Look at driver locations to compute distances (Manhattan |dx|+|dy|).
- Only IDLE drivers can be dispatched. Skip offline/blocked ones.
- For fraud review: only block accounts that the brief explicitly tells you
  to. Blocking innocent accounts is a serious failure.
- Be specific in messages: include driver names, IDs, ETAs, anything the
  rider would need.
- Don't loop. If you've done what the brief asks, call `finish`.
- One tool call per turn. The system will run it and show you the new state.
"""


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class DOMAgent:
    """LLM agent that picks one tool per turn from a compact DOM view."""

    def __init__(
        self,
        backend: str = "anthropic",            # "anthropic" or "litellm"
        model: str | None = None,
        max_steps: int = 12,
        verbose: bool = True,
    ):
        self.backend = backend
        self.model = model or self._default_model(backend)
        self.max_steps = max_steps
        self.verbose = verbose
        self._messages: list[dict[str, Any]] = []
        self._client = self._build_client()

    @staticmethod
    def _default_model(backend: str) -> str:
        if backend == "anthropic":
            # Sonnet 4.6 is a solid default for tool-calling in mid-2026.
            return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        return os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")

    def _build_client(self):
        if self.backend == "anthropic":
            from anthropic import Anthropic
            return Anthropic()                     # picks up ANTHROPIC_API_KEY
        # else litellm
        import litellm                              # noqa: F401
        return None                                 # litellm.completion is module-level

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(self, gym: GymClient, task_id: str, seed: int) -> Trajectory:
        reset_resp = gym.reset(task_id, seed)
        traj = new_trajectory(task_id=task_id, seed=seed,
                              agent_name=f"dom_agent[{self.backend}/{self.model}]")
        traj.initial_state = reset_resp["state"]

        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        try:
            for _ in range(self.max_steps):
                if gym.state().get("finished"):
                    break

                dom = gym.dom_summary()
                user_turn = self._format_observation(dom)
                self._messages.append({"role": "user", "content": user_turn})

                tool_call, raw, reasoning = self._ask_model()
                if tool_call is None:
                    # Model declined to call a tool — treat as finish.
                    if self.verbose:
                        print(f"[dom_agent] no tool call; raw={raw[:200]!r}")
                    resp = gym.finish()
                    traj.steps.append(Step(
                        step_idx=len(traj.steps), action_kind="finish",
                        action_args={}, server_result=resp.get("result", {}),
                        state_after=resp.get("state", {}),
                        reasoning=reasoning, raw_model_output=raw,
                    ))
                    break

                kind = tool_call["name"]
                args = tool_call.get("arguments", {}) or {}
                if self.verbose:
                    print(f"[dom_agent] step {len(traj.steps)}: "
                          f"{kind}({json.dumps(args)})")

                resp = gym.apply(kind, args)
                traj.steps.append(Step(
                    step_idx=len(traj.steps), action_kind=kind,
                    action_args=args, server_result=resp.get("result", {}),
                    state_after=resp.get("state", {}),
                    reasoning=reasoning, raw_model_output=raw,
                ))

                tool_result_msg = json.dumps({
                    "ok": resp.get("ok", True),
                    "result": resp.get("result", {}),
                })
                self._append_tool_result(tool_call, tool_result_msg)

                if kind == "finish":
                    break
        except Exception as e:
            traj.error = f"{type(e).__name__}: {e}"

        traj.final_state = gym.state()
        traj.verifier_result = gym.verify()["verifier"]
        traj.finished_at = time.time()
        return traj

    # ------------------------------------------------------------------ #
    # Backend-specific bits
    # ------------------------------------------------------------------ #
    def _format_observation(self, dom: dict[str, Any]) -> str:
        """Compact JSON view of the dashboard for the LLM."""
        return (
            "Current dashboard state:\n```json\n"
            f"{json.dumps(dom, indent=2, default=str)[:8000]}\n```\n"
            "Pick one tool call to make progress."
        )

    def _ask_model(self) -> tuple[dict[str, Any] | None, str, str]:
        """Returns (tool_call_dict_or_None, raw_output_text, reasoning)."""
        if self.backend == "anthropic":
            return self._ask_anthropic()
        return self._ask_litellm()

    def _ask_anthropic(self) -> tuple[dict[str, Any] | None, str, str]:
        # Anthropic SDK takes system as a separate arg, not a message.
        sys_msg = self._messages[0]["content"]
        # Convert the rest into anthropic-shaped messages.
        msgs = [m for m in self._messages[1:]]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=sys_msg,
            tools=UI_TOOLS_ANTHROPIC,
            messages=msgs,
        )
        # Extract first tool_use block + any text
        text_parts, tool_call, tool_use_id = [], None, None
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use" and tool_call is None:
                tool_call = {"name": block.name, "arguments": dict(block.input or {})}
                tool_use_id = block.id

        # Save assistant turn for follow-up
        self._messages.append({
            "role": "assistant",
            "content": [
                *([{"type": "text", "text": "\n".join(text_parts)}]
                  if text_parts else []),
                *([{"type": "tool_use", "id": tool_use_id,
                    "name": tool_call["name"], "input": tool_call["arguments"]}]
                  if tool_call else []),
            ],
            "_tool_use_id": tool_use_id,            # stash for tool result pairing
        })
        raw = "\n".join(text_parts)
        reasoning = raw
        return tool_call, raw, reasoning

    def _ask_litellm(self) -> tuple[dict[str, Any] | None, str, str]:
        import litellm
        resp = litellm.completion(
            model=self.model,
            messages=self._messages,
            tools=UI_TOOLS_OPENAI,
            tool_choice="auto",
        )
        choice = resp.choices[0].message
        raw = (choice.content or "")
        tool_call = None
        if getattr(choice, "tool_calls", None):
            tc = choice.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_call = {"name": tc.function.name, "arguments": args}
            self._messages.append({
                "role": "assistant", "content": raw,
                "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }],
            })
        else:
            self._messages.append({"role": "assistant", "content": raw})
        return tool_call, raw, raw

    def _append_tool_result(self, tool_call: dict[str, Any], result: str) -> None:
        """Add the tool's response back into the conversation so the model
        can react on the next turn."""
        if self.backend == "anthropic":
            tool_use_id = self._messages[-1].get("_tool_use_id")
            self._messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }],
            })
        else:
            # OpenAI-style
            self._messages.append({
                "role": "tool",
                "tool_call_id": "x",                  # litellm tolerates "x"
                "name": tool_call["name"],
                "content": result,
            })
