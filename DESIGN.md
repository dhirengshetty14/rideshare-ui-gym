# Design — rideshare-ui-gym

> A UI-based RL gym for rideshare dispatch. Agents interact with a real
> HTML dashboard via DOM-targeted actions; verifiers grade the final state.

## 1. Goal

Build a UI-style gym (TauBench / STARK shape) that lets an LLM agent
operate a realistic dispatch console — exactly the kind of environment a
real rideshare operator would see. The agent sees a structured snapshot of
the dashboard, picks one tool call per turn (dispatch, send_message,
block_account, finish), and the verifier grades the final state.

This is intentionally smaller than my [tool-calling
rideshare-gym](https://github.com/dhirengshetty14/rideshare-gym): three
tasks, one fleet, one operator UI. The point is to demonstrate the
end-to-end *shape* of a UI gym in a few thousand lines of code, then
discuss what scaling it to 100s of tasks looks like.

## 2. Architecture

```
        ┌──────────────────────────────────────────────────────────────┐
        │                       Browser (UI)                           │
        │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐ │
        │  │ Driver fleet     │ │ Rider queue      │ │ Fraud review │ │
        │  │ (table, dispatch │ │ (table, msg btn) │ │ (alerts +    │ │
        │  │  buttons)        │ │                  │ │  block btn)  │ │
        │  └──────────────────┘ └──────────────────┘ └──────────────┘ │
        │                                                              │
        │  ┌──────────────────────────────────────────────────────┐   │
        │  │  Activity log: trips, messages, audit                │   │
        │  └──────────────────────────────────────────────────────┘   │
        │                                                              │
        │  Each interactive element has data-test-id="…"               │
        │  → reliably targetable by both humans and agents             │
        └────────────────────────────┬─────────────────────────────────┘
                                     │ HTTP (fetch)
        ┌────────────────────────────▼─────────────────────────────────┐
        │                   FastAPI gym server                         │
        │                                                              │
        │  POST /api/reset      → start episode for (task_id, seed)    │
        │  GET  /api/state      → full state for UI rendering          │
        │  GET  /api/dom_summary → compact, agent-friendly view        │
        │  POST /api/apply      → mutate state with one action         │
        │  POST /api/verify     → run verifier, return score+category  │
        │                                                              │
        │  ┌────────────────┐    ┌──────────────────┐                  │
        │  │ State module   │    │ Tasks (factories │                  │
        │  │ Driver/Rider/  │    │ that build the   │                  │
        │  │ Trip/Message/  │◄──►│ initial world)   │                  │
        │  │ AuditLog       │    │                  │                  │
        │  └────────────────┘    └──────────────────┘                  │
        │           ▲                                                  │
        │           │                                                  │
        │  ┌────────┴───────┐                                          │
        │  │ Verifiers      │   one per task; returns                  │
        │  │ (state-based,  │   {score, success, error_category,       │
        │  │  partial-credit│    details}                              │
        │  │  scoring)      │                                          │
        │  └────────────────┘                                          │
        └──────────────────────────────────────────────────────────────┘
                                     ▲
                                     │ HTTP (httpx)
        ┌────────────────────────────┴─────────────────────────────────┐
        │                       Agents (swappable)                     │
        │                                                              │
        │  • oracle_agent.py   hand-coded Python; always 1.0           │
        │                       → used to validate verifiers           │
        │                                                              │
        │  • dom_agent.py      LLM that sees /api/dom_summary          │
        │                       and picks tool calls                   │
        │                       (Anthropic SDK or LiteLLM)             │
        │                                                              │
        │  Both produce: agents/base.py:Trajectory                     │
        │   ↳ initial_state, steps[], final_state, verifier_result     │
        └──────────────────────────────────────────────────────────────┘
```

### Why this shape

- **Backend-first.** The state mutations live in Python, not in the UI.
  The UI is a thin renderer that calls `/api/apply` on every click. This
  means the verifier sees the same state regardless of who drove the
  episode — human, oracle, or LLM. Same code path. **Same code path is
  the unification — it's what lets the same gym double as eval and
  training factory.**
- **`data-test-id` on every interactable.** Lets agents target elements
  reliably without screen-scraping or pixel-click brittleness. This is
  the same pattern Playwright/Selenium tests use, and what production
  WebArena-style benchmarks rely on.
- **Two views into state.** `/api/state` is verbose (full state for the
  UI to render). `/api/dom_summary` is compact (one element per row, with
  test-ids and current values) for agents that don't have vision. A
  vision agent (Computer Use) could ignore both and use screenshots.
- **Trajectories are first-class.** Every episode is recorded as a
  structured `Trajectory` with all steps, args, server responses, and
  the model's reasoning per step. **That trajectory is data** — for
  inspection, for SFT, for DPO pairs, for GRPO rollouts.

## 3. Component breakdown

### State (`server/state.py`)
Plain dataclasses: `Driver`, `Rider`, `Trip`, `FraudAlert`, `Message`,
`AuditLogEntry`, `GymState`. State is in-memory; mutations go through a
small set of pure functions (`dispatch`, `send_message`, `block_account`,
`set_surge`, `finish`). No DB, no ORM — keeps the gym a single-file
mental model.

### Tasks (`server/tasks.py`)
Each task is a function `make(seed: int) -> GymState` that builds a fresh
starting world. Three tasks span the difficulty curve:

| ID | Difficulty | What the agent must do |
|---|---|---|
| `easy/match_single_ride` | easy | Dispatch the closest IDLE driver to R1. |
| `medium/dispatch_with_message` | medium | Dispatch closest IDLE to priority rider R3, then send R3 a confirmation message that includes the driver's name and the ETA in minutes. |
| `hard/fraud_review_and_block` | hard | Three flagged accounts. Block account 1042 (high-severity fake_pickups) with the reason `fake_pickups_confirmed`. Don't block the other two. |

Each task's seed parameter perturbs driver positions, rider counts, etc.
Different seeds = different problems same shape, same verifier.

### Verifiers (`server/verifiers.py`)
One per task, all sharing this contract:

```python
verify(initial_state, final_state) -> {
    "score":          float in [0, 1],
    "success":        bool,
    "error_category": "none" | "goal_incomplete" | "wrong_action"
                     | "wrong_args" | "side_effect_missing",
    "details":        list[str],
}
```

Key design choices:
- **State-based, not text-based.** Verifier checks `final_state.trips`,
  `final_state.audit_log`, etc. — not the LLM's output text. This is
  what makes it Goodhart-resistant: the model can't cheat by emitting
  the right phrase; it has to actually do the thing.
- **Partial credit.** The medium task has two requirements (A and B),
  each worth 0.5. The hard task has three (block-correctly,
  reason-mentions-pattern, no-false-positives). Partial scores produce
  a richer training signal than binary success.
- **Categorical failure.** Every failure is tagged with one of the
  4 categories above. This is what turns the gym from an eval harness
  into a training signal — failure-mode shifts (chi-square testable)
  tell you whether your training fixed planning or formatting or
  something else.
- **Hard fail for false positives.** On the hard task, blocking an
  innocent account zeroes the score regardless of other actions.
  Over-blocking real users is the worst possible outcome for an ops
  gym; the verifier reflects that.

### Server (`server/main.py`)
FastAPI app. Endpoints are thin: each `/api/apply` call increments
`state.step`, dispatches to one of the state-mutation functions, and
returns the new full state. The UI re-renders on every response.

### UI (`ui/index.html`, `ui/app.js`)
Single-page app. Tailwind via CDN (zero build config). Vanilla JS — no
React/Vue — so anyone can read it. Three panels: drivers, riders, fraud.
Every interactive element has `data-test-id`. Modals for confirmation
(dispatch / message / block).

### Agents
Two reference implementations.

**`oracle_agent.py`.** Hand-coded Python. Knows each task's expected
solution and emits the right sequence of `client.apply(...)` calls. Used
to validate verifiers (oracle always = 1.0 — if not, the verifier has
a bug) and to produce gold trajectories for downstream training.

**`dom_agent.py`.** Generic LLM agent. Each turn:
1. Fetches `/api/dom_summary` (compact JSON view of the dashboard).
2. Sends it to the LLM with a system prompt and the tool schema.
3. The LLM emits ONE tool call (`dispatch`, `send_message`, etc.).
4. Runner translates the tool call into `client.apply(kind, args)`.
5. Tool result fed back into the LLM context for the next turn.

Backends: Anthropic SDK (claude-sonnet-4-5) or LiteLLM (works with
GPT, Qwen via vLLM, Llama, anything). Same agent code, swap the
backend with one CLI flag.

## 4. Failure modes (intentional)

The gym is calibrated so a baseline LLM doesn't trivially score 100% —
otherwise there's nothing to improve. The three task tiers expose
different failure types:

| Tier | What baseline typically gets wrong |
|---|---|
| Easy | Picks the geographically wrong driver, or picks an offline driver. |
| Medium | Sends a generic message ("we're on it") that doesn't include the name or ETA. Schema-valid but missing the partial-credit signal. |
| Hard | Either blocks too many accounts (over-eager) or none (under-eager). Confuses high-severity vs low/medium. |

The verifier's categorical labels (`goal_incomplete` / `wrong_action` /
`wrong_args`) point directly at the fix. That's the closed-loop ev/train
signal: every failure is debuggable.

## 5. How this scales

Three honest scaling questions and how I'd approach them:

### To 100s of tasks
- **Reusable UI components.** Right now everything is in one HTML file.
  At scale, I'd factor out `<DriverTable>`, `<RiderQueue>`, `<FraudPanel>`,
  `<MessageModal>` etc. as a template library, and each task imports the
  components it needs.
- **Templated initial states.** Instead of hand-writing `make_easy_match_*`,
  I'd write a small DSL (or YAML) for "what's in the world" and let task
  authors fill it in.
- **Compositional verifier predicates.** A library of small predicates
  (`tool_called(X)`, `state_has_trip(driver=X, rider=Y)`,
  `audit_contains(action=X, target=Y, reason_substring=Z)`) that snap
  together into per-task verifiers. New verifiers become 5-10 lines of
  composition rather than 100 lines of bespoke logic.

### To production
- **Sandbox container.** Right now the gym runs in a regular Python
  venv. Production needs Docker isolation, network egress filtering
  (so the agent can't actually hit external APIs from inside an
  episode), and resource limits.
- **Browser fleet.** A single FastAPI process serving the UI works for
  one episode at a time. For parallel rollouts, you'd want a pool of
  per-episode containers (Kubernetes deployment) so the agent's
  Playwright instances don't trample each other's state.
- **Trajectory store.** Right now trajectories go to local JSON files.
  At scale, they go to object storage (S3) keyed by
  `(agent, task, seed, episode_id)` for cheap querying.

### To other domains
- The whole structure (state model + task factories + verifiers + UI +
  agent) is domain-agnostic. To build a Jira gym, replace the dispatch
  domain with ticket workflows. To build an Excel gym, replace it with
  cell operations. The verifier pattern (state-based, partial-credit,
  categorical) ports directly.

## 6. What I didn't have time for

Honest gaps I'd close in the next iteration:

- **Adversarial seed search.** Right now seeds are random — they
  produce variation, but they don't actively *find* the seeds that
  break the agent. A real ops gym should fuzz-test, biasing toward
  failure-inducing initial states.
- **Pixel-level Computer Use agent.** The DOM-action agent is fast and
  reliable but less realistic than a screenshot-based agent. The
  architecture supports it (just swap the observation), but I didn't
  wire it up.
- **Multi-tenant.** The session is single-tenant — only one episode at
  a time per server. Production would parallelize across
  per-(agent, episode) backend instances.
- **No SDK packaging.** Right now `eval/run.py` is the entry point. A
  real productionization would ship a `pip install rideshare-ui-gym`
  package with a `Gym` class agents can import directly without
  spinning up a FastAPI process.

## 7. Connecting to my prior gym work

This is companion to [rideshare-gym](https://github.com/dhirengshetty14/rideshare-gym),
which is tool-calling only (12 tasks, full SFT/DPO/GRPO training pipeline,
running on UNC's Longleaf cluster against Qwen2.5-7B-Instruct + QLoRA).
Same domain, same verifier philosophy — but tool-calling tests
*reasoning* through APIs, while this UI version tests *operating* through
a real interface.

Both shapes are useful, and the unification is: **the verifier is the
reward function**. Same code path graded the eval episodes in
rideshare-gym; same code path can be wrapped as a TRL reward function
to power GRPO. That's the RLVR pattern — the gym is both the benchmark
and the training factory.
