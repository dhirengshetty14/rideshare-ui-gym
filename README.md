# rideshare-ui-gym

A **UI-based RL gym** for rideshare dispatch operations. Agents see a real
HTML dashboard (driver fleet, rider queue, fraud panel, activity log), pick
actions via tool calls, and the verifier grades the final state.

This is a smaller, focused companion to the tool-calling
[rideshare-gym](https://github.com/dhirengshetty14/rideshare-gym), built
specifically to demonstrate the **UI-style gym pattern** (TauBench / STARK
shape) that's becoming standard for production-grade enterprise RL data.

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐      │
│   │   Browser   │     │   FastAPI    │     │  Verifier per    │      │
│   │ Dispatch UI │◄───►│ /api/state   │◄───►│  task (state-    │      │
│   │ (HTML+TW)   │     │ /api/apply   │     │  based, partial- │      │
│   └─────┬───────┘     │ /api/verify  │     │  credit grading) │      │
│         │             └──────┬───────┘     └──────────────────┘      │
│         │                    │                                       │
│         │ data-test-id       │ /api/dom_summary                      │
│         │ click events       │ (agent-friendly view)                 │
│         │                    │                                       │
│         ▼                    ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Agents (swappable, same interface)                         │   │
│   │   • oracle      hand-coded Python (verifier validation)     │   │
│   │   • dom_agent   LLM via tool-call loop (Anthropic / Qwen /  │   │
│   │                  any LiteLLM endpoint)                       │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Quick start

### 1. Install

```bash
cd rideshare-ui-gym
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Start the gym server

```bash
uvicorn server.main:app --reload --port 8000
```

Open <http://localhost:8000> in a browser. You'll see the dispatch console.

### 3. Click around (verify the gym works as a human)

- Pick a task in the top-right dropdown, hit **Reset**
- Click **Dispatch** on a driver row → pick a rider → confirm
- Click **Submit task** → see the verifier's score

### 4. Run the hand-coded oracle (sanity check — should always score 1.0)

```bash
python -m eval.run --agent oracle --tasks all --seeds 0
```

If the oracle doesn't score 1.0 on every task, the verifier has a bug.

### 5. Run the LLM agent

The DOM agent works against any LiteLLM-compatible endpoint, so any of
these options works without code changes — just env vars and a model
string.

#### Option A — Together AI (Qwen, simplest, ~$0.01/run)

```bash
export TOGETHER_API_KEY=...   # https://api.together.ai
python -m eval.run --agent dom --backend litellm \
    --model together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo \
    --tasks all --seeds 0
```

#### Option B — OpenRouter (Qwen, has free tier)

```bash
export OPENROUTER_API_KEY=...   # https://openrouter.ai
python -m eval.run --agent dom --backend litellm \
    --model openrouter/qwen/qwen-2.5-7b-instruct \
    --tasks all --seeds 0
```

#### Option C — Anthropic Claude (no Qwen, but native tool-use)

```bash
export ANTHROPIC_API_KEY=...
python -m eval.run --agent dom --backend anthropic --tasks all --seeds 0
```

#### Option D — Self-hosted Qwen via vLLM (uses your own GPU)

```bash
# Terminal 1: serve Qwen2.5-7B with native tool-call parsing
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 \
    --enable-auto-tool-choice --tool-call-parser hermes

# Terminal 2: point LiteLLM at the local server
export OPENAI_API_BASE=http://localhost:8001/v1
export OPENAI_API_KEY=local-doesnt-matter
python -m eval.run --agent dom --backend litellm \
    --model openai/Qwen/Qwen2.5-7B-Instruct --tasks all --seeds 0
```

If your GPU is on a remote cluster (e.g. SLURM login node), tunnel the
port: `ssh -N -L 8001:<compute-node>:8001 user@cluster`.

## What's in here

| Path | What it is |
|---|---|
| `server/state.py` | Data model + state mutations |
| `server/tasks.py` | 3 task definitions (easy/medium/hard) |
| `server/verifiers.py` | One verifier per task, returns score + category |
| `server/main.py` | FastAPI app — `/api/reset`, `/api/state`, `/api/apply`, `/api/verify`, `/api/dom_summary` |
| `ui/index.html` | Dispatch dashboard — every interactable has `data-test-id` |
| `ui/app.js` | Frontend logic: render state, modals, submit |
| `agents/base.py` | `GymClient` HTTP wrapper + `Trajectory` dataclass |
| `agents/oracle_agent.py` | Hand-coded perfect solver (used to validate verifiers) |
| `agents/dom_agent.py` | LLM agent — Anthropic or LiteLLM, tool-call loop |
| `eval/run.py` | CLI runner: `python -m eval.run --agent dom --tasks all --seeds 0` |
| `DESIGN.md` | Architecture write-up |

## See also

- [`DESIGN.md`](./DESIGN.md) — the architecture document with diagrams and
  scaling considerations.
- The original
  [rideshare-gym](https://github.com/dhirengshetty14/rideshare-gym) — same
  domain, but tool-calling (no UI). Uses the same verifier pattern.
