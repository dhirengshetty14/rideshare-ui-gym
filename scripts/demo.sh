#!/usr/bin/env bash
# End-to-end demo: starts the server in the background, runs the oracle
# (as a sanity check that verifiers work), then runs the LLM agent on all
# 3 tasks. Prints scorecards for both.
#
# Usage: ./scripts/demo.sh
# Requires: ANTHROPIC_API_KEY env var (or change the agent flags below).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

# Start server in background
echo "Starting gym server..."
uvicorn server.main:app --port 8000 > /tmp/gym.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

# Wait for /api/tasks to respond
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/api/tasks > /dev/null; then
    echo "  server up after ${i}s"
    break
  fi
  sleep 1
done

# 1. Oracle sanity check
echo
echo "=== ORACLE (should all be 1.00) ==="
python -m eval.run --agent oracle --tasks all --seeds 0

# 2. LLM agent
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo
  echo "=== DOM AGENT (Claude) ==="
  python -m eval.run --agent dom --backend anthropic --tasks all --seeds 0
else
  echo
  echo "(skip: ANTHROPIC_API_KEY not set; export it to run the LLM agent demo)"
fi

echo
echo "Server log: /tmp/gym.log"
