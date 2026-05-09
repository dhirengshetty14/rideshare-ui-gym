#!/usr/bin/env bash
# Start the rideshare-ui-gym FastAPI server with auto-reload.
# After it's up, open http://localhost:8000 to use the dispatch console.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
