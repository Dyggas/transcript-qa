#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"   # run from repo root regardless of where it's called

# Load settings from .env (model names, client URL). -a exports them to child procs.
set -a; source .env; set +a

# Start Ollama bound to all interfaces so the container can reach it.
# OLLAMA_HOST here = bind address, which overrides .env's client-URL meaning.
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null' EXIT   # clean up on Ctrl-C / error / exit

# Wait until Ollama answers before pulling or starting the app.
until curl -sf http://localhost:11434/api/tags >/dev/null; do sleep 0.5; done

# Ensure required models exist (idempotent — fast no-op if already pulled).
ollama pull "$OLLAMA_EMBED_MODEL"
ollama pull "$OLLAMA_LLM_MODEL"

docker compose up
