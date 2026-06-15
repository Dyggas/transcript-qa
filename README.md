# transcript-qa

A retrieval-augmented Q&A backend for a documentary transcript
("An Investigation into Domestic Dangers Through History"). Ask a natural-language
question over HTTP and get an answer grounded in the transcript, with ranked,
timestamped source citations.

Runs **fully locally** — embeddings and generation both use [Ollama](https://ollama.com),
so there is no paid API or cloud dependency. See [DESIGN.md](DESIGN.md) for how it works.

## Prerequisites

- **Docker** (with Compose) — for the recommended run.
- **Ollama** installed on the host ([download](https://ollama.com/download)).

> Models are configurable — see [Configuration](#configuration). Larger models
> give better answers if your hardware allows.

## Run with Docker (recommended)

```bash
cp .env.example .env          # defaults work out of the box
./start_and_run.sh            # starts Ollama, pulls models, runs the container
```

`start_and_run.sh` orchestrates the whole thing. Step by step it:

1. **Loads `.env`** and exports every setting so both the script and the
   container see the same model names and config.
2. **Starts Ollama** bound to `0.0.0.0:11434` in the
   background, so the container can reach it from inside the Docker network. It
   registers a cleanup trap that stops Ollama when the script exits (Ctrl-C,
   error, or normal exit).
3. **Waits for Ollama to be ready** by polling `/api/tags` before doing anything
   that depends on it.
4. **Pulls the embedding and generation models** named in `.env`.
5. **Runs `docker compose up`**.

On startup the service parses the transcript and embeds all chunks (~2 minutes on
very, very modest hardware); it's ready once the log shows `Index ready`. The container
reaches the host's Ollama via `host.docker.internal`, configured in
`docker-compose.yml`.

## Run natively (alternative)

Without Docker you have to do by hand what `start_and_run.sh` does for you: run
Ollama, pull the models, then start the API.

```bash
# 1. Start Ollama (in a separate terminal, or background it) and pull the models.
ollama serve &
ollama pull nomic-embed-text     # embedding model
ollama pull qwen2.5:1.5b         # generation model

# 2. Set up the Python environment and config.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # defaults work out of the box

# 3. Start the API.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Running natively, the app talks to Ollama at `http://localhost:11434` (the
`OLLAMA_HOST` default) — no `host.docker.internal` override is involved. If you
change `OLLAMA_EMBED_MODEL` or `OLLAMA_LLM_MODEL` in `.env`, pull those model
names instead of the defaults above. As with Docker, the first boot embeds the
whole transcript and is ready once the log shows `Index ready`.

## Usage

Health check:

```bash
curl http://localhost:8000/
# {"status":"ready","index_loaded":true}
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What was added to bread to whiten it?"}'
```

Response:

```json
{
  "answer": "Alum, an aluminium-based compound, was added to bread as a whitener ...",
  "sources": [
    { "timestamp": "00:03:58", "excerpt": "The biggest adulterant ... was alum ...", "score": 0.83 }
  ],
  "timings": { "embed_ms": 90, "retrieve_ms": 0, "llm_ms": 24310, "total_ms": 24400 }
}
```

Questions whose answer isn't in the transcript are refused rather than answered:

```json
{ "answer": "I don't have information about that in the transcript.", "sources": [], "timings": { ... } }
```

A minimal web UI (text field + answer/sources) is served at
`http://localhost:8000/ui`. Interactive API docs are at `http://localhost:8000/docs`.

## Configuration

All settings are environment variables (see `.env.example`), loaded via
`app/config.py`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint (overridden to `host.docker.internal` in Docker) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_LLM_MODEL` | `qwen2.5:1.5b` | Generation model |
| `TOP_K` | `5` | Chunks retrieved per query |
| `SCORE_FLOOR` | `0.5` | Min cosine similarity to count as relevant; below it a question is treated as out-of-scope |
| `OVERLAP_TOKENS` | `50` | Token overlap between adjacent chunks |
| `TRANSCRIPT_PATH` | `data/transcript.txt` | Transcript to index |

## Notes

- **Startup:** the index is built in memory at startup and not persisted, so the
  first boot re-embeds the transcript each time.
