# Design

A retrieval-augmented Q&A service over a documentary transcript. The transcript
would fit in a modern context window, but per the brief every query goes through
retrieval rather than stuffing the whole transcript into the prompt.

Pipeline: **parse & chunk → embed → cosine retrieval (with a relevance floor) →
grounded generation → answer + ranked, timestamped sources.**

**Deliberately minimal dependencies.** No RAG framework (LangChain / LlamaIndex)
and no vector DB: the chunking, the Ollama API calls, and cosine retrieval are all
written by hand so the mechanics are explicit and auditable rather than hidden
behind abstractions. Runtime libraries are just FastAPI (HTTP), httpx, NumPy, and
tiktoken. Lean and mean :)

**Docker passes through to the host's Ollama** (`host.docker.internal`) rather than 
bundling an Ollama/model image — keeps the image small and reuses the host's 
already-pulled models and GPU.

## 1. Chunking (`app/chunking.py`)

The transcript is `HH:MM:SS` timestamps each followed by a block of speech, so
chunks are built around that structure:

- **Timestamp boundaries** are the base unit — each is a natural span of meaning
  and carries the time code needed for citations.
- **Oversized segments** (> ~500 tokens, counted with tiktoken) are split at
  sentence ends, keeping the parent timestamp; small ones are left as-is, so a
  chunk never spans two time codes.
- **50-token overlap** from the previous chunk keeps a fact that straddles a
  boundary retrievable from either side.

Overlap helps retrieval but would corrupt citations, so each chunk keeps two
fields: `text` (with overlap, what gets embedded) and `source_text` (the clean
original, what we show as the excerpt). ~260 chunks result.

## 2. Retrieval (`app/embedding.py`, `app/retrieval.py`)

- **Embed:** each chunk's `text` via Ollama `/api/embed` (`nomic-embed-text`,
  768-dim), batched at startup; the query is embedded the same way per request.
- **Search:** vectors are L2-normalised once, so cosine similarity is a single
  dot product and `argsort` returns the top-K (default 5). At ~260 chunks an exact
  NumPy scan is sub-millisecond — FAISS/Chroma would be needless complexity.
- **Relevance floor:** chunks below `SCORE_FLOOR` (0.5) are dropped; if none
  remain, the question is treated as out-of-scope and refused without an LLM
  call. Calibrated from observed scores — in-scope ~0.7–0.86, unrelated ~0.43.

## 3. Prompt (`app/llm.py`)

Retrieved chunks are formatted `[HH:MM:SS] text`, joined, and placed in one
instruction prompt sent to Ollama (`qwen2.5:1.5b`). It is deliberately strict
because a small local model otherwise drifts to its own knowledge. That refusal
string is a single shared constant, so the prompt path and the relevance-floor
path return identical text.

## 4. Improvements with more time

- **Cache embeddings** (keyed by transcript hash) — startup re-embeds for ~2 min.
- **Stronger models** — the current ones are sized for very modest hardware;
  larger ones would reduce the minor fabrications seen on broad questions.
- **Hybrid retrieval / reranking** (BM25 + dense, or a cross-encoder) to suppress
  the occasional ASR-noise segment that dense retrieval surfaces. Although it probably
  counts as overengineering :)
- **Token streaming** (`/ask/stream`, SSE) — a real UX win with long generations.
- **`nomic-embed-text` task prefixes** — the model expects `search_document:` /
  `search_query:` prefixes on documents and queries; I didn't get to wire these in
  (and to make them configurable so a different embed model isn't broken by them).
  Both sides are currently prefix-less and consistent, so retrieval still ranks
  sensibly, but adding them would improve separation — and `SCORE_FLOOR` would need
  recalibrating afterwards.

I also built and then removed a one-time generate→critique→revise loop: with a weak
model the critic flagged every answer as ungrounded and tripled latency without
improving grounding.
