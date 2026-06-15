"""FastAPI application — POST /ask endpoint."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import embedding
from app.chunking import parse_transcript
from app.config import settings
from app.llm import generate_answer
from app.retrieval import RetrievalIndex
from app.schemas import AskRequest, AskResponse, Source, Timings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Global state for the retrieval index
retrieval_index: RetrievalIndex | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the retrieval index on startup."""
    global retrieval_index

    logger.info("Loading transcript and building index...")

    # Parse transcript into chunks
    chunks = parse_transcript()
    logger.info("Parsed %d transcript chunks", len(chunks))

    # Generate embeddings for all chunks
    logger.info("Generating embeddings...")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = await embedding.embed_texts(chunk_texts)
    logger.info("Generated %d embeddings", len(embeddings))

    # Build retrieval index
    retrieval_index = RetrievalIndex(chunks, embeddings)
    logger.info("Index ready (dimension: %d)", retrieval_index.embedding_dim)

    yield

    # Cleanup
    retrieval_index = None
    await embedding.close_client()
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Transcript QA",
    description="RAG-based Q&A system for Victorian home documentary transcript",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ready",
        "index_loaded": retrieval_index is not None,
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Answer a question about the Victorian home transcript.

    This endpoint:
    1. Embeds the user's question
    2. Retrieves relevant transcript chunks (top-K)
    3. Generates an answer using LLM with retrieved context
    4. Returns answer with source citations
    """
    if retrieval_index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Please try again later."
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    t0 = time.perf_counter()

    # Embed the question
    try:
        query_embedding = await embedding.embed_text(question)
    except RuntimeError as e:
        logger.exception("Embedding failed")
        raise HTTPException(
            status_code=503, detail=f"Embedding service unavailable: {e}"
        ) from e
    t1 = time.perf_counter()

    # Retrieve top-K relevant chunks, keeping only those above the relevance floor
    results = retrieval_index.retrieve(query_embedding, k=settings.TOP_K)
    results = [(chunk, score) for chunk, score in results if score >= settings.SCORE_FLOOR]
    t2 = time.perf_counter()

    # Out-of-scope question: nothing relevant in the transcript. Refuse instead of
    # forcing the LLM to answer from irrelevant context.
    if not results:
        return AskResponse(
            answer="I don't have information about that in the transcript.",
            sources=[],
            timings=Timings(
                embed_ms=round((t1 - t0) * 1000),
                retrieve_ms=round((t2 - t1) * 1000),
                llm_ms=0,
                total_ms=round((t2 - t0) * 1000),
            ),
        )

    # Build context from retrieved chunks
    context = "\n\n".join(f"[{chunk.timestamp}] {chunk.text}" for chunk, _ in results)

    # Generate answer using LLM
    try:
        answer = await generate_answer(question, context)
    except RuntimeError as e:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=502, detail=f"LLM provider error: {e}"
        ) from e
    t3 = time.perf_counter()

    # Build response with sources
    sources = [
        Source(
            timestamp=chunk.timestamp,
            excerpt=(
                chunk.source_text[:300] + "..."
                if len(chunk.source_text) > 300
                else chunk.source_text
            ),
            score=round(score, 4),
        )
        for chunk, score in results
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        timings=Timings(
            embed_ms=round((t1 - t0) * 1000),
            retrieve_ms=round((t2 - t1) * 1000),
            llm_ms=round((t3 - t2) * 1000),
            total_ms=round((t3 - t0) * 1000),
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
