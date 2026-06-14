"""Batch embedding generation using Ollama's local embedding API."""

import asyncio
import httpx
from app.config import settings

# Reusable client for connection pooling
_client: httpx.AsyncClient | None = None

# Limit concurrent requests to avoid overwhelming Ollama
MAX_CONCURRENT_REQUESTS = 10


async def get_client() -> httpx.AsyncClient:
    """Get or create reusable HTTP client."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def embed_text(text: str) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: The text to embed

    Returns:
        Embedding vector as list of floats
    """
    result = await embed_texts([text])
    return result[0]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts with controlled concurrency.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If texts list is empty
        httpx.HTTPError: If API request fails
    """
    if not texts:
        raise ValueError("Cannot embed empty text list")

    client = await get_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Create concurrent tasks with semaphore limiting
    tasks = [
        _embed_single_text(client, semaphore, text)
        for text in texts
    ]

    # Execute all requests with controlled concurrency
    embeddings = await asyncio.gather(*tasks)
    return embeddings


async def _embed_single_text(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    text: str
) -> list[float]:
    """Generate embedding for a single text with semaphore control."""
    async with semaphore:
        response = await client.post(
            f"{settings.OLLAMA_HOST}/api/embeddings",
            json={
                "model": settings.OLLAMA_EMBED_MODEL,
                "prompt": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]


async def close_client():
    """Close the HTTP client (cleanup)."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
