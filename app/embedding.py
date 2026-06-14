"""Batch embedding generation using Ollama's local embedding API."""

import asyncio

import httpx

from app.config import settings

# Reusable client for connection pooling
_client: httpx.AsyncClient | None = None

# Batch size for embedding multiple texts in one request
BATCH_SIZE = 10


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
    Generate embeddings for multiple texts using Ollama batch API.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If texts list is empty
        RuntimeError: If API request fails
    """
    if not texts:
        raise ValueError("Cannot embed empty text list")

    client = await get_client()
    all_embeddings = []

    # Process texts in batches
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        try:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/embed",
                json={
                    "model": settings.OLLAMA_EMBED_MODEL,
                    "input": batch,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns embeddings in the response
            if "embeddings" in data:
                all_embeddings.extend(data["embeddings"])
            else:
                raise RuntimeError(f"Unexpected API response format: {data}")

        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error with Ollama batch embedding: {e}"
            ) from e

    return all_embeddings


async def close_client():
    """Close the HTTP client (cleanup)."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
