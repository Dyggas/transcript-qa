"""LLM provider interface with connection pooling and error handling."""

import httpx
from app.config import settings

# Reusable client for connection pooling
_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    """Get or create reusable HTTP client."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


async def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer to a question using retrieved context.

    Args:
        question: The user's question
        context: Retrieved transcript chunks as context

    Returns:
        Generated answer text

    Raises:
        ValueError: If LLM provider is not supported
        RuntimeError: If API request fails
    """
    prompt = _build_prompt(question, context)

    if settings.LLM_PROVIDER == "ollama":
        return await _generate_with_ollama(prompt)
    elif settings.LLM_PROVIDER == "zai":
        return await _generate_with_zai(prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


def _build_prompt(question: str, context: str) -> str:
    """Build context-aware prompt for answer generation."""
    return f"""You are a helpful assistant answering questions about a Victorian home documentary.

Use the following transcript excerpts to answer the question. Include specific details and timestamps when relevant.

Context:
{context}

Question: {question}

Provide a clear, informative answer based on the transcript above. If the answer is not in the transcript, say so explicitly."""


async def _generate_with_ollama(prompt: str) -> str:
    """
    Generate answer using Ollama API.

    Args:
        prompt: The prompt to send to Ollama

    Returns:
        Generated response text

    Raises:
        RuntimeError: If Ollama API request fails
    """
    client = await get_client()
    try:
        response = await client.post(
            f"{settings.OLLAMA_HOST}/api/generate",
            json={
                "model": settings.OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error with Ollama: {e}") from e


async def _generate_with_zai(prompt: str) -> str:
    """
    Generate answer using z.ai API (GLM models).

    Args:
        prompt: The prompt to send to z.ai

    Returns:
        Generated response text

    Raises:
        RuntimeError: If z.ai API request fails
    """
    client = await get_client()
    try:
        response = await client.post(
            f"{settings.ZAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.ZAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.ZAI_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except httpx.HTTPError as e:
        raise RuntimeError(f"z.ai API error: {e}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected z.ai API response format: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error with z.ai: {e}") from e


async def close_client():
    """Close the HTTP client (cleanup)."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
