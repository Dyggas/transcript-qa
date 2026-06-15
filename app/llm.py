"""LLM provider interface with connection pooling and error handling."""

import httpx

from app.config import settings

# Reusable client for connection pooling
_client: httpx.AsyncClient | None = None

# Returned verbatim when the transcript does not contain the answer.
REFUSAL = "I don't have information about that in the transcript."


async def get_client() -> httpx.AsyncClient:
    """Get or create reusable HTTP client."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


async def generate_answer(question: str, context: str) -> str:
    """Generate an answer to a question grounded in the retrieved context."""
    return await _ollama_generate(_build_prompt(question, context))


def _build_prompt(question: str, context: str) -> str:
    """Build the grounded answer-generation prompt."""
    return f"""You are answering questions about a documentary using ONLY the \
transcript excerpts provided below.

Rules:
- Use ONLY information contained in the excerpts. Do not use any outside knowledge.
- If the excerpts do not contain the answer, reply with EXACTLY: "{REFUSAL}"
- Include specific details and timestamps from the excerpts where relevant.
- Do not speculate or invent information that is not in the excerpts.

Transcript excerpts:
{context}

Question: {question}

Answer:"""


async def _ollama_generate(prompt: str) -> str:
    """
    Call Ollama's /api/generate and return the response text.

    Raises:
        RuntimeError: If the Ollama request fails.
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
        return response.json().get("response", "")
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error with Ollama: {e}") from e


async def close_client():
    """Close the HTTP client (cleanup)."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
