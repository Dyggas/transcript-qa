"""Centralised configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM provider
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "zai")

    # z.ai
    ZAI_API_KEY: str = os.getenv("ZAI_API_KEY", "")
    ZAI_BASE_URL: str = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")
    ZAI_MODEL: str = os.getenv("ZAI_MODEL", "glm-4-flash")

    # Ollama
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # Data
    TRANSCRIPT_PATH: str = os.getenv("TRANSCRIPT_PATH", "data/transcript.txt")


settings = Settings()
