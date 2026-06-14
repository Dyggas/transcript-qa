"""Pydantic models for API request and response."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="The user's natural-language question")


class Source(BaseModel):
    timestamp: str = Field(..., description="HH:MM:SS time code from the transcript")
    excerpt: str = Field(..., description="Excerpt from the transcript chunk")
    score: float = Field(..., description="Relevance score (cosine similarity, 0–1)")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
