"""Pydantic schemas for structured output mode."""
from __future__ import annotations
from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    id: str
    title: str
    score: float


class StructuredAnswer(BaseModel):
    """Schema for JSON-mode answers."""

    answer: str = Field(..., description="Final answer for the user")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    citations: List[Citation] = Field(default_factory=list)
    follow_up: Optional[str] = None


class QueryResponseModel(BaseModel):
    ok: bool = True
    answer: str
    provider: str
    citations: List[Citation] = Field(default_factory=list)
    route: dict = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)
