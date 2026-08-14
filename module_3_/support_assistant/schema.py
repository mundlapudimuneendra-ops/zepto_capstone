"""Pydantic request/response models and LangGraph state schema.

This file defines the wire format exposed by the FastAPI app and the
internal state carried between LangGraph nodes. Keeping them in one
place makes it easy to validate every response before it leaves the
process, and keeps the graph state self-documenting.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Public API request/response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Inbound request body for ``POST /ask``."""

    query: str = Field(..., description="User's natural-language question.")


class SupportResponse(BaseModel):
    """Outbound response body for ``POST /ask``.

    ``sources`` lists the chunk/doc IDs that were used to produce the
    answer. For ``general_question`` it is empty. ``confidence`` is a
    heuristic 0-1 score; in mock mode it is a fixed value, in real-LLM
    mode it is parsed from the model's structured output.
    """

    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    """Returned when the real-LLM path fails schema validation after retries."""

    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    error: str


# ---------------------------------------------------------------------------
# LangGraph internal state
# ---------------------------------------------------------------------------

# Literal intent labels the classifier can emit. Kept module-local so
# the conditional edge can use them as enum-style strings.
INTENT_POLICY = "policy_question"
INTENT_GENERAL = "general_question"


class GraphState(TypedDict, total=False):
    """State carried through the LangGraph pipeline.

    All keys are optional (``total=False``) so nodes can populate only
    the fields they own without forcing every node to provide a full
    initial state.
    """

    # Inbound
    query: str

    # Classification
    intent: str  # one of INTENT_POLICY / INTENT_GENERAL

    # Retrieval (populated by retrieve_and_answer)
    retrieved_chunks: List[dict]  # each: {"id": str, "text": str, "score": float}
    top_chunk_snippet: str

    # Generation
    answer: str
    sources: List[str]
    confidence: float

    # Error tracking (real-LLM path only)
    error: Optional[str]
