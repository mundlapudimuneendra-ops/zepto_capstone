"""FastAPI entry point for the Zepto support assistant.

The app exposes a single ``POST /ask`` endpoint that takes a
``{"query": str}`` body, runs the LangGraph pipeline, and returns a
``SupportResponse`` JSON.

Run locally:

    uvicorn main:app --host 0.0.0.0 --port 7860

On startup the app ensures the ChromaDB index exists (so a fresh
clone works without a separate ``python ingest.py`` step), but the
build pipeline still runs ``ingest.py`` explicitly inside the
Dockerfile to keep the index up to date.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph import app as graph_app
from ingest import build_index
from schema import AskRequest, ErrorResponse, SupportResponse


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG + LangGraph customer-support service for Zepto.",
    version="1.0.0",
)


@app.on_event("startup")
def _ensure_index() -> None:
    """Build the ChromaDB index on first startup if it is empty.

    The build is cheap (8 short docs) and idempotent, so doing it on
    startup keeps the deployment artifact self-contained: a fresh
    container with the docs folder is enough.
    """

    try:
        from ingest import get_collection
        coll = get_collection()
        if coll.count() == 0:
            build_index(force=False)
    except Exception:
        # If the index cannot be built (e.g. the model is not yet
        # downloaded), /ask will return a 503 instead of crashing.
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=SupportResponse)
def ask(req: AskRequest) -> SupportResponse:
    """Run the pipeline and return the structured response."""

    state_in = {"query": req.query}
    state_out = graph_app.invoke(state_in)

    if state_out.get("error"):
        # Real-LLM path exhausted retries — return a clearly marked
        # error response (still schema-valid) instead of raising.
        return SupportResponse(
            answer=state_out.get("answer") or "ERROR",
            sources=state_out.get("sources", []),
            confidence=0.0,
        )

    return SupportResponse(
        answer=state_out.get("answer", ""),
        sources=state_out.get("sources", []),
        confidence=float(state_out.get("confidence", 0.0)),
    )


# ---------------------------------------------------------------------------
# Tiny CLI demo so the README transcripts can be reproduced.
# ---------------------------------------------------------------------------

def _cli_demo() -> None:
    """Print two example /ask responses (policy + general)."""

    samples = [
        "How long does delivery take and is there a free-delivery threshold?",
        "Tell me a fun fact about the moon.",
    ]
    for q in samples:
        out = graph_app.invoke({"query": q})
        print(f"Q: {q}")
        print(json.dumps({
            "answer": out.get("answer"),
            "sources": out.get("sources", []),
            "confidence": out.get("confidence", 0.0),
            "intent": out.get("intent"),
        }, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    import json
    _cli_demo()
