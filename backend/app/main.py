"""FastAPI app. Run with:  uvicorn app.main:app --reload   (from the backend folder)"""
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import CORS_ORIGINS, GROQ_API_KEY, GROQ_MODEL, MIN_SCORE, TOP_K
from .rag import GroqLLM, RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the embedding model and index once, not on every request (tests inject their own)
    if getattr(app.state, "rag", None) is None:
        from .index import load_retriever

        app.state.rag = RAGService(load_retriever(), GroqLLM(GROQ_API_KEY, GROQ_MODEL), TOP_K, MIN_SCORE)
    yield


app = FastAPI(title="Python Docs RAG API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=10)


@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "chunks": len(request.app.state.rag.retriever.chunks)}


@app.post("/api/ask")
def ask(payload: AskRequest, request: Request):
    """The whole answer at once (handy for scripts and the evaluation)."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please type a question.")
    try:
        return request.app.state.rag.answer(question, [t.model_dump() for t in payload.history])
    except Exception as e:  # the LLM provider failed: say so instead of a bare 500
        raise HTTPException(status_code=502, detail=f"The language model is unavailable right now ({type(e).__name__}).")


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest, request: Request):
    """Server-Sent Events: the answer appears word by word in the UI."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please type a question.")
    service = request.app.state.rag
    history = [t.model_dump() for t in payload.history]

    def events():
        try:
            for event in service.stream(question, history):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'The language model is unavailable right now ({type(e).__name__}).'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
