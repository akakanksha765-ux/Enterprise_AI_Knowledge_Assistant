"""FastAPI application."""

from fastapi import FastAPI, HTTPException

from src.config import settings
from src.llm import check_ollama
from src.models import ChatRequest, ChatResponse
from src.rag import answer_question
from src.vector_store import rebuild_vector_store

from pydantic import BaseModel

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.logger import logger
from src.metrics import metrics_endpoint


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    ollama: bool


class VersionResponse(BaseModel):
    """Application version."""

    application: str
    version: str
    llm_model: str
    embedding_model: str


class StatusResponse(BaseModel):
    """Generic status response."""

    status: str
    message: str



app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description="LLMOps demo using LangChain, Ollama and FAISS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add Request Logging Middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every API request."""

    async def dispatch(self, request: Request, call_next):

        start = time.perf_counter()

        response = await call_next(request)

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s -> %d (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )

        return response


app.add_middleware(LoggingMiddleware)



@app.get("/",tags=["General"])
def root() -> dict[str, str]:
    """Root endpoint."""

    return {"message": "Enterprise AI Knowledge Assistant"}


@app.get("/health", tags=["General"],response_model=HealthResponse)
def health() -> dict:
    """Application health."""

    return {
        "status": "healthy",
        "ollama": check_ollama(),
    }


@app.get("/version", tags=["General"],response_model=VersionResponse)
def version() -> dict:
    """Application version."""

    return {
        "application": "Enterprise AI Knowledge Assistant",
        "version": app.version,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/chat", tags=["RAG"],response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer user question."""

    if not check_ollama():
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not running.",
        )

    return answer_question(request.question)



@app.post("/reindex")
def reindex() -> dict[str, str]:
    """Force rebuild vector index."""

    rebuild_vector_store()

    return {
        "status": "success",
        "message": "Vector index rebuilt.",
    }


@app.get("/test-llm", tags=["LLM"],response_model=StatusResponse)
def test_llm() -> dict[str, str]:
    """Simple Ollama connectivity test."""

    if not check_ollama():
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not running.",
        )

    return {
        "status": "success",
        "message": "Ollama is running.",
    }

@app.get("/metrics")
def metrics():
    """Prometheus metrics."""

    return metrics_endpoint()