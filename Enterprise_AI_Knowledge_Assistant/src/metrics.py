"""Prometheus metrics."""
from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Summary,
    generate_latest,
)

# -------------------------
# API Metrics
# -------------------------

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total RAG requests",
)

REQUEST_FAILURES = Counter(
    "rag_request_failures_total",
    "Failed RAG requests",
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "RAG request latency",
)

# -------------------------
# LLM Metrics
# -------------------------

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM requests",
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "LLM failures",
)


RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "Document retrieval latency",
)

LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "LLM generation latency",
)

DOCUMENTS_RETRIEVED = Summary(
    "retrieved_documents",
    "Number of retrieved documents",
)

def metrics_endpoint() -> Response:
    """Expose Prometheus metrics."""

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )