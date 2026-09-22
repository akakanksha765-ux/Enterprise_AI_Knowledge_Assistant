"""RAG utilities."""

import time

from src.config import settings, opik
from src.llm import check_ollama, generate
from src.metrics import (
    DOCUMENTS_RETRIEVED,
    LLM_LATENCY,
    REQUEST_COUNT,
    REQUEST_FAILURES,
    REQUEST_LATENCY,
    RETRIEVAL_LATENCY,
)
from src.models import ChatResponse,ResponseMetadata
from src.prompts import Prompts
from src.vector_store import initialize_vector_store


# =============================================================================
# Retrieval utilities
# =============================================================================


def get_retriever(vector_store):
    """Return document retriever."""

    return vector_store.as_retriever(
        search_kwargs={"k": 4},
    )


def format_context(documents):
    """Create LLM context."""

    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


def get_sources(documents):
    """Return unique source names."""

    return list(
        {
            doc.metadata.get("source", "")
            for doc in documents
        }
    )


def get_document_preview(documents):
    """
    Small preview of retrieved documents for Opik.
    Keeps traces readable.
    """

    previews = []

    for index, doc in enumerate(documents, start=1):

        previews.append(
            {
                "rank": index,
                "source": doc.metadata.get("source", "Unknown"),
                "preview": doc.page_content[:400],
            }
        )

    return previews


def retrieve(question: str):
    """Retrieve relevant documents."""

    vector_store = initialize_vector_store()
    retriever = get_retriever(vector_store)

    return retriever.invoke(question)


# =============================================================================
# Main RAG Pipeline
# =============================================================================

def answer_question(question: str) -> ChatResponse:
    """Answer a user question."""

    REQUEST_COUNT.inc()
    request_start = time.perf_counter()

    trace = opik.trace(
        name="RAG Request",
        input={
            "question": question,
        },
    )

    retrieval_span = None
    generation_span = None

    try:

        # =====================================================================
        # Retrieval
        # =====================================================================

        retrieval_start = time.perf_counter()

        retrieval_span = trace.span(
            name="Retrieval",
            type="tool",
            input={
                "question": question,
            },
        )

        documents = retrieve(question)

        DOCUMENTS_RETRIEVED.observe(len(documents))

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        RETRIEVAL_LATENCY.observe(
            retrieval_latency
        )

        context = format_context(documents)

        sources = get_sources(documents)

        retrieval_span.update(
            output={
                "documents_retrieved": len(documents),
                "sources": sources,
                "retrieved_documents": get_document_preview(documents),
                "context_characters": len(context),
            }
        )

        retrieval_span.end()

        # =====================================================================
        # LLM Generation
        # =====================================================================

        if not check_ollama():
            raise RuntimeError(
                "Ollama server is not running. "
                "Start it using 'ollama serve'."
            )

        system_prompt = Prompts.SYSTEM

        user_prompt = Prompts.rag(
            context=context,
            question=question,
        )

        llm_start = time.perf_counter()

        generation_span = trace.span(
            name="Generation",
            type="llm",
            model=settings.llm_model,
            provider="ollama",
            input={
                "question": question,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context_characters": len(context),
                "context_chunks": len(documents),
            },
        )

        answer = generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        llm_latency = (
            time.perf_counter()
            - llm_start
        )

        LLM_LATENCY.observe(
            llm_latency
        )

        generation_span.update(
            output={
                "answer": answer,
            }
        )

        generation_span.end()

        total_latency = (
            time.perf_counter()
            - request_start
        )

        trace.update(
            output={
                "answer": answer,
            },
            metadata={
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model,
                "vector_store": "FAISS",
                "top_k": 4,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "documents_retrieved": len(documents),
                "source_documents": len(sources),
                "sources": sources,
                "retrieval_latency_seconds": round(
                    retrieval_latency,
                    3,
                ),
                "llm_latency_seconds": round(
                    llm_latency,
                    3,
                ),
                "total_latency_seconds": round(
                    total_latency,
                    3,
                ),
            },
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata=ResponseMetadata(
                response_time_ms=int(total_latency * 1000),
                retrieved_chunks=len(documents),
                source_documents=len(sources),
                llm_model=settings.llm_model,
                embedding_model=settings.embedding_model,
                vector_store="FAISS",
            ),
        )

    except Exception as e:

        REQUEST_FAILURES.inc()
        error_message = str(e)

        if generation_span is not None:
            generation_span.update(
                error_info={
                    "message": error_message,
                }
            )
            generation_span.end()

        if retrieval_span is not None:
            retrieval_span.update(
                error_info={
                    "message": error_message,
                }
            )
            retrieval_span.end()

        trace.update(
            error_info={
                "message": error_message,
            }
        )

        raise

    finally:

        REQUEST_LATENCY.observe(
            time.perf_counter() - request_start
        )

        trace.end()
        opik.flush()