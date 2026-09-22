"""Application models."""

from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Chat request."""

    question: str


class ResponseMetadata(BaseModel):
    response_time_ms: int
    retrieved_chunks: int
    source_documents: int
    llm_model: str
    embedding_model: str
    vector_store: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    metadata: ResponseMetadata