from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TextSegment(BaseModel):
    locator: str
    text: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str | None
    size_bytes: int
    character_count: int
    text: str
    segments: list[TextSegment]
    processing_status: Literal["uploaded", "processing", "processed", "failed"] = "uploaded"


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    content_type: str | None
    size_bytes: int
    character_count: int
    processing_status: Literal["uploaded", "processing", "processed", "failed"]
    chunk_count: int
    embedding_model: str | None
    embedding_dimensions: int | None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int


class DocumentTextResponse(BaseModel):
    document_id: str
    text: str
    segments: list[TextSegment]


class ProcessDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int
    processing_status: Literal["processed"] = "processed"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace characters")
        return value


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    source_locator: str
    page_number: int | None
    similarity: float
    metadata: dict[str, object]


class SearchResponse(BaseModel):
    document_id: str
    query: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must contain non-whitespace characters")
        return value


class AnswerSource(BaseModel):
    chunk_id: str
    source_locator: str
    page_number: int | None
    similarity: float


class AskResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    sources: list[AnswerSource]
