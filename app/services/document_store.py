"""PostgreSQL persistence for uploaded document metadata and extracted text."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import psycopg
from psycopg.types.json import Jsonb

from app.services.vector_store import connect

DocumentStatus = Literal["uploaded", "processing", "processed", "failed"]


@dataclass
class StoredDocument:
    document_id: str
    filename: str
    content_type: str | None
    size_bytes: int
    character_count: int
    text: str
    segments: list[dict[str, str]]
    processing_status: DocumentStatus
    chunk_count: int
    embedding_model: str | None
    embedding_dimensions: int | None
    created_at: datetime
    updated_at: datetime


def initialize_schema(database_url: str | None = None) -> None:
    """Create metadata tables safely; existing document and vector rows are retained."""
    if database_url:
        with psycopg.connect(database_url) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
        return
    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)


_CREATE_TABLE = """CREATE TABLE IF NOT EXISTS contract_documents (
    document_id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    character_count BIGINT NOT NULL CHECK (character_count >= 0),
    extracted_text TEXT NOT NULL,
    source_segments JSONB NOT NULL DEFAULT '[]'::jsonb,
    processing_status TEXT NOT NULL DEFAULT 'uploaded'
        CHECK (processing_status IN ('uploaded', 'processing', 'processed', 'failed')),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    embedding_model TEXT,
    embedding_dimensions INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS contract_documents_updated_at_idx ON contract_documents (updated_at DESC)"
_COLUMNS = "document_id, filename, content_type, size_bytes, character_count, extracted_text, source_segments, processing_status, chunk_count, embedding_model, embedding_dimensions, created_at, updated_at"


def _document(row) -> StoredDocument:
    return StoredDocument(
        document_id=str(row[0]), filename=row[1], content_type=row[2], size_bytes=int(row[3]),
        character_count=int(row[4]), text=row[5], segments=row[6], processing_status=row[7],
        chunk_count=int(row[8]), embedding_model=row[9], embedding_dimensions=row[10],
        created_at=row[11], updated_at=row[12],
    )


def create_document(document: StoredDocument) -> StoredDocument:
    with connect() as conn:
        row = conn.execute(
            f"""INSERT INTO contract_documents
            (document_id, filename, content_type, size_bytes, character_count, extracted_text,
             source_segments, processing_status, chunk_count, embedding_model, embedding_dimensions)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {_COLUMNS}""",
            (document.document_id, document.filename, document.content_type, document.size_bytes,
             document.character_count, document.text, Jsonb(document.segments), document.processing_status,
             document.chunk_count, document.embedding_model, document.embedding_dimensions),
        ).fetchone()
    return _document(row)


def get_document(document_id: str) -> StoredDocument | None:
    with connect() as conn:
        row = conn.execute(f"SELECT {_COLUMNS} FROM contract_documents WHERE document_id = %s", (document_id,)).fetchone()
    return _document(row) if row else None


def list_documents(limit: int = 50, offset: int = 0) -> list[StoredDocument]:
    with connect() as conn:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM contract_documents ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        ).fetchall()
    return [_document(row) for row in rows]


def count_documents() -> int:
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM contract_documents").fetchone()
    return int(row[0])


def update_status(document_id: str, processing_status: DocumentStatus, *, chunk_count: int | None = None,
                  embedding_model: str | None = None, embedding_dimensions: int | None = None) -> StoredDocument | None:
    with connect() as conn:
        row = conn.execute(
            f"""UPDATE contract_documents SET processing_status = %s,
                chunk_count = COALESCE(%s, chunk_count), embedding_model = COALESCE(%s, embedding_model),
                embedding_dimensions = COALESCE(%s, embedding_dimensions), updated_at = NOW()
                WHERE document_id = %s RETURNING {_COLUMNS}""",
            (processing_status, chunk_count, embedding_model, embedding_dimensions, document_id),
        ).fetchone()
    return _document(row) if row else None


def check_database() -> None:
    with connect() as conn:
        conn.execute("SELECT 1").fetchone()
