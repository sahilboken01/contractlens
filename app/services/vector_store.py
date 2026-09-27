from dataclasses import dataclass
from typing import Protocol

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.core.config import get_settings
from app.services.chunking import DocumentChunk


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    source_locator: str
    page_number: int | None
    similarity: float
    metadata: dict[str, object]
    document_id: str | None = None


class VectorStore(Protocol):
    def replace_document_chunks(self, document_id: str, chunks: list[DocumentChunk]) -> None: ...
    def search(self, document_id: str, query_embedding: list[float], top_k: int) -> list[StoredChunk]: ...


def connect(database_url: str | None = None):
    url = database_url or get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not configured. Set it in your .env file.")
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


def initialize_schema(database_url: str | None = None) -> None:
    dimensions = get_settings().database_dimensions
    url = database_url or get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not configured. Set it in your .env file.")
    with psycopg.connect(url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id UUID PRIMARY KEY,
                document_id UUID NOT NULL,
                chunk_text TEXT NOT NULL,
                source_locator TEXT NOT NULL,
                page_number INTEGER,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding VECTOR({dimensions}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx ON document_chunks (document_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
        )
    # Metadata/text history lives beside the vector index and is safe to initialize repeatedly.
    from app.services.document_store import initialize_schema as initialize_document_schema

    initialize_document_schema(database_url)


def replace_document_chunks(document_id: str, chunks: list[DocumentChunk]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
        with conn.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO document_chunks
                   (chunk_id, document_id, chunk_text, source_locator, page_number, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.text,
                        chunk.source_locator,
                        chunk.page_number,
                        psycopg.types.json.Jsonb(chunk.metadata),
                        Vector(chunk.embedding),
                    )
                    for chunk in chunks
                ],
            )


def search(document_id: str, query_embedding: list[float], top_k: int) -> list[StoredChunk]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT document_id, chunk_id, chunk_text, source_locator, page_number, metadata,
                      1 - (embedding <=> %s) AS similarity
               FROM document_chunks
               WHERE document_id = %s
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (Vector(query_embedding), document_id, Vector(query_embedding), top_k),
        ).fetchall()
    return [
        StoredChunk(
            document_id=str(row[0]),
            chunk_id=str(row[1]),
            text=row[2],
            source_locator=row[3],
            page_number=row[4],
            metadata=row[5],
            similarity=float(row[6]),
        )
        for row in rows
    ]


def count_document_chunks(document_id: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s",
            (document_id,),
        ).fetchone()
    return int(row[0])
