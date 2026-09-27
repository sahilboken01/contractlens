"""Backfill metadata/text records from legacy local upload files without deleting data."""

from datetime import datetime, timezone
from uuid import UUID

from app.core.config import get_settings
from app.services.document_extractor import extract_document
from app.services.document_store import StoredDocument, create_document, get_document
from app.services.vector_store import count_document_chunks


def backfill() -> int:
    settings = get_settings()
    migrated = 0
    for path in settings.upload_dir.iterdir():
        if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            continue
        try:
            document_id = str(UUID(path.stem))
        except ValueError:
            continue
        if get_document(document_id):
            continue
        extracted = extract_document(path.name, path.read_bytes())
        chunks = count_document_chunks(document_id)
        now = datetime.now(timezone.utc)
        create_document(StoredDocument(
            document_id=document_id, filename=path.name, content_type=None,
            size_bytes=path.stat().st_size, character_count=len(extracted.text), text=extracted.text,
            segments=[{"locator": item.locator, "text": item.text} for item in extracted.segments],
            processing_status="processed" if chunks else "uploaded", chunk_count=chunks,
            embedding_model=settings.embedding_model if chunks else None,
            embedding_dimensions=settings.embedding_dimensions if chunks else None,
            created_at=now, updated_at=now,
        ))
        migrated += 1
    return migrated


if __name__ == "__main__":
    print(f"Backfilled {backfill()} legacy document record(s).")
