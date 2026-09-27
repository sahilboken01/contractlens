from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from app.services.embeddings import EmbeddingConfigurationError

from app.api.routes import documents
from app.core.config import Settings
from app.main import app


def _document(document_id: str, status: str = "processed") -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        document_id=document_id, filename="persisted.txt", content_type="text/plain", size_bytes=38,
        character_count=38, text="Payment is due within 30 calendar days.",
        segments=[{"locator": "Section 4 - Fees", "text": "Payment is due within 30 calendar days."}],
        processing_status=status, chunk_count=2 if status == "processed" else 0,
        embedding_model="BAAI/bge-small-en-v1.5" if status == "processed" else None,
        embedding_dimensions=384 if status == "processed" else None, created_at=now, updated_at=now,
    )


def test_document_history_metadata_text_and_missing(monkeypatch):
    document_id = str(uuid4())
    record = _document(document_id)
    monkeypatch.setattr(documents.document_store, "list_documents", lambda limit, offset: [record])
    monkeypatch.setattr(documents.document_store, "count_documents", lambda: 1)
    monkeypatch.setattr(documents.document_store, "get_document", lambda requested: record if requested == document_id else None)
    client = TestClient(app)

    history = client.get("/api/documents")
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["documents"][0]["processing_status"] == "processed"
    assert "text" not in history.json()["documents"][0]

    metadata = client.get(f"/api/documents/{document_id}")
    assert metadata.status_code == 200
    assert metadata.json()["chunk_count"] == 2
    text = client.get(f"/api/documents/{document_id}/text")
    assert text.json()["text"] == record.text
    assert text.json()["segments"][0]["locator"] == "Section 4 - Fees"
    assert client.get(f"/api/documents/{uuid4()}").status_code == 404
    assert client.get(f"/api/documents/{uuid4()}/text").status_code == 404


def test_document_processing_lifecycle_is_persisted(monkeypatch):
    document_id = str(uuid4())
    record = _document(document_id, "uploaded")
    events: list[str] = []
    monkeypatch.setattr(documents, "get_settings", lambda: Settings(chunk_size_chars=400, chunk_overlap_chars=20))
    monkeypatch.setattr(documents.document_store, "get_document", lambda requested: record if requested == document_id else None)

    def update(requested, state, **kwargs):
        events.append(state)
        record.processing_status = state
        record.chunk_count = kwargs.get("chunk_count", record.chunk_count)
        return record

    monkeypatch.setattr(documents.document_store, "update_status", update)
    monkeypatch.setattr(documents, "embed_texts", lambda texts: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(documents.vector_store, "replace_document_chunks", lambda *_: None)
    client = TestClient(app)

    response = client.post(f"/api/documents/{document_id}/process")
    assert response.status_code == 200
    assert events == ["processing", "processed"]
    assert response.json()["processing_status"] == "processed"
    assert record.chunk_count > 0


def test_failed_processing_is_recorded_and_not_marked_ready(monkeypatch):
    document_id = str(uuid4())
    record = _document(document_id, "uploaded")
    events: list[str] = []
    monkeypatch.setattr(documents, "get_settings", lambda: Settings(chunk_size_chars=400, chunk_overlap_chars=20))
    monkeypatch.setattr(documents.document_store, "get_document", lambda requested: record if requested == document_id else None)

    def update(requested, state, **kwargs):
        events.append(state)
        record.processing_status = state
        record.chunk_count = kwargs.get("chunk_count", record.chunk_count)
        return record

    monkeypatch.setattr(documents.document_store, "update_status", update)
    monkeypatch.setattr(documents, "embed_texts", lambda _: (_ for _ in ()).throw(EmbeddingConfigurationError("unavailable")))
    response = TestClient(app).post(f"/api/documents/{document_id}/process")
    assert response.status_code == 503
    assert events == ["processing", "failed"]
    assert record.processing_status == "failed"
    assert record.chunk_count == 0


def test_cors_configuration_is_explicit_and_production_validates_origins():
    settings = Settings(cors_origins="https://contracts.example, https://www.example")
    assert settings.allowed_origins == ["https://contracts.example", "https://www.example"]
    with pytest.raises(ValidationError):
        Settings(
            environment="production",
            database_url="postgresql://placeholder",
            llm_api_key="placeholder",
            cors_origins="*",
        )

