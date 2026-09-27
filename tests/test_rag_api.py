import json
from uuid import uuid4
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.routes import documents
from app.core.config import Settings
from app.main import app
from app.services.llm import LLMProviderError
from app.services.vector_store import StoredChunk


class DeterministicProvider:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {"answer": "The notice period is 30 days.", "citations": ["S1"]}
        self.error = error
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return json.dumps(self.payload)


@pytest.fixture
def rag_api(tmp_path, monkeypatch):
    document_id = str(uuid4())
    (tmp_path / f"{document_id}.txt").write_text("Termination requires 30 days' written notice.", encoding="utf-8")
    settings = Settings(upload_dir=tmp_path, retrieval_top_k=3, rag_min_similarity=0.35)
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    stored_document = SimpleNamespace(document_id=document_id, filename="agreement.txt", content_type="text/plain", size_bytes=44,
        character_count=44, text="Termination requires 30 days' written notice.", segments=[], processing_status="processed", chunk_count=2,
        embedding_model="bge", embedding_dimensions=384, created_at=None, updated_at=None)
    monkeypatch.setattr(documents.document_store, "get_document", lambda requested: stored_document if requested == document_id else None)
    chunks = [
        StoredChunk("chunk-a", "Termination requires 30 days' written notice.", "Section 8 - Termination", 4, 0.87, {}, document_id),
        StoredChunk("chunk-b", "This agreement is governed by New York law.", "Section 12 - Governing Law", 7, 0.71, {}, document_id),
    ]
    search_calls = []

    def search(scoped_document_id, embedding, top_k):
        search_calls.append((scoped_document_id, embedding, top_k))
        return chunks[:top_k]

    monkeypatch.setattr(documents.vector_store, "search", search)
    monkeypatch.setattr("app.services.rag.embed_texts", lambda texts: [[0.25, 0.75]])
    provider = DeterministicProvider()
    monkeypatch.setattr(documents, "create_llm_provider", lambda: provider)
    return {
        "client": TestClient(app),
        "document_id": document_id,
        "settings": settings,
        "chunks": chunks,
        "search_calls": search_calls,
        "provider": provider,
        "tmp_path": tmp_path,
        "monkeypatch": monkeypatch,
    }


def test_successful_answer_and_deterministic_mock_provider(rag_api):
    client = rag_api["client"]
    doc_id = rag_api["document_id"]

    first = client.post(f"/api/documents/{doc_id}/ask", json={"question": "What notice is required?"})
    second = client.post(f"/api/documents/{doc_id}/ask", json={"question": "What notice is required?"})

    assert first.status_code == 200
    assert first.json()["answer"] == "The notice period is 30 days."
    assert second.json()["answer"] == first.json()["answer"]
    assert len(rag_api["provider"].calls) == 2
    system_prompt, user_prompt = rag_api["provider"].calls[0]
    assert "only the supplied contract context" in system_prompt
    assert "Section 8 - Termination" in user_prompt


def test_retrieval_is_scoped_to_requested_document(rag_api):
    doc_id = rag_api["document_id"]
    response = rag_api["client"].post(f"/api/documents/{doc_id}/ask", json={"question": "What notice is required?"})

    assert response.status_code == 200
    assert rag_api["search_calls"] == [(doc_id, [0.25, 0.75], 3)]


def test_missing_document_returns_404_before_database_lookup(rag_api, monkeypatch):
    response = rag_api["client"].post(f"/api/documents/{uuid4()}/ask", json={"question": "What is the term?"})

    assert response.status_code == 404


def test_empty_question_is_rejected(rag_api):
    response = rag_api["client"].post(
        f"/api/documents/{rag_api['document_id']}/ask", json={"question": "   "}
    )

    assert response.status_code == 422


def test_unprocessed_document_returns_409(rag_api, monkeypatch):
    monkeypatch.setattr(documents.document_store, "get_document", lambda _: SimpleNamespace(document_id=rag_api["document_id"], filename="a", content_type=None, size_bytes=1, character_count=1, text="x", segments=[], processing_status="uploaded", chunk_count=0, embedding_model=None, embedding_dimensions=None, created_at=None, updated_at=None))

    response = rag_api["client"].post(
        f"/api/documents/{rag_api['document_id']}/ask", json={"question": "What is the term?"}
    )

    assert response.status_code == 409
    assert "not been processed" in response.json()["detail"]


def test_no_relevant_chunks_returns_abstention_without_llm(rag_api, monkeypatch):
    monkeypatch.setattr(
        documents.vector_store,
        "search",
        lambda *_: [StoredChunk("weak", "Unrelated text", "Page 9", 9, 0.1, {})],
    )

    response = rag_api["client"].post(
        f"/api/documents/{rag_api['document_id']}/ask", json={"question": "What is the liability cap?"}
    )

    assert response.status_code == 200
    assert "does not contain enough information" in response.json()["answer"]
    assert response.json()["sources"] == []
    assert rag_api["provider"].calls == []


def test_llm_failure_returns_safe_502(rag_api, monkeypatch):
    provider = DeterministicProvider(error=LLMProviderError("secret provider response"))
    monkeypatch.setattr(documents, "create_llm_provider", lambda: provider)

    response = rag_api["client"].post(
        f"/api/documents/{rag_api['document_id']}/ask", json={"question": "What notice is required?"}
    )

    assert response.status_code == 502
    assert "secret" not in response.text
    assert response.json()["detail"] == "Answer generation failed."


def test_citation_metadata_is_preserved(rag_api):
    response = rag_api["client"].post(
        f"/api/documents/{rag_api['document_id']}/ask", json={"question": "What notice is required?"}
    )

    assert response.status_code == 200
    assert response.json()["sources"] == [
        {
            "chunk_id": "chunk-a",
            "source_locator": "Section 8 - Termination",
            "page_number": 4,
            "similarity": 0.87,
        }
    ]


def test_provider_cannot_invent_citation_ids(rag_api, monkeypatch):
    provider = DeterministicProvider(
        payload={"answer": "The contract states 30 days.", "citations": ["S1", "unretrieved-chunk"]}
    )
    monkeypatch.setattr(documents, "create_llm_provider", lambda: provider)

    response = rag_api["client"].post(
        f"/api/documents/{rag_api['document_id']}/ask", json={"question": "What notice is required?"}
    )

    assert response.status_code == 200
    assert [source["chunk_id"] for source in response.json()["sources"]] == ["chunk-a"]


def test_answer_without_any_valid_citation_abstains(rag_api, monkeypatch):
    provider = DeterministicProvider(
        payload={"answer": "The contract allows immediate termination.", "citations": ["not-a-source"]}
    )
    monkeypatch.setattr(documents, "create_llm_provider", lambda: provider)

    response = rag_api["client"].post(
        f"/api/documents/{rag_api['document_id']}/ask", json={"question": "Can it terminate immediately?"}
    )

    assert response.status_code == 200
    assert "does not contain enough information" in response.json()["answer"]
    assert response.json()["sources"] == []


def test_openapi_documents_ask_request_and_response(rag_api):
    response = rag_api["client"].get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"]["/api/documents/{document_id}/ask"]["post"]
    assert operation["summary"] == "Ask a question about one uploaded contract"
    assert "AskRequest" in str(operation["requestBody"])
    assert "AskResponse" in str(operation["responses"]["200"])
