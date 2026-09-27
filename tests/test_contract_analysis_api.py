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


class MockAnalysisProvider:
    def __init__(self, response=None, error=None):
        self.response = response or json.dumps(
            {
                "contract_type": {
                    "value": "Professional Services Agreement",
                    "basis": "explicit",
                    "source_labels": ["C1"],
                },
                "parties": [
                    {"value": "Northwind Labs", "basis": "explicit", "source_labels": ["C1"]},
                    {"value": "Acme Consulting", "basis": "explicit", "source_labels": ["C1"]},
                ],
                "effective_date": {"value": "January 1, 2025", "basis": "explicit", "source_labels": ["C1"]},
                "payment_terms": [
                    {"value": "Invoices are due within 30 days.", "basis": "explicit", "source_labels": ["C1"]}
                ],
            }
        )
        self.error = error
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def analysis_api(tmp_path, monkeypatch):
    document_id = str(uuid4())
    (tmp_path / f"{document_id}.txt").write_text(
        "Services Agreement between Northwind Labs and Acme Consulting. Effective January 1, 2025. "
        "Invoices are due within 30 days.",
        encoding="utf-8",
    )
    settings = Settings(upload_dir=tmp_path, analysis_top_k=1, rag_min_similarity=0.35)
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    stored_document = SimpleNamespace(document_id=document_id, filename="agreement.txt", content_type="text/plain", size_bytes=136,
        character_count=136, text="Services Agreement between Northwind Labs and Acme Consulting. Effective January 1, 2025. Invoices are due within 30 days.",
        segments=[], processing_status="processed", chunk_count=1, embedding_model="bge", embedding_dimensions=384,
        created_at=None, updated_at=None)
    monkeypatch.setattr(documents.document_store, "get_document", lambda requested: stored_document if requested == document_id else None)
    chunk = StoredChunk(
        "chunk-contract-1",
        "Services Agreement between Northwind Labs and Acme Consulting. Effective January 1, 2025. Invoices are due within 30 days.",
        "Page 4",
        4,
        0.88,
        {"section": "Parties and Payment"},
        document_id,
    )
    calls = []

    def search(scoped_id, vector, top_k):
        calls.append((scoped_id, vector, top_k))
        return [chunk][:top_k] if scoped_id == document_id else []

    monkeypatch.setattr(documents.vector_store, "search", search)
    monkeypatch.setattr("app.services.contract_analysis.embed_texts", lambda texts: [[0.1, 0.9] for _ in texts])
    provider = MockAnalysisProvider()
    monkeypatch.setattr(documents, "create_llm_provider", lambda: provider)
    return {
        "client": TestClient(app),
        "document_id": document_id,
        "chunk": chunk,
        "calls": calls,
        "provider": provider,
        "tmp_path": tmp_path,
        "monkeypatch": monkeypatch,
    }


def analyze(fixture, document_id=None):
    return fixture["client"].post(f"/api/documents/{document_id or fixture['document_id']}/analyze")


def test_successful_analysis_returns_structured_grounded_fields(analysis_api):
    response = analyze(analysis_api)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["contract_type"]["value"] == "Professional Services Agreement"
    assert body["parties"][0]["value"] == "Northwind Labs"
    assert body["payment_terms"][0]["value"] == "Invoices are due within 30 days."
    assert "only the supplied context" in analysis_api["provider"].calls[0][0].lower()


def test_response_schema_validates_and_preserves_explicit_basis(analysis_api):
    response = analyze(analysis_api)
    parsed = documents.ContractAnalysisResponse.model_validate(response.json())

    assert parsed.contract_type is not None
    assert parsed.contract_type.basis == "explicit"
    assert parsed.effective_date.value == "January 1, 2025"


def test_missing_contract_information_is_null_or_empty(analysis_api):
    analysis_api["provider"].response = json.dumps({"parties": []})

    response = analyze(analysis_api)

    assert response.status_code == 200
    body = response.json()
    assert body["effective_date"] is None
    assert body["expiration_date"] is None
    assert body["renewal_terms"] == []
    assert body["governing_law"] is None
    assert body["status"] == "insufficient_context"


def test_citation_source_chunk_locator_page_and_similarity_are_mapped(analysis_api):
    response = analyze(analysis_api)

    assert response.status_code == 200
    citation = response.json()["payment_terms"][0]["sources"][0]
    assert citation == {
        "chunk_id": "chunk-contract-1",
        "source_locator": "Page 4",
        "page_number": 4,
        "similarity": 0.88,
    }


def test_retrieval_is_scoped_to_requested_document(analysis_api):
    doc_id = analysis_api["document_id"]

    response = analyze(analysis_api, doc_id)

    assert response.status_code == 200
    assert analysis_api["calls"]
    assert all(call[0] == doc_id for call in analysis_api["calls"])
    assert response.json()["contract_type"]["sources"][0]["chunk_id"] == "chunk-contract-1"


def test_document_not_found_returns_404_before_database_lookup(analysis_api, monkeypatch):

    response = analyze(analysis_api, str(uuid4()))

    assert response.status_code == 404


def test_unprocessed_document_returns_409(analysis_api, monkeypatch):
    monkeypatch.setattr(documents.document_store, "get_document", lambda _: SimpleNamespace(document_id=analysis_api["document_id"], filename="a", content_type=None, size_bytes=1, character_count=0, text="x", segments=[], processing_status="uploaded", chunk_count=0, embedding_model=None, embedding_dimensions=None, created_at=None, updated_at=None))

    response = analyze(analysis_api)

    assert response.status_code == 409
    assert "not been processed" in response.json()["detail"]


def test_no_relevant_retrieval_returns_empty_grounded_analysis(analysis_api, monkeypatch):
    monkeypatch.setattr(
        documents.vector_store,
        "search",
        lambda *_: [StoredChunk("weak", "irrelevant", "Page 9", 9, 0.1, {})],
    )

    response = analyze(analysis_api)

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_context"
    assert response.json()["contract_type"] is None
    assert response.json()["parties"] == []
    assert analysis_api["provider"].calls == []


def test_llm_failure_returns_safe_502(analysis_api, monkeypatch):
    provider = MockAnalysisProvider(error=LLMProviderError("sensitive provider response"))
    monkeypatch.setattr(documents, "create_llm_provider", lambda: provider)

    response = analyze(analysis_api)

    assert response.status_code == 502
    assert "sensitive" not in response.text
    assert response.json()["detail"] == "Contract analysis generation failed."


def test_malformed_llm_response_returns_safe_502(analysis_api):
    analysis_api["provider"].response = "not valid JSON"

    response = analyze(analysis_api)

    assert response.status_code == 502
    assert response.json()["detail"] == "The analysis provider returned an invalid response."


def test_missing_or_invented_citations_suppress_unsupported_finding(analysis_api):
    analysis_api["provider"].response = json.dumps(
        {"governing_law": {"value": "California", "basis": "explicit", "source_labels": ["invented"]}}
    )

    response = analyze(analysis_api)

    assert response.status_code == 200
    assert response.json()["governing_law"] is None
    assert response.json()["status"] == "insufficient_context"


def test_empty_contract_text_returns_422_before_retrieval(analysis_api, monkeypatch):
    monkeypatch.setattr(documents.document_store, "get_document", lambda _: SimpleNamespace(
        document_id=analysis_api["document_id"], filename="a", content_type=None, size_bytes=1,
        character_count=0, text="   ", segments=[], processing_status="processed", chunk_count=1,
        embedding_model="bge", embedding_dimensions=384, created_at=None, updated_at=None,
    ))

    response = analyze(analysis_api)

    assert response.status_code == 422
    assert "empty" in response.json()["detail"]


def test_openapi_documents_analysis_schema(analysis_api):
    response = analysis_api["client"].get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"]["/api/documents/{document_id}/analyze"]["post"]
    assert operation["summary"] == "Analyze a processed contract"
    assert "ContractAnalysisResponse" in str(operation["responses"]["200"])
