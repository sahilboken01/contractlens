from fastapi.testclient import TestClient

from app.api import routes as route_package  # noqa: F401
from app.api.routes import documents
from app.core.config import Settings
from app.main import app


def test_health_docs_and_existing_txt_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "get_settings", lambda: Settings(upload_dir=tmp_path))
    monkeypatch.setattr(documents.document_store, "create_document", lambda document: document)
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    preflight = client.options(
        "/api/documents",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
    response = client.post(
        "/api/documents",
        files={"file": ("agreement.txt", b"Payment is due within thirty days.", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert "Payment is due within thirty days." in body["text"]
    assert body["segments"][0]["locator"] == "Document"
    assert (tmp_path / f"{body['document_id']}.txt").exists()
