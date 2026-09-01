from fastapi.testclient import TestClient

from ragsource.api import create_app, get_container
from ragsource.config import Settings
from ragsource.models import AskResponse, IngestResponse


class FakeStore:
    def count(self):
        return 3


class FakeIngestion:
    def ingest(self, filename, content):
        if content == b"broken":
            raise ValueError("The uploaded file is not a readable PDF")
        return IngestResponse(document_id="abc", filename=filename, pages=1, chunks=2)


class FakeRag:
    def ask(self, question, top_k):
        return AskResponse(answer="Grounded answer [S1].", sources=[], grounded=True)


class FakeContainer:
    settings = Settings(max_upload_mb=1)
    store = FakeStore()
    ingestion = FakeIngestion()
    rag = FakeRag()


def client():
    app = create_app()
    app.dependency_overrides[get_container] = lambda: FakeContainer()
    return TestClient(app)


def test_health():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "indexed_chunks": 3}


def test_ask():
    response = client().post("/v1/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert response.json()["grounded"] is True


def test_rejects_non_pdf():
    response = client().post("/v1/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_rejects_unreadable_pdf():
    response = client().post(
        "/v1/documents", files={"file": ("broken.pdf", b"broken", "application/pdf")}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a readable PDF"
