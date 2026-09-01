from types import SimpleNamespace

import pytest

from ragsource.ingestion import PdfIngestionService
from ragsource.store import JsonVectorStore


class DeterministicEmbedder:
    def encode(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


def test_ingestion_preserves_filename_and_page_number(tmp_path, monkeypatch):
    reader = SimpleNamespace(
        pages=[FakePage("First page evidence."), FakePage("Second page evidence.")]
    )
    monkeypatch.setattr("ragsource.ingestion.PdfReader", lambda _: reader)
    store = JsonVectorStore(tmp_path / "index.json")
    service = PdfIngestionService(store, DeterministicEmbedder(), 900, 150)

    result = service.ingest("policy.pdf", b"valid-pdf-fixture")
    matches = store.search([20.0, 1.0], top_k=5)

    assert result.pages == 2
    assert result.chunks == 2
    assert {chunk.page for chunk, _ in matches} == {1, 2}
    assert {chunk.filename for chunk, _ in matches} == {"policy.pdf"}


def test_ingestion_rejects_pdf_without_extractable_text(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ragsource.ingestion.PdfReader", lambda _: SimpleNamespace(pages=[FakePage("")])
    )
    service = PdfIngestionService(
        JsonVectorStore(tmp_path / "index.json"), DeterministicEmbedder(), 900, 150
    )

    with pytest.raises(ValueError, match="no extractable text"):
        service.ingest("scan.pdf", b"image-only")


def test_vector_store_persists_index(tmp_path):
    path = tmp_path / "index.json"
    store = JsonVectorStore(path)
    from ragsource.models import Chunk

    store.replace_document(
        "doc",
        [
            Chunk(
                id="chunk-1",
                document_id="doc",
                filename="guide.pdf",
                page=3,
                text="Persistent evidence",
                embedding=[1.0, 0.0],
            )
        ],
    )

    restored = JsonVectorStore(path)
    assert restored.count() == 1
    assert restored.search([1.0, 0.0], 1)[0][0].page == 3
