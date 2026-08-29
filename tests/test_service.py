import hashlib

from ragsource.models import Chunk
from ragsource.service import RagService
from ragsource.store import JsonVectorStore


class KeywordEmbedder:
    vocabulary = ("python", "firebase", "dentist")

    def encode(self, texts):
        return [[float(text.lower().count(word)) for word in self.vocabulary] for text in texts]


class RecordingGenerator:
    def __init__(self):
        self.context = ""

    def answer(self, question, context):
        self.context = context
        return "Python is discussed in the source [S1]."


def make_chunk(text, embedding, page=1):
    return Chunk(
        id=hashlib.md5(text.encode(), usedforsecurity=False).hexdigest(),
        document_id="doc-1",
        filename="guide.pdf",
        page=page,
        text=text,
        embedding=embedding,
    )


def test_returns_ranked_sources_with_page_citation(tmp_path):
    store = JsonVectorStore(tmp_path / "index.json")
    embedder = KeywordEmbedder()
    store.replace_document(
        "doc-1",
        [
            make_chunk("Python powers the API.", embedder.encode(["python"])[0], page=4),
            make_chunk("Firebase stores data.", embedder.encode(["firebase"])[0], page=7),
        ],
    )
    generator = RecordingGenerator()
    result = RagService(store, embedder, generator, minimum_score=0.1).ask("Python?", 2)

    assert result.grounded is True
    assert result.sources[0].filename == "guide.pdf"
    assert result.sources[0].page == 4
    assert "[S1] File: guide.pdf, page 4" in generator.context


def test_refuses_when_retrieval_is_not_supported(tmp_path):
    store = JsonVectorStore(tmp_path / "index.json")
    embedder = KeywordEmbedder()
    store.replace_document("doc-1", [make_chunk("Python", [1.0, 0.0, 0.0])])
    result = RagService(store, embedder, RecordingGenerator(), minimum_score=0.5).ask(
        "unrelated question", 3
    )
    assert result.grounded is False
    assert result.sources == []
