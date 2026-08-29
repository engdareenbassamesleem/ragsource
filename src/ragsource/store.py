import json
from pathlib import Path
from threading import RLock

from ragsource.embeddings import cosine_similarity
from ragsource.models import Chunk


class JsonVectorStore:
    """Small, transparent vector store suitable for a portfolio MVP."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._chunks = [Chunk.model_validate(item) for item in payload]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps([item.model_dump() for item in self._chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def replace_document(self, document_id: str, chunks: list[Chunk]) -> None:
        with self._lock:
            self._chunks = [c for c in self._chunks if c.document_id != document_id]
            self._chunks.extend(chunks)
            self._save()

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            scores = cosine_similarity(query_vector, [c.embedding for c in self._chunks])
            ranked = np_argsort_desc(scores)[:top_k]
            return [(self._chunks[i], float(scores[i])) for i in ranked]

    def count(self) -> int:
        return len(self._chunks)


def np_argsort_desc(values):
    return values.argsort()[::-1].tolist()
