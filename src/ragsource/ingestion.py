import hashlib
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ragsource.chunking import chunk_text
from ragsource.embeddings import Embedder
from ragsource.models import Chunk, IngestResponse
from ragsource.store import JsonVectorStore


class PdfIngestionService:
    def __init__(
        self, store: JsonVectorStore, embedder: Embedder, chunk_size: int, chunk_overlap: int
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, filename: str, content: bytes) -> IngestResponse:
        document_id = hashlib.sha256(content).hexdigest()[:16]
        try:
            reader = PdfReader(BytesIO(content))
        except (PdfReadError, EOFError, ValueError) as exc:
            raise ValueError("The uploaded file is not a readable PDF") from exc
        pending: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            for text in chunk_text(page.extract_text() or "", self.chunk_size, self.chunk_overlap):
                pending.append((page_number, text))
        if not pending:
            raise ValueError("The PDF contains no extractable text")

        vectors = self.embedder.encode([text for _, text in pending])
        chunks = [
            Chunk(
                id=f"{document_id}-p{page}-c{index}",
                document_id=document_id,
                filename=filename,
                page=page,
                text=text,
                embedding=vector,
            )
            for index, ((page, text), vector) in enumerate(zip(pending, vectors, strict=True))
        ]
        self.store.replace_document(document_id, chunks)
        return IngestResponse(
            document_id=document_id, filename=filename, pages=len(reader.pages), chunks=len(chunks)
        )
