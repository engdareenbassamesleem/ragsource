from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from ragsource.config import Settings, get_settings
from ragsource.embeddings import FastEmbedder
from ragsource.ingestion import PdfIngestionService
from ragsource.llm import ExtractiveGenerator, GeminiGenerator
from ragsource.models import AskRequest, AskResponse, HealthResponse, IngestResponse
from ragsource.service import RagService
from ragsource.store import JsonVectorStore


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = JsonVectorStore(settings.index_path)
        self.embedder = FastEmbedder(settings.embedding_model)
        self.generator = (
            GeminiGenerator(settings.gemini_api_key, settings.gemini_model)
            if settings.gemini_api_key
            else ExtractiveGenerator()
        )
        self.ingestion = PdfIngestionService(
            self.store, self.embedder, settings.chunk_size, settings.chunk_overlap
        )
        self.rag = RagService(self.store, self.embedder, self.generator, settings.minimum_score)


@lru_cache
def get_container() -> Container:
    return Container(get_settings())


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Citation-first RAG API for PDF knowledge bases.",
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(container: Annotated[Container, Depends(get_container)]) -> HealthResponse:
        return HealthResponse(status="ok", indexed_chunks=container.store.count())

    @app.post(
        "/v1/documents",
        response_model=IngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    async def ingest_document(
        file: Annotated[UploadFile, File()],
        container: Annotated[Container, Depends(get_container)],
    ) -> IngestResponse:
        if file.content_type != "application/pdf" or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Only PDF files are supported")
        content = await file.read(container.settings.max_upload_mb * 1024 * 1024 + 1)
        if len(content) > container.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="PDF exceeds the configured size limit")
        try:
            return container.ingestion.ingest(file.filename, content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/ask", response_model=AskResponse, tags=["questions"])
    def ask(
        request: AskRequest,
        container: Annotated[Container, Depends(get_container)],
    ) -> AskResponse:
        return container.rag.ask(request.question, request.top_k or settings.retrieval_k)

    return app


app = create_app()
