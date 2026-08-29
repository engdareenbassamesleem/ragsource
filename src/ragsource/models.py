from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    document_id: str
    filename: str
    page: int = Field(ge=1)
    text: str
    embedding: list[float]


class Source(BaseModel):
    citation: str
    document_id: str
    filename: str
    page: int
    score: float
    excerpt: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
