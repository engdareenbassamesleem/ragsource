from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAGSource"
    environment: str = "development"
    data_dir: Path = Path("data")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5
    minimum_score: float = 0.20
    max_upload_mb: int = 20

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
