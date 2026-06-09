from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Loaded once from the environment / .env file.

    Every secret, URL, and tunable lives here. Nothing else in the app reads
    os.environ directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agentic-rag"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str
    redis_url: str
    qdrant_url: str
    qdrant_api_key: str | None = None

    # Embeddings: one model for the whole system (ingestion AND query must match).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # Chunking: deterministic fixed-size strategy (characters).
    chunk_size: int = 1000
    chunk_overlap: int = 200


@lru_cache
def get_settings() -> Settings:
    # Required fields are populated from the environment by pydantic-settings.
    return Settings()
