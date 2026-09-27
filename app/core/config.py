from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ContractLens API"
    environment: str = "development"
    cors_origins: str = "http://localhost:3000"
    max_upload_size_mb: int = Field(default=20, ge=1, le=100)
    upload_dir: Path = Path("data/uploads")
    log_level: str = "INFO"
    database_url: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = Field(default=384, ge=1, le=4096)
    chunk_size_chars: int = Field(default=1200, ge=200, le=10000)
    chunk_overlap_chars: int = Field(default=180, ge=0, le=2000)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    analysis_top_k: int = Field(default=3, ge=1, le=10)
    rag_min_similarity: float = Field(default=0.35, ge=-1.0, le=1.0)
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_seconds: float = Field(default=45.0, gt=0, le=180)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def database_dimensions(self) -> int:
        return self.embedding_dimensions

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip().rstrip("/") for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap_chars * 2 >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be less than half of CHUNK_SIZE_CHARS")
        if self.environment.lower() == "production":
            if not self.database_url:
                raise ValueError("DATABASE_URL is required in production")
            if not self.llm_api_key or not self.llm_api_key.get_secret_value().strip():
                raise ValueError("LLM_API_KEY is required in production")
            origins = self.allowed_origins
            if not origins or "*" in origins:
                raise ValueError("CORS_ORIGINS must contain explicit frontend origins in production")
            if any(not origin.startswith("https://") for origin in origins):
                raise ValueError("Production CORS origins must use HTTPS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
