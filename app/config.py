from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    server_port: int = 8095

    database_url: str | None = None
    pg_user: str = Field(
        default="postgres",
        validation_alias=AliasChoices("PGUSER", "PG_USER", "pg_user"),
    )
    pg_password: str = Field(
        default="postgres",
        validation_alias=AliasChoices("PGPASSWORD", "PG_PASSWORD", "pg_password"),
    )
    pg_host: str = Field(default="localhost", validation_alias=AliasChoices("PG_HOST", "pg_host"))
    pg_port: int = Field(default=5433, validation_alias=AliasChoices("PG_PORT", "pg_port"))
    pg_database: str = Field(
        default="edgar",
        validation_alias=AliasChoices("PG_DATABASE", "pg_database"),
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:30b"
    ollama_chat_temperature: float = 0.2
    ollama_chat_num_predict: int = 2048
    ollama_embedding_model: str = "bge-m3"

    search_top_k: int = 25
    embedding_dimensions: int = 1024
    hybrid_retrieval_top_k: int = 50

    pgsearch_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("PGSEARCH_ENABLED", "pgsearch_enabled"),
    )

    default_vector_store: str = "pgvector"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "filing_chunks"
    qdrantsearch_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("QDRANTSEARCH_ENABLED", "qdrantsearch_enabled"),
    )
    qdrant_dense_vector: str = Field(
        default="dense",
        validation_alias=AliasChoices("QDRANT_DENSE_VECTOR", "qdrant_dense_vector"),
    )
    qdrant_bm25_vector: str = Field(
        default="content-bm25",
        validation_alias=AliasChoices("QDRANT_BM25_VECTOR", "qdrant_bm25_vector"),
    )
    qdrant_bm25_model: str = Field(
        default="Qdrant/bm25",
        validation_alias=AliasChoices("QDRANT_BM25_MODEL", "qdrant_bm25_model"),
    )

    session_secret_key: str = "dev-only-change-in-production"
    conversation_max_turns: int = 40

    @model_validator(mode="after")
    def resolve_database_url(self) -> Settings:
        if self.database_url:
            return self
        user = quote_plus(self.pg_user)
        if self.pg_password:
            credentials = f"{user}:{quote_plus(self.pg_password)}@"
        else:
            credentials = f"{user}@"
        object.__setattr__(
            self,
            "database_url",
            f"postgresql://{credentials}{self.pg_host}:{self.pg_port}/{self.pg_database}",
        )
        return self

    @property
    def resolved_database_url(self) -> str:
        assert self.database_url is not None
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
