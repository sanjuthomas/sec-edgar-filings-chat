from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class VectorStoreType(str, Enum):
    PGVECTOR = "pgvector"
    QDRANT = "qdrant"

    @classmethod
    def from_value(cls, raw: str | None) -> VectorStoreType:
        if raw is None or not raw.strip():
            raise ValueError("Vector store is required.")
        normalized = raw.strip().lower()
        for store in cls:
            if store.value == normalized:
                return store
        raise ValueError(f"Unknown vector store: {raw}")


class SearchForm(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_model: str = Field(min_length=1, max_length=100)
    vector_store: str = Field(min_length=1, max_length=20)
    chunk_count: int = Field(ge=1, le=500)
    ticker: str = Field(default="", max_length=10)
    form: str = Field(default="", max_length=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Please enter a question.")
        return value

    @field_validator("chat_model")
    @classmethod
    def chat_model_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Please select an Ollama model.")
        return value

    @field_validator("vector_store")
    @classmethod
    def vector_store_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Please select a vector store.")
        return value

    def normalized_ticker(self) -> str | None:
        if not self.ticker or not self.ticker.strip():
            return None
        return self.ticker.strip().upper()

    def normalized_form(self) -> str | None:
        if not self.form or not self.form.strip():
            return None
        return self.form.strip().upper()

    def vector_store_type(self) -> VectorStoreType:
        return VectorStoreType.from_value(self.vector_store)


class ChunkMatch(BaseModel):
    citation_number: int
    content: str
    distance: float
    accession_number: str
    chunk_index: int
    ticker: str
    company_name: str
    form: str
    filing_date: date | None
    document_url: str | None
    section: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: int
    content: str
    accession_number: str
    chunk_index: int
    ticker: str
    company_name: str
    form: str
    filing_date: date | None
    document_url: str | None
    section: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChunkMatch]
    chat_model: str
    vector_store: str
    applied_ticker: str | None
    ticker_inferred: bool
    retrieval_ms: int
    generation_ms: int


class CompanyRecord(BaseModel):
    ticker: str
    company_name: str


class ResolvedTicker(BaseModel):
    ticker: str | None
    inferred: bool


class ConversationSettings(BaseModel):
    chat_model: str = Field(min_length=1, max_length=100)
    vector_store: str = Field(min_length=1, max_length=20)
    chunk_count: int = Field(ge=1, le=500)
    ticker: str = Field(default="", max_length=10)
    form: str = Field(default="", max_length=20)

    def normalized_ticker(self) -> str | None:
        if not self.ticker or not self.ticker.strip():
            return None
        return self.ticker.strip().upper()

    def normalized_form(self) -> str | None:
        if not self.form or not self.form.strip():
            return None
        return self.form.strip().upper()

    def vector_store_type(self) -> VectorStoreType:
        return VectorStoreType.from_value(self.vector_store)


class TurnMetadata(BaseModel):
    retrieval_ms: int = 0
    generation_ms: int = 0
    vector_store: str = ""
    chat_model: str = ""
    applied_ticker: str | None = None
    ticker_inferred: bool = False
    source_count: int = 0


class ConversationTurn(BaseModel):
    role: str
    content: str
    sources: list[ChunkMatch] = Field(default_factory=list)
    metadata: TurnMetadata | None = None


class Conversation(BaseModel):
    id: str
    settings: ConversationSettings
    turns: list[ConversationTurn]
    created_at: datetime


class ChatForm(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    chat_model: str = Field(min_length=1, max_length=100)
    vector_store: str = Field(min_length=1, max_length=20)
    chunk_count: int = Field(ge=1, le=500)
    ticker: str = Field(default="", max_length=10)
    form: str = Field(default="", max_length=20)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Please enter a message.")
        return value.strip()

    def to_settings(self) -> ConversationSettings:
        return ConversationSettings(
            chat_model=self.chat_model,
            vector_store=self.vector_store,
            chunk_count=self.chunk_count,
            ticker=self.ticker,
            form=self.form,
        )
