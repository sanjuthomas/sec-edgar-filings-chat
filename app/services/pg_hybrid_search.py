from __future__ import annotations

from app.models import ChunkMatch
from app.repositories.pgvector_repo import PgBm25ChunkRepository, PgVectorHybridChunkRepository
from app.services import hybrid_reranker


class PgHybridSearchService:
    def __init__(
        self,
        enabled: bool,
        hybrid_retrieval_top_k: int,
        vector_repository: PgVectorHybridChunkRepository,
        bm25_repository: PgBm25ChunkRepository,
    ) -> None:
        self._enabled = enabled
        self._hybrid_retrieval_top_k = hybrid_retrieval_top_k
        self._vector_repository = vector_repository
        self._bm25_repository = bm25_repository

    def is_enabled(self) -> bool:
        return self._enabled

    def search(
        self,
        question: str,
        query_embedding: list[float],
        top_n: int,
        ticker: str | None,
        form: str | None,
    ) -> list[ChunkMatch]:
        vector_chunks = self._vector_repository.find_similar_chunks(
            query_embedding,
            self._hybrid_retrieval_top_k,
            ticker,
            form,
        )
        bm25_chunks = self._bm25_repository.find_keyword_chunks(
            question,
            self._hybrid_retrieval_top_k,
            ticker,
            form,
        )
        return hybrid_reranker.rerank(vector_chunks, bm25_chunks, top_n)
