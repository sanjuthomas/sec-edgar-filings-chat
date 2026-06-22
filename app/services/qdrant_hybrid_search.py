from __future__ import annotations

from app.models import ChunkMatch
from app.repositories.qdrant_hybrid_repo import QdrantHybridChunkRepository
from app.services import hybrid_reranker


class QdrantHybridSearchService:
    def __init__(
        self,
        enabled: bool,
        hybrid_retrieval_top_k: int,
        repository: QdrantHybridChunkRepository,
    ) -> None:
        self._enabled = enabled
        self._hybrid_retrieval_top_k = hybrid_retrieval_top_k
        self._repository = repository

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
        vector_chunks = self._repository.find_dense_chunks(
            query_embedding,
            self._hybrid_retrieval_top_k,
            ticker,
            form,
        )
        bm25_chunks = self._repository.find_bm25_chunks(
            question,
            self._hybrid_retrieval_top_k,
            ticker,
            form,
        )
        return hybrid_reranker.rerank(vector_chunks, bm25_chunks, top_n)
