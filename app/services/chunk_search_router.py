from __future__ import annotations

from app.models import ChunkMatch, RetrievedChunk, VectorStoreType
from app.repositories.pgvector_repo import ChunkSearchRepository


class ChunkSearchRouter:
    def __init__(self, repositories: list[ChunkSearchRepository]) -> None:
        self._repositories = {repo.vector_store_type(): repo for repo in repositories}

    def find_similar_chunks(
        self,
        vector_store: VectorStoreType,
        query_embedding: list[float],
        top_k: int,
        ticker: str | None,
        form: str | None,
    ) -> list[ChunkMatch]:
        repository = self._repositories.get(vector_store)
        if repository is None:
            raise ValueError(f"Unsupported vector store: {vector_store}")
        return repository.find_similar_chunks(query_embedding, top_k, ticker, form)
