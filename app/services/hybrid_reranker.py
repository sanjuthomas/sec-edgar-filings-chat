from __future__ import annotations

from app.models import ChunkMatch, RetrievedChunk
from app.repositories.pgvector_repo import PgBm25ChunkRepository, PgVectorHybridChunkRepository


RRF_K = 60


def rerank(
    vector_chunks: list[RetrievedChunk],
    bm25_chunks: list[RetrievedChunk],
    top_n: int,
) -> list[ChunkMatch]:
    chunks_by_id: dict[int, RetrievedChunk] = {}
    fused_scores: dict[int, float] = {}

    _accumulate_rrf_scores(vector_chunks, fused_scores, chunks_by_id)
    _accumulate_rrf_scores(bm25_chunks, fused_scores, chunks_by_id)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

    matches: list[ChunkMatch] = []
    for citation_number, (chunk_id, fused_score) in enumerate(ranked, start=1):
        chunk = chunks_by_id[chunk_id]
        matches.append(_to_chunk_match(citation_number, chunk, fused_score))
    return matches


def _accumulate_rrf_scores(
    ranked_chunks: list[RetrievedChunk],
    fused_scores: dict[int, float],
    chunks_by_id: dict[int, RetrievedChunk],
) -> None:
    for rank, chunk in enumerate(ranked_chunks):
        chunks_by_id.setdefault(chunk.chunk_id, chunk)
        fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
            1.0 / (RRF_K + rank + 1)
        )


def _to_chunk_match(
    citation_number: int,
    chunk: RetrievedChunk,
    fused_score: float,
) -> ChunkMatch:
    return ChunkMatch(
        citation_number=citation_number,
        content=chunk.content,
        distance=fused_score,
        accession_number=chunk.accession_number,
        chunk_index=chunk.chunk_index,
        ticker=chunk.ticker,
        company_name=chunk.company_name,
        form=chunk.form,
        filing_date=chunk.filing_date,
        document_url=chunk.document_url,
        section=chunk.section,
        metadata=chunk.metadata,
    )
