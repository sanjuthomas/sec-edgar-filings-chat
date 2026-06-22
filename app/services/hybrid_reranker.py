from __future__ import annotations

from app.models import ChunkMatch, RetrievedChunk


RRF_K = 60


def rerank(
    vector_chunks: list[RetrievedChunk],
    bm25_chunks: list[RetrievedChunk],
    top_n: int,
) -> list[ChunkMatch]:
    chunks_by_key: dict[str, RetrievedChunk] = {}
    fused_scores: dict[str, float] = {}

    _accumulate_rrf_scores(vector_chunks, fused_scores, chunks_by_key)
    _accumulate_rrf_scores(bm25_chunks, fused_scores, chunks_by_key)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

    matches: list[ChunkMatch] = []
    for citation_number, (merge_key, fused_score) in enumerate(ranked, start=1):
        chunk = chunks_by_key[merge_key]
        matches.append(_to_chunk_match(citation_number, chunk, fused_score))
    return matches


def _accumulate_rrf_scores(
    ranked_chunks: list[RetrievedChunk],
    fused_scores: dict[str, float],
    chunks_by_key: dict[str, RetrievedChunk],
) -> None:
    for rank, chunk in enumerate(ranked_chunks):
        chunks_by_key.setdefault(chunk.merge_key, chunk)
        fused_scores[chunk.merge_key] = fused_scores.get(chunk.merge_key, 0.0) + (
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
