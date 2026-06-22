from __future__ import annotations

import httpx

from app.exceptions import SearchError
from app.models import RetrievedChunk
from app.repositories.qdrant_repo import (
    _build_filter,
    _extract_error_detail,
    _extract_metadata,
    _int_value,
    _parse_filing_date,
    _string_value,
)
from app.services.qdrant_info import get_collection_vector_size


class QdrantHybridChunkRepository:
    def __init__(
        self,
        base_url: str,
        collection: str,
        dense_vector: str,
        bm25_vector: str,
        bm25_model: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._collection = collection
        self._dense_vector = dense_vector
        self._bm25_vector = bm25_vector
        self._bm25_model = bm25_model
        self._vector_size: int | None = None

    def collection_vector_size(self) -> int | None:
        if self._vector_size is None:
            self._vector_size = get_collection_vector_size(self._base_url, self._collection)
        return self._vector_size

    def find_dense_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        ticker: str | None,
        form: str | None,
    ) -> list[RetrievedChunk]:
        expected_size = self.collection_vector_size()
        if expected_size is not None and len(query_embedding) != expected_size:
            raise SearchError(
                f"Embedding dimension mismatch: query vector has {len(query_embedding)} dimensions "
                f"but Qdrant collection '{self._collection}' expects {expected_size}. "
                "Re-index with bge-m3 (1024-dim) or set OLLAMA_EMBEDDING_MODEL to match your collection."
            )

        request_body: dict = {
            "query": query_embedding,
            "using": self._dense_vector,
            "limit": top_k,
            "with_payload": True,
        }
        filter_body = _build_filter(ticker, form)
        if filter_body is not None:
            request_body["filter"] = filter_body

        return self._query_points(request_body)

    def find_bm25_chunks(
        self,
        query: str,
        top_k: int,
        ticker: str | None,
        form: str | None,
    ) -> list[RetrievedChunk]:
        request_body: dict = {
            "query": {
                "text": query,
                "model": self._bm25_model,
            },
            "using": self._bm25_vector,
            "limit": top_k,
            "with_payload": True,
        }
        filter_body = _build_filter(ticker, form)
        if filter_body is not None:
            request_body["filter"] = filter_body

        return self._query_points(request_body)

    def _query_points(self, request_body: dict) -> list[RetrievedChunk]:
        try:
            with httpx.Client(base_url=self._base_url, timeout=60.0) as client:
                response = client.post(
                    f"/collections/{self._collection}/points/query",
                    json=request_body,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = _extract_error_detail(exc)
            raise SearchError(
                f"Qdrant search failed ({exc.response.status_code}): {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise SearchError(
                f"Could not reach Qdrant at {self._base_url}. Is it running?"
            ) from exc

        points = (
            payload.get("result", {}).get("points")
            if isinstance(payload.get("result"), dict)
            else None
        )
        if not points:
            return []

        return [_to_retrieved_chunk(point) for point in points]


def _to_retrieved_chunk(point: dict) -> RetrievedChunk:
    payload = point.get("payload") or {}
    section = _string_value(payload.get("section")) or None
    if section == "":
        section = None
    filing_date = _parse_filing_date(payload.get("filing_date"))
    return RetrievedChunk(
        merge_key=_string_value(point.get("id")),
        content=_string_value(payload.get("content")),
        accession_number=_string_value(payload.get("accession_number")),
        chunk_index=_int_value(payload.get("chunk_index")),
        ticker=_string_value(payload.get("ticker")),
        company_name=_string_value(payload.get("company_name")),
        form=_string_value(payload.get("form")),
        filing_date=filing_date,
        document_url=_string_value(payload.get("document_url")) or None,
        section=section,
        metadata=_extract_metadata(payload),
    )
