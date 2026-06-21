from __future__ import annotations

from datetime import date

import httpx

from app.exceptions import SearchError
from app.models import ChunkMatch, VectorStoreType
from app.services.qdrant_info import get_collection_vector_size

RESERVED_PAYLOAD_KEYS = frozenset(
    {
        "content",
        "accession_number",
        "chunk_index",
        "ticker",
        "company_name",
        "form",
        "filing_date",
        "document_url",
        "section",
    }
)


class QdrantChunkRepository:
    def __init__(self, base_url: str, collection: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._collection = collection
        self._vector_size: int | None = None

    def vector_store_type(self) -> VectorStoreType:
        return VectorStoreType.QDRANT

    def collection_vector_size(self) -> int | None:
        if self._vector_size is None:
            self._vector_size = get_collection_vector_size(self._base_url, self._collection)
        return self._vector_size

    def find_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        ticker: str | None,
        form: str | None,
    ) -> list[ChunkMatch]:
        expected_size = self.collection_vector_size()
        if expected_size is not None and len(query_embedding) != expected_size:
            raise SearchError(
                f"Embedding dimension mismatch: query vector has {len(query_embedding)} dimensions "
                f"but Qdrant collection '{self._collection}' expects {expected_size}. "
                "Re-index with bge-m3 (1024-dim) or set OLLAMA_EMBEDDING_MODEL to match your collection."
            )

        request_body: dict = {
            "query": query_embedding,
            "limit": top_k,
            "with_payload": True,
        }
        filter_body = _build_filter(ticker, form)
        if filter_body is not None:
            request_body["filter"] = filter_body

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

        matches: list[ChunkMatch] = []
        for index, point in enumerate(points, start=1):
            point_payload = point.get("payload") or {}
            score = point.get("score")
            matches.append(_to_chunk_match(index, score, point_payload))
        return matches


def _build_filter(ticker: str | None, form: str | None) -> dict | None:
    must: list[dict] = []
    if ticker is not None:
        must.append(_keyword_match("ticker", ticker))
    if form is not None:
        must.append(_keyword_match("form", form))
    if not must:
        return None
    return {"must": must}


def _keyword_match(key: str, value: str) -> dict:
    return {"key": key, "match": {"value": value}}


def _to_chunk_match(citation_number: int, score: float | None, payload: dict) -> ChunkMatch:
    distance = 1.0 - score if score is not None else 1.0
    section = _string_value(payload.get("section")) or None
    filing_date = _parse_filing_date(payload.get("filing_date"))
    return ChunkMatch(
        citation_number=citation_number,
        content=_string_value(payload.get("content")),
        distance=distance,
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


def _extract_metadata(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key not in RESERVED_PAYLOAD_KEYS}


def _parse_filing_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return date.fromisoformat(text[: min(10, len(text))])


def _string_value(value: object) -> str:
    return "" if value is None else str(value)


def _int_value(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if value is None:
        return 0
    return int(str(value))


def _extract_error_detail(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
        error = payload.get("status", {}).get("error")
        if isinstance(error, str) and error:
            return error
    except Exception:
        pass
    return exc.response.text[:300] or "Unknown Qdrant error"
