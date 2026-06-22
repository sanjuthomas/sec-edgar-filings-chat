from __future__ import annotations

import json
from datetime import date
from typing import Protocol

import psycopg
from pgvector.psycopg import register_vector

from app.models import ChunkMatch, RetrievedChunk, VectorStoreType


def _parse_metadata(raw: str | dict | None) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not str(raw).strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class ChunkSearchRepository(Protocol):
    def vector_store_type(self) -> VectorStoreType: ...

    def find_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        ticker: str | None,
        form: str | None,
    ) -> list[ChunkMatch]: ...


class PgVectorChunkRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._database_url)
        register_vector(conn)
        return conn

    def vector_store_type(self) -> VectorStoreType:
        return VectorStoreType.PGVECTOR

    def find_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        ticker: str | None,
        form: str | None,
    ) -> list[ChunkMatch]:
        sql = """
            SELECT
                c.content,
                c.embedding <=> %s::vector AS distance,
                c.accession_number,
                c.chunk_index,
                c.metadata,
                f.ticker,
                f.company_name,
                f.form,
                f.filing_date,
                f.document_url
            FROM filing_chunks c
            JOIN filings f ON f.accession_number = c.accession_number
            WHERE TRUE
        """
        params: list[object] = [query_embedding]

        if ticker is not None:
            sql += " AND f.ticker = %s"
            params.append(ticker)
        if form is not None:
            sql += " AND f.form = %s"
            params.append(form)

        sql += " ORDER BY distance LIMIT %s"
        params.append(top_k)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        matches: list[ChunkMatch] = []
        for row_num, row in enumerate(rows, start=1):
            metadata = _parse_metadata(row[4])
            section = metadata.get("section") if isinstance(metadata.get("section"), str) else None
            filing_date: date | None = row[8]
            matches.append(
                ChunkMatch(
                    citation_number=row_num,
                    content=row[0],
                    distance=float(row[1]),
                    accession_number=row[2],
                    chunk_index=row[3],
                    ticker=row[5],
                    company_name=row[6],
                    form=row[7],
                    filing_date=filing_date,
                    document_url=row[9],
                    section=section,
                    metadata=metadata,
                )
            )
        return matches


class PgVectorHybridChunkRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._database_url)
        register_vector(conn)
        return conn

    def find_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        ticker: str | None,
        form: str | None,
    ) -> list[RetrievedChunk]:
        sql = """
            SELECT
                c.id,
                c.content,
                c.accession_number,
                c.chunk_index,
                c.metadata,
                f.ticker,
                f.company_name,
                f.form,
                f.filing_date,
                f.document_url
            FROM filing_chunks c
            JOIN filings f ON f.accession_number = c.accession_number
            WHERE TRUE
        """
        params: list[object] = []

        if ticker is not None:
            sql += " AND f.ticker = %s"
            params.append(ticker)
        if form is not None:
            sql += " AND f.form = %s"
            params.append(form)

        sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, top_k])

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [_map_retrieved_row(row) for row in rows]


class PgBm25ChunkRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._database_url)
        register_vector(conn)
        return conn

    def find_keyword_chunks(
        self,
        query: str,
        top_k: int,
        ticker: str | None,
        form: str | None,
    ) -> list[RetrievedChunk]:
        sql = """
            SELECT
                c.id,
                c.content,
                c.accession_number,
                c.chunk_index,
                c.metadata,
                f.ticker,
                f.company_name,
                f.form,
                f.filing_date,
                f.document_url,
                pdb.score(c.id) AS rank
            FROM filing_chunks c
            JOIN filings f ON f.accession_number = c.accession_number
            WHERE c.content ||| %s
        """
        params: list[object] = [query]

        if ticker is not None:
            sql += " AND f.ticker = %s"
            params.append(ticker)
        if form is not None:
            sql += " AND f.form = %s"
            params.append(form)

        sql += " ORDER BY rank DESC LIMIT %s"
        params.append(top_k)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [_map_retrieved_row(row[:10]) for row in rows]


def _map_retrieved_row(row: tuple) -> RetrievedChunk:
    metadata = _parse_metadata(row[4])
    section = metadata.get("section") if isinstance(metadata.get("section"), str) else None
    filing_date: date | None = row[8]
    return RetrievedChunk(
        merge_key=str(row[0]),
        content=row[1],
        accession_number=row[2],
        chunk_index=row[3],
        ticker=row[5],
        company_name=row[6],
        form=row[7],
        filing_date=filing_date,
        document_url=row[9],
        section=section,
        metadata=metadata,
    )
