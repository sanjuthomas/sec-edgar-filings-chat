from __future__ import annotations

import json
from datetime import date

import psycopg
from pgvector.psycopg import register_vector

from app.models import CompanyRecord


class FilingMetadataRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._database_url)
        register_vector(conn)
        return conn

    def find_distinct_tickers(self) -> list[str]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT ticker FROM filings GROUP BY ticker ORDER BY LENGTH(ticker) DESC, ticker"
            )
            return [row[0] for row in cur.fetchall()]

    def find_distinct_companies(self) -> list[CompanyRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ticker, company_name
                FROM filings
                WHERE company_name IS NOT NULL AND company_name <> ''
                ORDER BY ticker
                """
            )
            return [
                CompanyRecord(ticker=row[0], company_name=row[1])
                for row in cur.fetchall()
            ]
