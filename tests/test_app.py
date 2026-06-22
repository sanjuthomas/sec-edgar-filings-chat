from __future__ import annotations

import pytest

from app.models import ChunkMatch, SearchForm, VectorStoreType
from app.services import hybrid_reranker
from app.services.chunk_search_router import ChunkSearchRouter
from app.services.ticker_resolver import TickerResolver
from app.models import CompanyRecord, RetrievedChunk


class TestSearchForm:
    def test_normalizes_ticker_and_form(self) -> None:
        form = SearchForm(
            question="question",
            chat_model="qwen3:30b",
            vector_store="qdrant",
            chunk_count=25,
            ticker=" gs ",
            form=" 10-k ",
        )

        assert form.normalized_ticker() == "GS"
        assert form.normalized_form() == "10-K"
        assert form.vector_store_type() == VectorStoreType.QDRANT

    def test_treats_blank_filters_as_null(self) -> None:
        form = SearchForm(
            question="question",
            chat_model="qwen3:14b",
            vector_store="pgvector",
            chunk_count=10,
            ticker="  ",
            form="",
        )

        assert form.normalized_ticker() is None
        assert form.normalized_form() is None
        assert form.vector_store_type() == VectorStoreType.PGVECTOR


class TestVectorStoreType:
    def test_parses_known_values(self) -> None:
        assert VectorStoreType.from_value("pgvector") == VectorStoreType.PGVECTOR
        assert VectorStoreType.from_value("QDRANT") == VectorStoreType.QDRANT

    def test_rejects_blank_value(self) -> None:
        with pytest.raises(ValueError, match="required"):
            VectorStoreType.from_value("  ")

    def test_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError):
            VectorStoreType.from_value("pinecone")


class TestTickerResolver:
    @pytest.fixture
    def resolver(self) -> TickerResolver:
        return TickerResolver.for_testing(
            ["A", "ADBE", "GS", "IT", "SO"],
            [
                CompanyRecord(ticker="GS", company_name="GOLDMAN SACHS GROUP INC"),
                CompanyRecord(ticker="ADBE", company_name="ADOBE INC."),
            ],
        )

    def test_prefers_explicit_ticker(self, resolver: TickerResolver) -> None:
        resolved = resolver.resolve("buyback question", "ADBE")
        assert resolved.ticker == "ADBE"
        assert resolved.inferred is False

    def test_infers_ticker_from_question(self, resolver: TickerResolver) -> None:
        resolved = resolver.resolve("Was there a share buyback program announced by ADBE?", None)
        assert resolved.ticker == "ADBE"
        assert resolved.inferred is True

    def test_infers_ticker_from_company_name(self, resolver: TickerResolver) -> None:
        resolved = resolver.resolve("Who are the elected directors in Goldman Sachs?", None)
        assert resolved.ticker == "GS"
        assert resolved.inferred is True

    def test_infers_ticker_from_company_name_adobe(self, resolver: TickerResolver) -> None:
        resolved = resolver.resolve(
            "Do you know if the Adobe board approved a buyback program? If so, how much was it for?",
            None,
        )
        assert resolved.ticker == "ADBE"
        assert resolved.inferred is True

    def test_does_not_treat_common_words_as_tickers(self, resolver: TickerResolver) -> None:
        resolved = resolver.resolve("How much was it for if so?", None)
        assert resolved.ticker is None

    def test_ignores_ambiguous_single_letter_tickers(self, resolver: TickerResolver) -> None:
        resolved = resolver.resolve("Was there a share buyback program announced?", None)
        assert resolved.ticker is None


class TestHybridReranker:
    @staticmethod
    def _chunk(merge_key: str, content: str) -> RetrievedChunk:
        return RetrievedChunk(
            merge_key=merge_key,
            content=content,
            accession_number=f"0000000000-00-{merge_key}",
            chunk_index=0,
            ticker="GS",
            company_name="Example Corp",
            form="10-K",
            filing_date=None,
            document_url=None,
            section=None,
            metadata={},
        )

    def test_merges_results_by_chunk_id_and_reranks_with_reciprocal_rank_fusion(self) -> None:
        shared = self._chunk("1", "shared chunk")
        vector_only = self._chunk("2", "vector only")
        bm25_only = self._chunk("3", "bm25 only")

        reranked = hybrid_reranker.rerank(
            [shared, vector_only],
            [shared, bm25_only],
            3,
        )

        assert len(reranked) == 3
        assert reranked[0].accession_number == "0000000000-00-1"
        assert reranked[0].chunk_index == 0
        assert reranked[0].citation_number == 1
        assert reranked[0].distance > 0.0

    def test_limits_results_to_requested_top_n(self) -> None:
        vector_chunks = [
            self._chunk("1", "one"),
            self._chunk("2", "two"),
            self._chunk("3", "three"),
        ]

        reranked = hybrid_reranker.rerank(vector_chunks, [], 2)

        assert len(reranked) == 2
        assert reranked[0].citation_number == 1
        assert reranked[1].citation_number == 2


class TestChunkSearchRouter:
    @staticmethod
    def _chunk_match(ticker: str) -> ChunkMatch:
        return ChunkMatch(
            citation_number=1,
            content="excerpt",
            distance=0.12,
            accession_number="0000000000-00-000000",
            chunk_index=0,
            ticker=ticker,
            company_name="Example Corp",
            form="10-K",
            filing_date=None,
            document_url=None,
            section=None,
            metadata={},
        )

    def test_routes_to_selected_vector_store(self) -> None:
        pg_match = self._chunk_match("PG")
        qdrant_match = self._chunk_match("QD")

        class StubRepository:
            def __init__(self, store_type: VectorStoreType, match: ChunkMatch) -> None:
                self._store_type = store_type
                self._match = match

            def vector_store_type(self) -> VectorStoreType:
                return self._store_type

            def find_similar_chunks(
                self,
                query_embedding: list[float],
                top_k: int,
                ticker: str | None,
                form: str | None,
            ) -> list[ChunkMatch]:
                return [self._match]

        router = ChunkSearchRouter(
            [
                StubRepository(VectorStoreType.PGVECTOR, pg_match),
                StubRepository(VectorStoreType.QDRANT, qdrant_match),
            ]
        )

        from_pg = router.find_similar_chunks(
            VectorStoreType.PGVECTOR,
            [0.1],
            10,
            "GS",
            "10-K",
        )
        from_qdrant = router.find_similar_chunks(
            VectorStoreType.QDRANT,
            [0.2],
            25,
            None,
            None,
        )

        assert from_pg == [pg_match]
        assert from_qdrant == [qdrant_match]

    def test_rejects_unsupported_vector_store(self) -> None:
        class StubRepository:
            def vector_store_type(self) -> VectorStoreType:
                return VectorStoreType.PGVECTOR

            def find_similar_chunks(
                self,
                query_embedding: list[float],
                top_k: int,
                ticker: str | None,
                form: str | None,
            ) -> list[ChunkMatch]:
                return [self._chunk_match("PG")]

        router = ChunkSearchRouter([StubRepository()])

        with pytest.raises(ValueError, match="Unsupported vector store"):
            router.find_similar_chunks(VectorStoreType.QDRANT, [0.1], 10, None, None)
