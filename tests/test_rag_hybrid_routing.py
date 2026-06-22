from __future__ import annotations

from unittest.mock import MagicMock

from app.models import ChunkMatch, ConversationSettings, TurnMetadata, VectorStoreType
from app.services.rag_search import RagSearchService


class TestRagSearchHybridRouting:
    def _settings(self, vector_store: str) -> ConversationSettings:
        return ConversationSettings(
            chat_model="qwen3:14b",
            vector_store=vector_store,
            chunk_count=10,
            ticker="GS",
            form="10-K",
        )

    def _chunk_match(self) -> ChunkMatch:
        return ChunkMatch(
            citation_number=1,
            content="excerpt",
            distance=0.5,
            accession_number="0000000000-00-000001",
            chunk_index=0,
            ticker="GS",
            company_name="Example Corp",
            form="10-K",
            filing_date=None,
            document_url=None,
            section=None,
            metadata={},
        )

    def test_uses_pg_hybrid_when_pgvector_selected(self) -> None:
        pg_hybrid = MagicMock()
        pg_hybrid.is_enabled.return_value = True
        pg_hybrid.search.return_value = [self._chunk_match()]

        qdrant_hybrid = MagicMock()
        qdrant_hybrid.is_enabled.return_value = True

        router = MagicMock()
        ollama = MagicMock()
        ollama.embed.return_value = [0.1]
        ollama.chat_messages.return_value = "answer"

        ticker_resolver = MagicMock()
        ticker_resolver.resolve.return_value = MagicMock(ticker="GS", inferred=False)

        service = RagSearchService(
            ollama,
            router,
            pg_hybrid,
            qdrant_hybrid,
            ticker_resolver,
        )

        turn = service.answer_message("question", self._settings("pgvector"), [])
        assert turn.content == "answer"
        pg_hybrid.search.assert_called_once()
        qdrant_hybrid.search.assert_not_called()
        router.find_similar_chunks.assert_not_called()

    def test_uses_qdrant_hybrid_when_qdrant_selected(self) -> None:
        pg_hybrid = MagicMock()
        pg_hybrid.is_enabled.return_value = True

        qdrant_hybrid = MagicMock()
        qdrant_hybrid.is_enabled.return_value = True
        qdrant_hybrid.search.return_value = [self._chunk_match()]

        router = MagicMock()
        ollama = MagicMock()
        ollama.embed.return_value = [0.1]
        ollama.chat_messages.return_value = "answer"

        ticker_resolver = MagicMock()
        ticker_resolver.resolve.return_value = MagicMock(ticker="GS", inferred=False)

        service = RagSearchService(
            ollama,
            router,
            pg_hybrid,
            qdrant_hybrid,
            ticker_resolver,
        )

        turn = service.answer_message("question", self._settings("qdrant"), [])
        assert turn.content == "answer"
        qdrant_hybrid.search.assert_called_once()
        pg_hybrid.search.assert_not_called()
        router.find_similar_chunks.assert_not_called()

    def test_falls_back_to_vector_only_when_hybrid_disabled(self) -> None:
        pg_hybrid = MagicMock()
        pg_hybrid.is_enabled.return_value = False
        qdrant_hybrid = MagicMock()
        qdrant_hybrid.is_enabled.return_value = False

        router = MagicMock()
        router.find_similar_chunks.return_value = [self._chunk_match()]

        ollama = MagicMock()
        ollama.embed.return_value = [0.1]
        ollama.chat_messages.return_value = "answer"

        ticker_resolver = MagicMock()
        ticker_resolver.resolve.return_value = MagicMock(ticker=None, inferred=False)

        service = RagSearchService(
            ollama,
            router,
            pg_hybrid,
            qdrant_hybrid,
            ticker_resolver,
        )

        turn = service.answer_message("question", self._settings("qdrant"), [])
        assert turn.content == "answer"
        router.find_similar_chunks.assert_called_once_with(
            VectorStoreType.QDRANT,
            [0.1],
            10,
            None,
            "10-K",
        )
