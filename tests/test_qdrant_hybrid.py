from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.exceptions import SearchError
from app.models import RetrievedChunk
from app.repositories.qdrant_hybrid_repo import QdrantHybridChunkRepository
from app.services.qdrant_hybrid_search import QdrantHybridSearchService


class TestQdrantHybridChunkRepository:
    @pytest.fixture
    def repository(self) -> QdrantHybridChunkRepository:
        return QdrantHybridChunkRepository(
            "http://qdrant.test",
            "filing_chunks",
            "dense",
            "content-bm25",
            "Qdrant/bm25",
        )

    def test_find_dense_chunks_posts_query_with_dense_vector(self, repository: QdrantHybridChunkRepository) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "points": [
                    {
                        "id": "point-1",
                        "score": 0.91,
                        "payload": {
                            "content": "dense excerpt",
                            "accession_number": "0000000000-00-000001",
                            "chunk_index": 0,
                            "ticker": "GS",
                            "company_name": "Example Corp",
                            "form": "10-K",
                            "filing_date": "2025-01-01",
                            "document_url": "https://example.com/filing",
                        },
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch.object(repository, "collection_vector_size", return_value=None), patch(
            "app.repositories.qdrant_hybrid_repo.httpx.Client"
        ) as client_cls:
            client_cls.return_value.__enter__.return_value = mock_client
            chunks = repository.find_dense_chunks([0.1, 0.2], 5, "GS", "10-K")

        assert len(chunks) == 1
        assert chunks[0].merge_key == "point-1"
        assert chunks[0].ticker == "GS"
        assert chunks[0].content == "dense excerpt"

        request = mock_client.post.call_args
        assert request.args[0] == "/collections/filing_chunks/points/query"
        body = request.kwargs["json"]
        assert body["using"] == "dense"
        assert body["limit"] == 5
        assert body["filter"]["must"][0]["match"]["value"] == "GS"

    def test_find_bm25_chunks_posts_document_query_with_sparse_vector(
        self, repository: QdrantHybridChunkRepository
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "points": [
                    {
                        "id": "point-2",
                        "score": 4.2,
                        "payload": {
                            "content": "bm25 excerpt",
                            "accession_number": "0000000000-00-000002",
                            "chunk_index": 2,
                            "ticker": "AAPL",
                            "company_name": "Apple Inc",
                            "form": "10-Q",
                        },
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("app.repositories.qdrant_hybrid_repo.httpx.Client") as client_cls:
            client_cls.return_value.__enter__.return_value = mock_client
            chunks = repository.find_bm25_chunks("revenue growth", 3, None, None)

        assert len(chunks) == 1
        assert chunks[0].merge_key == "point-2"
        assert chunks[0].content == "bm25 excerpt"

        body = mock_client.post.call_args.kwargs["json"]
        assert body["using"] == "content-bm25"
        assert body["query"] == {"text": "revenue growth", "model": "Qdrant/bm25"}

    def test_returns_empty_list_when_qdrant_response_has_no_points(
        self, repository: QdrantHybridChunkRepository
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"points": []}, "status": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("app.repositories.qdrant_hybrid_repo.httpx.Client") as client_cls:
            client_cls.return_value.__enter__.return_value = mock_client
            chunks = repository.find_bm25_chunks("missing", 3, None, None)

        assert chunks == []

    def test_maps_custom_metadata_and_blank_section(self, repository: QdrantHybridChunkRepository) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "points": [
                    {
                        "id": "point-3",
                        "score": 0.5,
                        "payload": {
                            "content": "metadata excerpt",
                            "accession_number": "0000000000-00-000003",
                            "chunk_index": 1,
                            "ticker": "MSFT",
                            "company_name": "Microsoft Corp",
                            "form": "10-K",
                            "filing_date": "2025-02-01",
                            "section": "",
                            "custom_field": "value",
                        },
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch.object(repository, "collection_vector_size", return_value=None), patch(
            "app.repositories.qdrant_hybrid_repo.httpx.Client"
        ) as client_cls:
            client_cls.return_value.__enter__.return_value = mock_client
            chunks = repository.find_dense_chunks([0.1], 1, None, None)

        assert chunks[0].section is None
        assert chunks[0].metadata == {"custom_field": "value"}

    def test_raises_search_error_on_dimension_mismatch(self, repository: QdrantHybridChunkRepository) -> None:
        with patch.object(repository, "collection_vector_size", return_value=1024):
            with pytest.raises(SearchError, match="Embedding dimension mismatch"):
                repository.find_dense_chunks([0.1, 0.2], 5, None, None)


class TestQdrantHybridSearchService:
    def test_reranks_dense_and_bm25_legs(self) -> None:
        repository = MagicMock()
        repository.find_dense_chunks.return_value = [
            RetrievedChunk(
                merge_key="shared",
                content="shared",
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
        ]
        repository.find_bm25_chunks.return_value = [
            RetrievedChunk(
                merge_key="shared",
                content="shared",
                accession_number="0000000000-00-000001",
                chunk_index=0,
                ticker="GS",
                company_name="Example Corp",
                form="10-K",
                filing_date=None,
                document_url=None,
                section=None,
                metadata={},
            ),
            RetrievedChunk(
                merge_key="bm25-only",
                content="bm25 only",
                accession_number="0000000000-00-000002",
                chunk_index=1,
                ticker="GS",
                company_name="Example Corp",
                form="10-K",
                filing_date=None,
                document_url=None,
                section=None,
                metadata={},
            ),
        ]

        service = QdrantHybridSearchService(True, 50, repository)
        matches = service.search("revenue", [0.1], 2, "GS", "10-K")

        repository.find_dense_chunks.assert_called_once_with([0.1], 50, "GS", "10-K")
        repository.find_bm25_chunks.assert_called_once_with("revenue", 50, "GS", "10-K")
        assert len(matches) == 2
        assert matches[0].accession_number == "0000000000-00-000001"
        assert matches[0].citation_number == 1

    def test_is_disabled_when_flag_off(self) -> None:
        service = QdrantHybridSearchService(False, 50, MagicMock())
        assert service.is_enabled() is False
