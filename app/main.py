from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.config import Settings, get_settings
from app.repositories.filing_metadata_repo import FilingMetadataRepository
from app.repositories.pgvector_repo import (
    PgBm25ChunkRepository,
    PgVectorChunkRepository,
    PgVectorHybridChunkRepository,
)
from app.repositories.qdrant_repo import QdrantChunkRepository
from app.routes.search import router as search_router
from app.services.conversation_store import ConversationStore
from app.services.chunk_search_router import ChunkSearchRouter
from app.services.ollama_client import OllamaClient
from app.services.ollama_service import OllamaModelService
from app.services.pg_hybrid_search import PgHybridSearchService
from app.services.rag_search import RagSearchService
from app.services.ticker_resolver import TickerResolver

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    filing_metadata_repo = FilingMetadataRepository(settings.database_url)
    pgvector_repo = PgVectorChunkRepository(settings.database_url)
    qdrant_repo = QdrantChunkRepository(settings.qdrant_url, settings.qdrant_collection)
    pg_hybrid_vector_repo = PgVectorHybridChunkRepository(settings.database_url)
    pg_bm25_repo = PgBm25ChunkRepository(settings.database_url)

    ticker_resolver = TickerResolver()
    ollama_model_service = OllamaModelService(
        settings.ollama_base_url,
        settings.ollama_chat_model,
    )
    ollama_client = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_embedding_model,
        settings.ollama_chat_temperature,
        settings.ollama_chat_num_predict,
    )
    chunk_search_router = ChunkSearchRouter([pgvector_repo, qdrant_repo])
    pg_hybrid_search_service = PgHybridSearchService(
        settings.pgsearch_enabled,
        settings.hybrid_retrieval_top_k,
        pg_hybrid_vector_repo,
        pg_bm25_repo,
    )
    rag_search_service = RagSearchService(
        ollama_client,
        chunk_search_router,
        pg_hybrid_search_service,
        ticker_resolver,
    )
    conversation_store = ConversationStore(max_turns=settings.conversation_max_turns)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            tickers = filing_metadata_repo.find_distinct_tickers()
            companies = filing_metadata_repo.find_distinct_companies()
            ticker_resolver.load_metadata(tickers, companies)
            log.info(
                "Loaded %d tickers and %d companies for ticker resolution",
                len(tickers),
                len(companies),
            )
        except Exception as exc:
            log.warning("Failed to load filing metadata for ticker resolution: %s", exc)
        yield

    app = FastAPI(title="SEC EDGAR Filings Chat", lifespan=lifespan)
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret_key)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.templates = templates
    app.state.settings = settings
    app.state.rag_search_service = rag_search_service
    app.state.ollama_model_service = ollama_model_service
    app.state.ticker_resolver = ticker_resolver
    app.state.filing_metadata_repo = filing_metadata_repo
    app.state.conversation_store = conversation_store

    app.include_router(search_router)
    return app


app = create_app()
