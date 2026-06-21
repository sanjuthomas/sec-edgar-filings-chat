from __future__ import annotations

import time

import httpx
import psycopg

from app.exceptions import SearchError
from app.models import (
    ChunkMatch,
    Conversation,
    ConversationSettings,
    ConversationTurn,
    SearchForm,
    SearchResponse,
    TurnMetadata,
    VectorStoreType,
)
from app.services.chunk_search_router import ChunkSearchRouter
from app.services.ollama_client import OllamaClient, SYSTEM_PROMPT
from app.services.pg_hybrid_search import PgHybridSearchService
from app.services.retrieval_query import build_retrieval_query
from app.services.ticker_resolver import TickerResolver


class RagSearchService:
    def __init__(
        self,
        ollama_client: OllamaClient,
        chunk_search_router: ChunkSearchRouter,
        pg_hybrid_search_service: PgHybridSearchService,
        ticker_resolver: TickerResolver,
    ) -> None:
        self._ollama_client = ollama_client
        self._chunk_search_router = chunk_search_router
        self._pg_hybrid_search_service = pg_hybrid_search_service
        self._ticker_resolver = ticker_resolver

    def answer(self, form: SearchForm) -> SearchResponse:
        turn = self.answer_message(
            form.question,
            ConversationSettings(
                chat_model=form.chat_model,
                vector_store=form.vector_store,
                chunk_count=form.chunk_count,
                ticker=form.ticker,
                form=form.form,
            ),
            prior_turns=[],
        )
        meta = turn.metadata or TurnMetadata()
        return SearchResponse(
            question=form.question,
            answer=turn.content,
            sources=turn.sources,
            chat_model=meta.chat_model,
            vector_store=meta.vector_store,
            applied_ticker=meta.applied_ticker,
            ticker_inferred=meta.ticker_inferred,
            retrieval_ms=meta.retrieval_ms,
            generation_ms=meta.generation_ms,
        )

    def answer_message(
        self,
        message: str,
        settings: ConversationSettings,
        prior_turns: list[ConversationTurn],
    ) -> ConversationTurn:
        vector_store = settings.vector_store_type()
        meta = TurnMetadata(
            vector_store=vector_store.value,
            chat_model=settings.chat_model,
        )

        try:
            return self._answer_message(message, settings, prior_turns, vector_store, meta)
        except SearchError as exc:
            return ConversationTurn(role="assistant", content=str(exc), metadata=meta)
        except psycopg.OperationalError as exc:
            return ConversationTurn(
                role="assistant",
                content=(
                    "Could not connect to PostgreSQL for pgvector search. "
                    "Check that pgvector is running on port 5433 and set PGPASSWORD if required."
                ),
                metadata=meta,
            )
        except httpx.RequestError as exc:
            return ConversationTurn(
                role="assistant",
                content=f"Could not reach a required service: {exc}",
                metadata=meta,
            )
        except RuntimeError as exc:
            return ConversationTurn(
                role="assistant",
                content=str(exc),
                metadata=meta,
            )

    def _answer_message(
        self,
        message: str,
        settings: ConversationSettings,
        prior_turns: list[ConversationTurn],
        vector_store: VectorStoreType,
        meta: TurnMetadata,
    ) -> ConversationTurn:
        prior_user_messages = [
            turn.content for turn in prior_turns if turn.role == "user"
        ]
        retrieval_query = build_retrieval_query(message, prior_user_messages)

        retrieval_start = time.perf_counter()
        resolved_ticker = self._ticker_resolver.resolve(
            retrieval_query,
            settings.normalized_ticker(),
        )
        query_vector = self._ollama_client.embed(retrieval_query)

        if vector_store == VectorStoreType.PGVECTOR and self._pg_hybrid_search_service.is_enabled():
            sources = self._pg_hybrid_search_service.search(
                retrieval_query,
                query_vector,
                settings.chunk_count,
                resolved_ticker.ticker,
                settings.normalized_form(),
            )
        else:
            sources = self._chunk_search_router.find_similar_chunks(
                vector_store,
                query_vector,
                settings.chunk_count,
                resolved_ticker.ticker,
                settings.normalized_form(),
            )

        retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)
        meta.retrieval_ms = retrieval_ms
        meta.applied_ticker = resolved_ticker.ticker
        meta.ticker_inferred = resolved_ticker.inferred
        meta.source_count = len(sources)

        if not sources:
            return ConversationTurn(
                role="assistant",
                content=(
                    f"No matching filing excerpts were found in {vector_store.value} "
                    "for this question."
                ),
                sources=[],
                metadata=meta,
            )

        context = self._build_context(sources)
        generation_start = time.perf_counter()
        answer = self._ollama_client.chat_messages(
            settings.chat_model,
            self._build_chat_messages(prior_turns, message, context),
        )
        meta.generation_ms = int((time.perf_counter() - generation_start) * 1000)

        return ConversationTurn(
            role="assistant",
            content=answer,
            sources=sources,
            metadata=meta,
        )

    def continue_conversation(
        self,
        conversation: Conversation,
        message: str,
    ) -> Conversation:
        user_turn = ConversationTurn(role="user", content=message.strip())
        assistant_turn = self.answer_message(
            message,
            conversation.settings,
            conversation.turns,
        )
        updated_turns = [*conversation.turns, user_turn, assistant_turn]
        return conversation.model_copy(update={"turns": updated_turns})

    @staticmethod
    def _build_chat_messages(
        prior_turns: list[ConversationTurn],
        question: str,
        context: str,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in prior_turns:
            if turn.role not in {"user", "assistant"}:
                continue
            messages.append({"role": turn.role, "content": turn.content})
        messages.append(
            {
                "role": "user",
                "content": RagSearchService._user_prompt(question, context),
            }
        )
        return messages

    @staticmethod
    def _build_context(sources: list[ChunkMatch]) -> str:
        return "\n\n".join(RagSearchService._format_source(source) for source in sources)

    @staticmethod
    def _format_source(source: ChunkMatch) -> str:
        header = f"[{source.citation_number}] {source.ticker} {source.form}"
        if source.filing_date is not None:
            header += f" (filed {source.filing_date.isoformat()})"
        header += f" | accession {source.accession_number} | chunk {source.chunk_index}"
        if source.section:
            header += f" | section {source.section}"
        return f"{header}\n{source.content.strip()}"

    @staticmethod
    def _user_prompt(question: str, context: str) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Source excerpts:\n{context}\n\n"
            "Answer the question using only the excerpts above. Include inline citations like [1]."
        )
