import pytest

from app.models import ConversationSettings, ConversationTurn
from app.services.conversation_store import ConversationStore
from app.services.retrieval_query import build_retrieval_query


class TestRetrievalQuery:
    def test_returns_current_message_when_no_history(self) -> None:
        assert build_retrieval_query("Who are the directors?", []) == "Who are the directors?"

    def test_expands_short_follow_up_with_prior_question(self) -> None:
        query = build_retrieval_query(
            "How much was it?",
            ["Do you know if the Adobe board approved a buyback program?"],
        )
        assert "Adobe board approved a buyback program" in query
        assert "How much was it?" in query

    def test_keeps_long_standalone_question(self) -> None:
        long_question = (
            "Summarize Goldman Sachs risk factors related to credit exposure, "
            "liquidity constraints, and market volatility disclosed in the latest 10-K filing."
        )
        query = build_retrieval_query(long_question, ["Who are the directors?"])
        assert query == long_question


class TestConversationStore:
    def test_trims_turns_to_max(self) -> None:
        store = ConversationStore(max_turns=4)
        settings = ConversationSettings(
            chat_model="qwen3:14b",
            vector_store="qdrant",
            chunk_count=10,
        )
        conversation = store.create(settings)
        conversation = conversation.model_copy(
            update={
                "turns": [
                    ConversationTurn(role="user", content=f"message {index}")
                    for index in range(6)
                ]
            }
        )
        store.save(conversation)
        loaded = store.get(conversation.id)
        assert loaded is not None
        assert len(loaded.turns) == 4
        assert loaded.turns[0].content == "message 2"
