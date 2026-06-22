from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.models import Conversation, ConversationSettings, ConversationTurn, TurnMetadata
from app.services.rag_search import RagSearchService


class TestRagSearchConversation:
    def test_build_chat_messages_includes_prior_turns(self) -> None:
        prior = [
            ConversationTurn(role="user", content="Who are the directors?"),
            ConversationTurn(role="assistant", content="Directors include [1] ..."),
        ]
        messages = RagSearchService._build_chat_messages(
            prior,
            "How much was the buyback?",
            "[1] excerpt",
        )
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Who are the directors?"
        assert messages[2]["content"] == "Directors include [1] ..."
        assert "How much was the buyback?" in messages[3]["content"]
        assert "[1] excerpt" in messages[3]["content"]

    def test_continue_conversation_appends_user_and_assistant_turns(self) -> None:
        service = RagSearchService(
            ollama_client=MagicMock(),
            chunk_search_router=MagicMock(),
            pg_hybrid_search_service=MagicMock(),
            qdrant_hybrid_search_service=MagicMock(),
            ticker_resolver=MagicMock(),
        )
        service.answer_message = MagicMock(
            return_value=ConversationTurn(
                role="assistant",
                content="Answer",
                metadata=TurnMetadata(chat_model="qwen3:14b", vector_store="qdrant"),
            )
        )

        conversation = Conversation(
            id="test-id",
            settings=ConversationSettings(
                chat_model="qwen3:14b",
                vector_store="qdrant",
                chunk_count=10,
            ),
            turns=[],
            created_at=datetime.now(timezone.utc),
        )

        updated = service.continue_conversation(conversation, "Follow up question")
        assert len(updated.turns) == 2
        assert updated.turns[0].role == "user"
        assert updated.turns[0].content == "Follow up question"
        assert updated.turns[1].role == "assistant"
        assert updated.turns[1].content == "Answer"
