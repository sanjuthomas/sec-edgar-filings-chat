from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from app.models import Conversation, ConversationSettings, ConversationTurn


class ConversationStore:
    def __init__(self, max_turns: int = 40) -> None:
        self._max_turns = max_turns
        self._conversations: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def create(self, settings: ConversationSettings) -> Conversation:
        conversation = Conversation(
            id=str(uuid.uuid4()),
            settings=settings,
            turns=[],
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._conversations[conversation.id] = conversation
        return conversation

    def get(self, conversation_id: str | None) -> Conversation | None:
        if not conversation_id:
            return None
        with self._lock:
            return self._conversations.get(conversation_id)

    def save(self, conversation: Conversation) -> None:
        trimmed_turns = conversation.turns[-self._max_turns :]
        updated = conversation.model_copy(update={"turns": trimmed_turns})
        with self._lock:
            self._conversations[updated.id] = updated

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)
