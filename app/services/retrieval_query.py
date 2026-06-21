from __future__ import annotations

import re

FOLLOW_UP_PATTERN = re.compile(
    r"\b("
    r"it|that|they|those|this|these|same|more|also|"
    r"how much|how many|what about|tell me more|"
    r"why|when|who else|which one|compared to|"
    r"follow up|earlier|previous|above|mentioned"
    r")\b",
    re.IGNORECASE,
)
FOLLOW_UP_START = re.compile(
    r"^\s*(and|also|what about|how much|how many|tell me more)\b",
    re.IGNORECASE,
)


def build_retrieval_query(current_message: str, prior_user_messages: list[str]) -> str:
    """Expand short or follow-up questions with prior user context for embedding search."""
    current = current_message.strip()
    if not current or not prior_user_messages:
        return current

    looks_like_follow_up = (
        len(current) <= 120
        and (FOLLOW_UP_PATTERN.search(current) is not None or FOLLOW_UP_START.match(current) is not None)
    )
    if not looks_like_follow_up:
        return current

    last_user_message = prior_user_messages[-1].strip()
    if not last_user_message:
        return current

    return f"{last_user_message}\n{current}"
