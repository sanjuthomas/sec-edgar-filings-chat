from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a financial research assistant in a multi-turn conversation about SEC EDGAR filings.
Use ONLY the source excerpts provided in the latest user message for new facts and citations.
You may use earlier conversation turns to interpret follow-up questions (e.g. "how much was it?",
"what about the board?", "tell me more").

Rules:
- Answer in clear, concise prose.
- Cite sources inline using bracketed numbers like [1], [2] that refer to the excerpt labels
  in the latest message.
- Do not invent facts, names, or dates that are not supported by the excerpts.
- Do not include chain-of-thought or reasoning sections; respond with the final answer only.
- If the latest excerpts do not contain enough information, say so clearly.
""".strip()


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        embedding_model: str,
        chat_temperature: float,
        chat_num_predict: int,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._embedding_model = embedding_model
        self._chat_temperature = chat_temperature
        self._chat_num_predict = chat_num_predict

    def embed(self, text: str) -> list[float]:
        with httpx.Client(base_url=self._base_url, timeout=120.0) as client:
            response = client.post(
                "/api/embeddings",
                json={"model": self._embedding_model, "prompt": text},
            )
            response.raise_for_status()
            payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama embedding response did not include an embedding vector.")
        return [float(value) for value in embedding]

    def chat(self, model: str, user_prompt: str) -> str:
        return self.chat_messages(
            model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

    def chat_messages(self, model: str, messages: list[dict[str, str]]) -> str:
        with httpx.Client(base_url=self._base_url, timeout=600.0) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self._chat_temperature,
                        "num_predict": self._chat_num_predict,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()

        message = payload.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama chat response did not include message content.")
        return content
