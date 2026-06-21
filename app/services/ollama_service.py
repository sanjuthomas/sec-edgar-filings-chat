from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class OllamaModelService:
    def __init__(self, base_url: str, default_chat_model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_chat_model = default_chat_model

    def default_chat_model(self) -> str:
        return self._default_chat_model

    def list_chat_models(self) -> list[str]:
        try:
            with httpx.Client(base_url=self._base_url, timeout=10.0) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                payload = response.json()

            models = payload.get("models") or []
            chat_models = sorted(
                model["name"]
                for model in models
                if self._is_chat_model(model)
            )
            return chat_models or [self._default_chat_model]
        except Exception as exc:
            log.warning("Failed to load models from Ollama, using default only: %s", exc)
            return [self._default_chat_model]

    def is_known_chat_model(self, model: str) -> bool:
        return model in self.list_chat_models()

    @staticmethod
    def _is_chat_model(model: dict) -> bool:
        capabilities = model.get("capabilities")
        if not capabilities:
            return True
        return "completion" in capabilities
