import logging
from typing import Protocol

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the selected provider is missing required configuration."""


class LLMProviderError(RuntimeError):
    """Raised when a provider request fails or returns an invalid response."""


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAICompatibleProvider:
    """Small synchronous adapter for OpenAI Chat Completions compatible APIs."""

    def __init__(self, api_key: str | None, base_url: str, model: str, timeout: float = 45.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise LLMConfigurationError("LLM_API_KEY is not configured.")
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Provider returned no text content")
            return content.strip()
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            # Do not log response bodies or request headers: provider errors can contain
            # sensitive information and prompt context contains contract text.
            logger.warning("LLM provider request failed (model=%s)", self.model)
            raise LLMProviderError("LLM provider request failed.") from None


def create_llm_provider() -> LLMProvider:
    settings = get_settings()
    return OpenAICompatibleProvider(
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )
