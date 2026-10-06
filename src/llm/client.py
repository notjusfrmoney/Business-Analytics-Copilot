from __future__ import annotations

import json

import requests

from src.utils.config import get_openrouter_settings


class LLMError(RuntimeError):
    """Raised when the model call fails or returns invalid output."""


class LLMClient:
    """Small OpenRouter chat-completions client for text output."""

    def __init__(self, timeout: float = 60.0) -> None:
        api_key, self.model, base_url = get_openrouter_settings()
        self.api_key = api_key
        self.endpoint = f"{base_url}/chat/completions"
        self.timeout = timeout

    def generate_text(self, prompt: str) -> str:
        """Send a prompt and return the hosted model's message content."""
        request_body: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise LLMError("OpenRouter request timed out.") from exc
        except requests.ConnectionError as exc:
            raise LLMError("Could not connect to OpenRouter.") from exc
        except requests.RequestException as exc:
            raise LLMError("OpenRouter request failed before a response was received.") from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise LLMError("OpenRouter returned an invalid response body.") from exc

        if not response.ok:
            raise LLMError(
                self._format_api_error(
                    response.status_code,
                    response_data,
                    api_key=self.api_key,
                )
            )

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenRouter response did not contain a model message.") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("OpenRouter returned an empty model response.")
        return content.strip()

    @staticmethod
    def _format_api_error(
        status_code: int,
        response_data: Any,
        *,
        api_key: str = "",
    ) -> str:
        if status_code == 401:
            return "OpenRouter rejected the API key (HTTP 401). Check OPENROUTER_API_KEY."
        if status_code == 429:
            return "OpenRouter rate limit or quota reached (HTTP 429). Try again later."

        provider_message = ""
        if isinstance(response_data, dict):
            error = response_data.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                provider_message = error["message"].strip()
        if api_key:
            provider_message = provider_message.replace(api_key, "[redacted]")
        provider_message = provider_message[:240]
        if status_code in {400, 404, 422}:
            detail = f": {provider_message}" if provider_message else ""
            return (
                f"OpenRouter rejected the model or request (HTTP {status_code})"
                f"{detail}. Check OPENROUTER_MODEL and request support."
            )
        detail = f": {provider_message}" if provider_message else ""
        return f"OpenRouter API error (HTTP {status_code}){detail}."
