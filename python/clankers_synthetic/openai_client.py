"""OpenAI API client wrapper with retry, timeout, and provenance metadata."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class OpenAIClientError(Exception):
    """Base error for OpenAI client operations."""


class OpenAIClient:
    """Thin wrapper around OpenAI API with retry and provenance tracking.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        max_retries: Max retries on transient/rate-limit errors.
        timeout: Request timeout in seconds.
        base_url: Optional custom base URL for API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = base_url
        self._client: Any = None  # lazy init

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise OpenAIClientError(
                    "openai package not installed. Install with: pip install openai"
                ) from e
            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def request_json(
        self,
        request: Any,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Send a structured JSON request to the OpenAI API.

        Args:
            request: LLMRequest with system_message, user_message, model,
                temperature, and max_tokens attributes.
            n: Number of completions to generate.
            seed: Optional seed for deterministic sampling.

        Returns:
            List of dicts, each with keys:
            - "content": parsed JSON dict from the response
            - "provenance": dict with model, request_id, prompt_hash,
              response_hash, timestamp

        Raises:
            OpenAIClientError: On non-retryable errors or max retries exceeded.
        """
        client = self._get_client()
        prompt_hash = hashlib.sha256(
            (request.system_message + request.user_message).encode()
        ).hexdigest()[:16]

        messages = [
            {"role": "system", "content": request.system_message},
            {"role": "user", "content": request.user_message},
        ]

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": request.model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "n": n,
                    "response_format": {"type": "json_object"},
                }
                if seed is not None:
                    kwargs["seed"] = seed

                response = client.chat.completions.create(**kwargs)

                results = []
                for choice in response.choices:
                    content_str = choice.message.content or "{}"
                    try:
                        content = json.loads(content_str)
                    except json.JSONDecodeError as e:
                        raise OpenAIClientError(f"LLM returned non-JSON response: {e}") from e

                    response_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

                    provenance = {
                        "model": response.model,
                        "request_id": response.id,
                        "prompt_hash": prompt_hash,
                        "response_hash": response_hash,
                        "timestamp": time.time(),
                    }
                    results.append({"content": content, "provenance": provenance})

                return results

            except OpenAIClientError:
                raise
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Retry on rate limit, timeout, or transient server errors
                is_retryable = any(
                    kw in error_str
                    for kw in [
                        "rate_limit",
                        "rate limit",
                        "timeout",
                        "timed out",
                        "server_error",
                        "500",
                        "502",
                        "503",
                        "529",
                        "connection",
                        "overloaded",
                    ]
                )
                if not is_retryable or attempt >= self.max_retries:
                    raise OpenAIClientError(
                        f"OpenAI request failed after {attempt + 1} attempts: {e}"
                    ) from e
                # Exponential backoff
                time.sleep(min(2**attempt, 30))

        raise OpenAIClientError(
            f"OpenAI request failed after {self.max_retries + 1} attempts: {last_error}"
        )
