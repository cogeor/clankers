"""Tests for clankers_synthetic.openai_client module."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from clankers_synthetic.openai_client import OpenAIClient, OpenAIClientError
from clankers_synthetic.specs import LLMRequest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockChoice:
    """Mimics openai ChatCompletion choice."""

    def __init__(self, content: str) -> None:
        self.message = type("Msg", (), {"content": content})()


class MockResponse:
    """Mimics openai ChatCompletion response."""

    def __init__(
        self,
        choices: list[MockChoice],
        model: str = "gpt-5",
        id: str = "req-123",
    ) -> None:
        self.choices = choices
        self.model = model
        self.id = id


def _make_request(**overrides) -> LLMRequest:
    defaults = {
        "system_message": "You are a robot planner.",
        "user_message": "Pick up the red cube.",
        "model": "gpt-5",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)


def _make_client(**overrides) -> OpenAIClient:
    """Create an OpenAIClient with a pre-injected mock _client."""
    client = OpenAIClient(**overrides)
    mock_openai = MagicMock()
    client._client = mock_openai
    return client, mock_openai


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRequestJsonSuccess:
    """test_request_json_success -- mock client returns valid JSON."""

    def test_returns_parsed_content_and_provenance(self) -> None:
        client, mock_openai = _make_client()
        response_data = {"plan_id": "p1", "skills": []}
        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice(json.dumps(response_data))],
        )

        request = _make_request()
        results = client.request_json(request)

        assert len(results) == 1
        assert results[0]["content"] == response_data
        assert "provenance" in results[0]
        prov = results[0]["provenance"]
        assert prov["model"] == "gpt-5"
        assert prov["request_id"] == "req-123"


class TestRequestJsonMultipleCandidates:
    """test_request_json_multiple_candidates -- n=2, verify 2 results."""

    def test_returns_multiple_results(self) -> None:
        client, mock_openai = _make_client()
        data1 = {"plan_id": "p1", "skills": []}
        data2 = {"plan_id": "p2", "skills": []}
        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[
                MockChoice(json.dumps(data1)),
                MockChoice(json.dumps(data2)),
            ],
        )

        request = _make_request()
        results = client.request_json(request, n=2)

        assert len(results) == 2
        assert results[0]["content"] == data1
        assert results[1]["content"] == data2


class TestRetryOnRateLimit:
    """test_retry_on_rate_limit -- first call raises rate limit, second succeeds."""

    @patch("time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep: MagicMock) -> None:
        client, mock_openai = _make_client(max_retries=3)
        response_data = {"ok": True}
        mock_openai.chat.completions.create.side_effect = [
            RuntimeError("rate_limit exceeded"),
            MockResponse(choices=[MockChoice(json.dumps(response_data))]),
        ]

        request = _make_request()
        results = client.request_json(request)

        assert len(results) == 1
        assert results[0]["content"] == response_data
        assert mock_openai.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()


class TestRetryOnTimeout:
    """test_retry_on_timeout -- first call raises timeout, second succeeds."""

    @patch("time.sleep")
    def test_retries_on_timeout(self, mock_sleep: MagicMock) -> None:
        client, mock_openai = _make_client(max_retries=3)
        response_data = {"ok": True}
        mock_openai.chat.completions.create.side_effect = [
            RuntimeError("Request timed out"),
            MockResponse(choices=[MockChoice(json.dumps(response_data))]),
        ]

        request = _make_request()
        results = client.request_json(request)

        assert len(results) == 1
        assert results[0]["content"] == response_data
        assert mock_openai.chat.completions.create.call_count == 2


class TestMaxRetriesExceeded:
    """test_max_retries_exceeded -- all attempts fail, raises OpenAIClientError."""

    @patch("time.sleep")
    def test_raises_after_max_retries(self, mock_sleep: MagicMock) -> None:
        client, mock_openai = _make_client(max_retries=2)
        mock_openai.chat.completions.create.side_effect = RuntimeError(
            "rate_limit exceeded"
        )

        request = _make_request()
        with pytest.raises(OpenAIClientError, match="failed after 3 attempts"):
            client.request_json(request)

        # 1 initial + 2 retries = 3 calls
        assert mock_openai.chat.completions.create.call_count == 3


class TestNonJsonResponseRaises:
    """test_non_json_response_raises -- response content is not valid JSON."""

    def test_raises_on_invalid_json(self) -> None:
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice("This is not JSON at all!")],
        )

        request = _make_request()
        with pytest.raises(OpenAIClientError, match="non-JSON response"):
            client.request_json(request)


class TestNonRetryableError:
    """test_non_retryable_error -- auth error raises immediately."""

    def test_raises_immediately_on_auth_error(self) -> None:
        client, mock_openai = _make_client(max_retries=3)
        mock_openai.chat.completions.create.side_effect = RuntimeError(
            "Invalid API key provided"
        )

        request = _make_request()
        with pytest.raises(OpenAIClientError, match="failed after 1 attempts"):
            client.request_json(request)

        # Should NOT retry -- only 1 call
        assert mock_openai.chat.completions.create.call_count == 1


class TestProvenanceMetadata:
    """test_provenance_metadata -- verify all provenance fields populated."""

    def test_provenance_fields_complete(self) -> None:
        client, mock_openai = _make_client()
        response_data = {"result": "ok"}
        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice(json.dumps(response_data))],
            model="gpt-5-turbo",
            id="chatcmpl-abc123",
        )

        request = _make_request()
        results = client.request_json(request)

        prov = results[0]["provenance"]
        assert prov["model"] == "gpt-5-turbo"
        assert prov["request_id"] == "chatcmpl-abc123"
        assert isinstance(prov["prompt_hash"], str)
        assert len(prov["prompt_hash"]) == 16
        assert isinstance(prov["response_hash"], str)
        assert len(prov["response_hash"]) == 16
        assert isinstance(prov["timestamp"], float)
        assert prov["timestamp"] > 0

    def test_prompt_hash_deterministic(self) -> None:
        client, mock_openai = _make_client()
        response_data = {"result": "ok"}
        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice(json.dumps(response_data))],
        )

        request = _make_request()
        r1 = client.request_json(request)

        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice(json.dumps(response_data))],
        )
        r2 = client.request_json(request)

        assert r1[0]["provenance"]["prompt_hash"] == r2[0]["provenance"]["prompt_hash"]

    def test_different_responses_different_hash(self) -> None:
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice(json.dumps({"a": 1}))],
        )
        r1 = client.request_json(_make_request())

        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice(json.dumps({"b": 2}))],
        )
        r2 = client.request_json(_make_request())

        assert (
            r1[0]["provenance"]["response_hash"] != r2[0]["provenance"]["response_hash"]
        )


class TestLazyClientImportError:
    """test_lazy_client_import_error -- when openai not installed, raises."""

    def test_raises_import_error(self) -> None:
        client = OpenAIClient(api_key="test-key")
        # Do NOT inject _client -- let it try to import openai
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(OpenAIClientError, match="openai package not installed"):
                client.request_json(_make_request())


class TestSeedPassedThrough:
    """test_seed_passed_through -- verify seed kwarg reaches the API call."""

    def test_seed_in_api_call(self) -> None:
        client, mock_openai = _make_client()
        response_data = {"plan": "ok"}
        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice(json.dumps(response_data))],
        )

        request = _make_request()
        client.request_json(request, seed=42)

        call_kwargs = mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("seed") == 42 or call_kwargs[1].get("seed") == 42

    def test_no_seed_when_not_provided(self) -> None:
        client, mock_openai = _make_client()
        response_data = {"plan": "ok"}
        mock_openai.chat.completions.create.return_value = MockResponse(
            choices=[MockChoice(json.dumps(response_data))],
        )

        request = _make_request()
        client.request_json(request)

        call_kwargs = mock_openai.chat.completions.create.call_args
        # seed should not be in kwargs when not provided
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "seed" not in all_kwargs
