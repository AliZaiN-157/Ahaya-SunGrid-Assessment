import httpx
import pytest

from sungrid.model import create_classifier, retry_model_call


def test_missing_api_key_message_points_to_env_and_api(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError) as error:
        create_classifier()

    assert str(error.value) == (
        "Set OPENROUTER_API_KEY in .env before starting the API service."
    )


def test_transient_model_failure_retries_twice_then_returns():
    attempts = []
    delays = []

    def call():
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ConnectError("temporary")
        return "ok"

    assert retry_model_call(call, sleep=delays.append) == "ok"
    assert len(attempts) == 3
    assert delays == [0.25, 0.5]


def test_authentication_failure_is_not_retried():
    attempts = []
    delays = []

    def call():
        attempts.append(1)
        raise httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("POST", "https://example.invalid"),
            response=httpx.Response(401),
        )

    with pytest.raises(httpx.HTTPStatusError):
        retry_model_call(call, sleep=delays.append)
    assert len(attempts) == 1
    assert delays == []
