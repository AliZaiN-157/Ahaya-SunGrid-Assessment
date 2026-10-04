import json

import httpx
import pytest
from pydantic_ai.models.test import TestModel

from sungrid import model
from sungrid.chat import ChatServices, RetrievedChunk, handle_chat_message
from sungrid.eligibility import EligibilityResult
from sungrid.model import agent_usage_limits, create_classifier, retry_model_call


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


def test_agent_step_limit_defaults_to_four_and_can_be_configured(monkeypatch):
    monkeypatch.delenv("AGENT_MAX_STEPS", raising=False)
    assert agent_usage_limits().request_limit == 4

    monkeypatch.setenv("AGENT_MAX_STEPS", "2")
    limits = agent_usage_limits()
    assert limits.request_limit == 2
    assert limits.tool_calls_limit == 2


def test_agent_step_limit_rejects_invalid_configuration(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "zero")

    with pytest.raises(ValueError, match="AGENT_MAX_STEPS"):
        agent_usage_limits()


def test_classifier_rejects_invalid_step_limit_during_setup(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CLASSIFIER_MODEL_ID", "provider/model")
    monkeypatch.setenv("AGENT_MAX_STEPS", "0")

    with pytest.raises(ValueError, match="AGENT_MAX_STEPS"):
        create_classifier()


def test_classifier_run_usage_is_added_to_request_log(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CLASSIFIER_MODEL_ID", "provider/model")
    monkeypatch.setenv("AGENT_MAX_STEPS", "4")
    monkeypatch.setattr(model, "OpenRouterProvider", lambda api_key: object())
    monkeypatch.setattr(
        model,
        "OpenRouterModel",
        lambda model_id, provider: TestModel(
            custom_output_args={
                "primary_category": "billing_account",
                "confidence": 0.9,
                "eligibility_intent": False,
            }
        ),
    )
    caplog.set_level("INFO", logger="sungrid.requests")
    services = ChatServices(
        classify=create_classifier(),
        search=lambda question, category: [
            RetrievedChunk(
                document_title="Billing FAQs",
                section_heading="Bill due date",
                body="Bills are due within 21 days.",
                score=0.9,
            )
        ],
        answer=lambda question, chunks: "Bills are due within 21 days.",
        check_eligibility=lambda facts: EligibilityResult(
            eligible=False, reason="Not eligible.", estimated_rebate_usd=0
        ),
    )

    reply = handle_chat_message("When is my bill due?", services)
    event = json.loads(
        next(record.message for record in caplog.records if record.name == "sungrid.requests")
    )

    assert reply.outcome == "answered"
    assert event["model_requests"] == 1
    assert event["input_tokens"] > 0
    assert event["output_tokens"] > 0
