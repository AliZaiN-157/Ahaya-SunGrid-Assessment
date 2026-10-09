import json

import httpx
import pytest
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from sungrid import model
from sungrid.chat import ChatServices, RetrievedChunk, handle_chat_message, request_metrics
from sungrid.eligibility import EligibilityResult
from sungrid.taxonomy import Classification
from sungrid.model import (
    SYSTEM_PROMPT,
    agent_usage_limits,
    count_model_retries,
    create_classifier,
    retry_model_call,
    run_agent,
)


def test_answer_prompt_requires_source_support_and_separates_processes():
    assert "do not infer details" in SYSTEM_PROMPT
    assert "explain them separately" in SYSTEM_PROMPT
    assert "do not imply one caused" in SYSTEM_PROMPT


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


def test_model_output_retry_is_counted():
    from pydantic_ai import ModelRequest, RetryPromptPart

    class Result:
        def all_messages(self):
            return [ModelRequest(parts=[RetryPromptPart(content="invalid output")])]

    assert count_model_retries(Result()) == 1


def test_exhausted_model_output_retries_are_counted():
    agent = Agent(
        TestModel(
            custom_output_args={
                "primary_category": "billing_account",
                "confidence": 0.9,
                "eligibility_intent": False,
            }
        ),
        output_type=Classification,
        retries=2,
    )

    @agent.output_validator
    async def always_retry(ctx, output):
        raise ModelRetry("retry output")

    metrics = {"retry_count": 0}
    token = request_metrics.set(metrics)
    try:
        with pytest.raises(
            UnexpectedModelBehavior, match="Exceeded maximum output retries"
        ):
            run_agent(agent, "question")
    finally:
        request_metrics.reset(token)

    assert metrics["retry_count"] == 2


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


def test_decision_classifier_returns_same_classification_shape(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CLASSIFIER_BACKEND", "decision")
    monkeypatch.setenv("CLASSIFIER_MODEL_ID", "typesafe/jev-1.13")
    response = {
        "answers": {
            "primary_category": {
                "type": "choice",
                "choice": "billing_account",
                "confidence": 0.87,
                "probabilities": {"billing_account": 0.9},
            },
            "related_incentive_rebate": {"type": "noul", "noul": 0.8},
            "related_program_policies": {"type": "noul", "noul": 0.1},
            "related_billing_account": {"type": "noul", "noul": 0.2},
            "related_technical_installation": {"type": "noul", "noul": 0.05},
            "related_company_updates": {"type": "noul", "noul": 0.05},
            "eligibility_intent": {"type": "noul", "noul": 0.05},
        },
        "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.001},
    }
    requests = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return response

    def post(url, **kwargs):
        requests.append((url, kwargs))
        return Response()

    monkeypatch.setattr(model.httpx, "post", post)

    metrics = {
        "retry_count": 0,
        "model_requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    token = request_metrics.set(metrics)
    try:
        classification = create_classifier()(
            "Did my missing rebate affect the billing adjustment?"
        )
    finally:
        request_metrics.reset(token)

    assert classification == Classification(
        primary_category="billing_account",
        related_categories=["incentive_rebate"],
        confidence=0.9,
        eligibility_intent=False,
    )
    url, request = requests[0]
    assert url.endswith("/api/alpha/decisions")
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["model"] == "typesafe/jev-1.13"
    assert request["json"]["questions"]["primary_category"]["type"] == "choice"
    assert metrics["model_requests"] == 1
    assert metrics["input_tokens"] == 100
    assert metrics["output_tokens"] == 20
    assert metrics["reported_cost_usd"] == 0.001


def test_classifier_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CLASSIFIER_BACKEND", "unknown")
    monkeypatch.setenv("CLASSIFIER_MODEL_ID", "provider/model")

    with pytest.raises(ValueError, match="CLASSIFIER_BACKEND"):
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
