from fastapi.testclient import TestClient

from sungrid import api
from sungrid.chat import ChatReply, ChatServices
from sungrid.eligibility import EligibilityResult
from sungrid.taxonomy import Classification


client = TestClient(api.app)


def test_api_indexes_fixed_documents_at_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "ingest_documents", lambda: calls.append(True) or 14)

    with TestClient(api.app) as startup_client:
        assert startup_client.get("/openapi.json").status_code == 200

    assert calls == [True]


def test_ingest_endpoint_reports_indexed_chunks(monkeypatch):
    monkeypatch.setattr(api, "ingest_documents", lambda: 14)

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"chunks_indexed": 14}


def test_ingest_endpoint_returns_service_unavailable_when_indexing_fails(monkeypatch):
    def fail_to_index():
        raise RuntimeError(
            "Could not build the document index. Check OPENROUTER_API_KEY, "
            "EMBEDDING_MODEL, OpenRouter connectivity, and Qdrant."
        )

    monkeypatch.setattr(api, "ingest_documents", fail_to_index)

    response = client.post("/ingest")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Could not build the document index. Check OPENROUTER_API_KEY, "
        "EMBEDDING_MODEL, OpenRouter connectivity, and Qdrant."
    }


def test_agent_endpoint_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(api, "create_chat_services", lambda: object())
    monkeypatch.setattr(
        api,
        "handle_chat_message",
        lambda question, services, state: ChatReply(
            answer=f"Answer to: {question}", sources=[], outcome="answered"
        ),
    )

    response = client.post("/agent/run", json={"question": "When is my bill due?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer to: When is my bill due?",
        "sources": [],
        "outcome": "answered",
        "chat_state": {"session_id": None},
    }


def test_agent_endpoint_keeps_eligibility_facts_out_of_client_state(monkeypatch):
    services = ChatServices(
        classify=lambda question: Classification(
            primary_category="incentive_rebate",
            confidence=0.95,
            eligibility_intent=True,
        ),
        search=lambda question, category: [],
        answer=lambda question, chunks: "unused",
        check_eligibility=lambda facts: EligibilityResult(
            eligible=True, reason="Checks passed.", estimated_rebate_usd=2000
        ),
    )
    monkeypatch.setattr(api, "create_chat_services", lambda: services)

    first = client.post("/agent/run", json={"question": "Am I eligible?"})
    returned_state = first.json()["chat_state"]

    assert first.json()["outcome"] == "needs_more_input"
    assert returned_state["session_id"]
    assert "eligibility_facts" not in returned_state
    assert "94101" not in first.text

    second = client.post(
        "/agent/run",
        json={
            "question": "ZIP 94101, annual income $80,000, system size 5 kW, installer approved",
            "chat_state": returned_state,
        },
    )

    assert second.status_code == 200
    assert second.json()["outcome"] == "eligible"
    assert second.json()["chat_state"] == {"session_id": None}

    expired = client.post(
        "/agent/run",
        json={"question": "Continue", "chat_state": returned_state},
    )
    assert expired.status_code == 400


def test_agent_endpoint_rejects_expired_eligibility_session():
    response = client.post(
        "/agent/run",
        json={"question": "Continue", "chat_state": {"session_id": "unknown"}},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "This eligibility session expired. Please start the check again."
    }


def test_agent_endpoint_returns_service_unavailable_when_not_configured(monkeypatch):
    def fail_to_create_services():
        raise ValueError(
            "Set OPENROUTER_API_KEY in .env before starting the API service."
        )

    monkeypatch.setattr(api, "create_chat_services", fail_to_create_services)

    response = client.post("/agent/run", json={"question": "When is my bill due?"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Set OPENROUTER_API_KEY in .env before starting the API service."
    }
