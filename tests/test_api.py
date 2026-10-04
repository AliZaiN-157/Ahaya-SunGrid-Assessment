from fastapi.testclient import TestClient

from sungrid import api
from sungrid.chat import ChatReply


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
        raise RuntimeError("Could not build the local document index.")

    monkeypatch.setattr(api, "ingest_documents", fail_to_index)

    response = client.post("/ingest")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Could not build the local document index."
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
        "chat_state": {
            "eligibility_pending": False,
            "eligibility_facts": {
                "household_zip": None,
                "annual_income_usd": None,
                "system_size_kw": None,
                "installer_approved": None,
            },
        },
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
