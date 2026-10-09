import json

import httpx
from pydantic_ai.usage import RunUsage

from sungrid.chat import (
    ChatServices,
    ChatState,
    Classification,
    RetrievedChunk,
    handle_chat_message,
    record_model_usage,
)
from sungrid.eligibility import EligibilityResult


def chunk(score=0.91, title="Billing FAQs"):
    return RetrievedChunk(
        document_title=title,
        section_heading="When are bills issued?",
        body="Bills are issued on the 1st and are due within 21 days.",
        score=score,
    )


def services(search, classification=None, answer=None):
    return ChatServices(
        classify=lambda question: (
            classification
            or Classification(
                primary_category="billing_account",
                confidence=0.9,
                eligibility_intent=False,
            )
        ),
        search=search,
        answer=answer
        or (
            lambda question, context: (
                "Bills arrive on the first and are due within 21 days."
            )
        ),
        check_eligibility=lambda facts: EligibilityResult(
            eligible=True, reason="All checks passed.", estimated_rebate_usd=2000
        ),
    )


def test_searches_primary_category_and_returns_citation():
    seen = []
    services_ = services(lambda question, category: seen.append(category) or [chunk()])

    reply = handle_chat_message("When is my bill due?", services_)

    assert seen == [["billing_account"]]
    assert reply.answer.startswith("Bills arrive")
    assert [(s.document_title, s.section_heading) for s in reply.sources] == [
        ("Billing FAQs", "When are bills issued?")
    ]


def test_multi_topic_question_searches_both_categories_and_cites_both_documents():
    searched_categories = []
    rebate = chunk(title="Rebate Refund & Billing Adjustment Notice")
    billing = chunk(title="Billing & Account FAQs")
    classification = Classification(
        primary_category="incentive_rebate",
        related_categories=["billing_account"],
        confidence=0.9,
        eligibility_intent=False,
    )

    def search(question, categories):
        searched_categories.extend(categories)
        return [rebate, billing]

    reply = handle_chat_message(
        "My rebate is delayed and there is a billing adjustment. What should I do?",
        services(search, classification),
    )

    assert searched_categories == ["incentive_rebate", "billing_account"]
    assert {source.document_title for source in reply.sources} == {
        "Rebate Refund & Billing Adjustment Notice",
        "Billing & Account FAQs",
    }


def test_low_confidence_asks_for_clarification_without_search():
    calls = []
    services_ = services(
        lambda question, category: calls.append(category) or [chunk()],
        Classification(
            primary_category="billing_account",
            confidence=0.59,
            eligibility_intent=False,
        ),
    )

    reply = handle_chat_message("Can you help?", services_)

    assert "which SunGrid topic" in reply.answer
    assert calls == []
    assert reply.sources == []


def test_non_relevant_question_is_not_searched():
    calls = []
    services_ = services(
        lambda question, category: calls.append(category) or [],
        Classification(
            primary_category="non_relevant", confidence=0.99, eligibility_intent=False
        ),
    )

    reply = handle_chat_message("What is the moon made of?", services_)

    assert "SunGrid documents" in reply.answer
    assert calls == []


def test_weak_filtered_search_widens_once_and_discloses_it():
    calls = []
    ambiguous = chunk(title="Rebate Refund & Billing Adjustment Notice")

    def search(question, category):
        calls.append(category)
        return [chunk(0.2)] if category else [ambiguous]

    reply = handle_chat_message("How is my rebate reversal billed?", services(search))

    assert calls == [["billing_account"], None]
    assert "broadened the search" in reply.answer
    assert (
        reply.sources[0].document_title == "Rebate Refund & Billing Adjustment Notice"
    )


def test_weak_widened_search_does_not_call_answerer():
    calls = []
    services_ = services(
        lambda question, category: [chunk(0.2)],
        answer=lambda question, context: calls.append(context) or "Should not be used",
    )

    reply = handle_chat_message("What is my account status?", services_)

    assert "couldn't find" in reply.answer
    assert calls == []
    assert reply.sources == []


def test_ambiguous_document_has_both_categories():
    from pathlib import Path

    from sungrid.knowledge import load_chunks

    chunks = load_chunks(Path("docs"))
    ambiguous = [
        chunk
        for chunk in chunks
        if chunk.document_id == "06_ambiguous_rebate_billing_adjustments"
    ]

    assert ambiguous
    assert all(
        chunk.categories == ["incentive_rebate", "billing_account"]
        for chunk in ambiguous
    )
    assert all(chunk.primary_category == "incentive_rebate" for chunk in ambiguous)


def test_qdrant_search_filters_category_membership():
    from sungrid.knowledge import QdrantKnowledgeStore

    class FakeClient:
        query_filter = None

        def search(self, **kwargs):
            self.query_filter = kwargs["query_filter"]
            return []

    client = FakeClient()
    store = QdrantKnowledgeStore(client, "openai/text-embedding-3-small")
    for category in ("incentive_rebate", "billing_account"):
        store.search([0.0], [category])
        assert client.query_filter.must[1].key == "categories"
        assert client.query_filter.must[1].match.value == category


def test_qdrant_multi_category_search_keeps_results_from_each_document():
    from types import SimpleNamespace
    from qdrant_client import models

    from sungrid.knowledge import QdrantKnowledgeStore

    class FakeClient:
        searches = []

        def search(self, **kwargs):
            condition = kwargs["query_filter"].must[1]
            assert condition.key == "categories"
            assert isinstance(condition.match, models.MatchValue)
            category = condition.match.value
            self.searches.append((category, kwargs["limit"]))
            hits = [
                SimpleNamespace(
                    id=f"ambiguous-{index}",
                    payload={
                        "document_title": "Rebate Refund & Billing Adjustment Notice",
                        "section_heading": f"Section {index}",
                        "body": "Billing adjustment text.",
                    },
                    score=1 - index / 100,
                )
                for index in range(4)
            ]
            if category == "billing_account":
                hits.append(
                    SimpleNamespace(
                        id="billing-faq",
                        payload={
                            "document_title": "Billing & Account FAQs",
                            "section_heading": "My rebate has not appeared",
                            "body": "Contact the Incentive Program committee.",
                        },
                        score=0.95,
                    )
                )
            return hits

    client = FakeClient()
    store = QdrantKnowledgeStore(client, "openai/text-embedding-3-small")

    results = store.search([0.0], ["incentive_rebate", "billing_account"])

    assert client.searches == [("incentive_rebate", 8), ("billing_account", 8)]
    assert {chunk.document_title for chunk in results} == {
        "Rebate Refund & Billing Adjustment Notice",
        "Billing & Account FAQs",
    }
    assert sum(
        chunk.document_title == "Rebate Refund & Billing Adjustment Notice"
        for chunk in results
    ) <= 2


def test_eligibility_question_requests_all_missing_facts():
    state = ChatState()
    classification = Classification(
        primary_category="incentive_rebate", confidence=0.95, eligibility_intent=True
    )
    reply = handle_chat_message(
        "Am I eligible for the rooftop rebate?",
        services(lambda q, c: [], classification),
        state,
    )

    assert reply.outcome == "needs_more_input"
    assert all(
        field in reply.answer
        for field in ("ZIP code", "annual household income", "system size", "installer")
    )
    assert state.eligibility_pending is True


def test_supplied_eligibility_facts_are_checked_without_guessing():
    state = ChatState()
    calls = []
    classification = Classification(
        primary_category="incentive_rebate", confidence=0.95, eligibility_intent=True
    )
    services_ = services(
        lambda q, c: [], classification, answer=lambda q, chunks: "must not compose"
    )
    services_.check_eligibility = lambda facts: (
        calls.append(facts.model_dump())
        or EligibilityResult(
            eligible=True, reason="All checks passed.", estimated_rebate_usd=2000
        )
    )
    handle_chat_message("Am I eligible?", services_, state)

    reply = handle_chat_message(
        "ZIP 94101, annual income $80,000, system size 5 kW, installer approved",
        services_,
        state,
    )

    assert calls == [
        {
            "household_zip": "94101",
            "annual_income_usd": 80000,
            "system_size_kw": 5,
            "installer_approved": True,
        }
    ]
    assert reply.outcome == "eligible"
    assert "$2,000" in reply.answer
    assert "estimate" in reply.answer.lower()
    assert (
        reply.sources[0].document_title
        == "SunGrid Cooperative — Incentive & Rebate Programs"
    )
    assert reply.sources[0].section_heading == "Rooftop Rebate Program"


def test_ineligible_result_stops_before_answer_composition():
    state = ChatState()
    answer_calls = []
    classification = Classification(
        primary_category="incentive_rebate", confidence=0.95, eligibility_intent=True
    )
    services_ = services(
        lambda q, c: [],
        classification,
        answer=lambda q, chunks: answer_calls.append(q) or "guessed",
    )
    services_.check_eligibility = lambda facts: EligibilityResult(
        eligible=False,
        reason="Installer is not on the SunGrid approved installer list.",
        estimated_rebate_usd=0,
    )
    handle_chat_message("Am I eligible?", services_, state)

    reply = handle_chat_message(
        "ZIP 94101, annual income $80,000, system size 5 kW, installer not approved",
        services_,
        state,
    )

    assert reply.outcome == "ineligible"
    assert "$0" in reply.answer
    assert "appeal within 30 days" in reply.answer
    assert reply.sources[0].section_heading == "Rooftop Rebate Program"
    assert answer_calls == []


def test_income_number_is_not_mistaken_for_zip_code():
    state = ChatState()
    classification = Classification(
        primary_category="incentive_rebate", confidence=0.95, eligibility_intent=True
    )
    calls = []
    services_ = services(lambda q, c: [], classification)
    services_.check_eligibility = lambda facts: (
        calls.append(facts)
        or EligibilityResult(
            eligible=True, reason="All checks passed.", estimated_rebate_usd=1000
        )
    )
    handle_chat_message("Am I eligible?", services_, state)

    reply = handle_chat_message(
        "Annual income 80000, system size 5 kW, installer approved", services_, state
    )

    assert reply.outcome == "needs_more_input"
    assert "ZIP code" in reply.answer
    assert calls == []


def test_empty_and_overlong_questions_are_rejected():
    services_ = services(lambda q, c: (_ for _ in ()).throw(AssertionError("searched")))
    assert handle_chat_message("  ", services_).outcome == "rejected"
    assert handle_chat_message("x" * 2001, services_).outcome == "rejected"


def test_model_failure_is_safe_and_logs_no_raw_question(caplog):
    caplog.set_level("INFO", logger="sungrid.requests")
    secret_question = "private query should not appear in logs"
    services_ = services(
        lambda q, c: [],
        answer=lambda q, chunks: "",
    )
    services_.classify = lambda question: (_ for _ in ()).throw(
        RuntimeError("secret traceback detail")
    )

    reply = handle_chat_message(secret_question, services_)

    assert reply.outcome == "error"
    assert "secret traceback" not in reply.answer
    assert secret_question not in caplog.text
    assert "secret traceback" not in caplog.text
    assert "request_id" in caplog.text
    assert "total_latency_ms" in caplog.text


def test_eligibility_logs_show_progress_without_member_values(caplog):
    caplog.set_level("INFO", logger="sungrid.requests")
    state = ChatState()
    classification = Classification(
        primary_category="incentive_rebate", confidence=0.95, eligibility_intent=True
    )
    services_ = services(lambda question, category: [], classification)

    first = handle_chat_message(
        "Can you check my rebate eligibility? ZIP 94101", services_, state
    )
    second = handle_chat_message(
        "Annual income $80,000, system size 5 kW, installer approved",
        services_,
        state,
    )
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "sungrid.requests"
    ]

    assert first.outcome == "needs_more_input"
    assert second.outcome == "eligible"
    assert events[0]["event"] == "chat_request_completed"
    assert events[0]["path"] == "eligibility"
    assert events[0]["missing_fields"] == [
        "annual_income_usd",
        "system_size_kw",
        "installer_approved",
    ]
    assert events[1]["eligibility_fields"] == [
        "household_zip",
        "annual_income_usd",
        "system_size_kw",
        "installer_approved",
    ]
    assert "94101" not in caplog.text
    assert "80000" not in caplog.text
    assert "5 kW" not in caplog.text


def test_request_log_includes_retry_and_token_usage(caplog, monkeypatch):
    from sungrid.model import retry_model_call

    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    caplog.set_level("INFO", logger="sungrid.requests")
    attempts = []

    def classify(question):
        def call():
            if not attempts:
                attempts.append(True)
                raise httpx.ConnectError("temporary")
            record_model_usage(
                RunUsage(requests=2, input_tokens=12, output_tokens=4, cost=0.001)
            )
            return Classification(
                primary_category="billing_account",
                confidence=0.9,
                eligibility_intent=False,
            )

        return retry_model_call(call, sleep=lambda _: None)

    services_ = services(lambda question, category: [chunk()])
    services_.classify = classify

    reply = handle_chat_message("When is my bill due?", services_)
    event = json.loads(
        next(record.message for record in caplog.records if record.name == "sungrid.requests")
    )

    assert reply.outcome == "answered"
    assert event["retry_count"] == 1
    assert event["model_requests"] == 2
    assert event["input_tokens"] == 12
    assert event["output_tokens"] == 4
    assert event["reported_cost_usd"] == 0.001
    assert event["embedding_model"] == "test-embedding-model"


def test_retrieval_failure_does_not_show_exception():
    reply = handle_chat_message(
        "When is my bill due?",
        services(
            lambda q, c: (_ for _ in ()).throw(RuntimeError("private host and key"))
        ),
    )
    assert reply.outcome == "error"
    assert "private host" not in reply.answer


def test_eligibility_failure_makes_no_claim():
    state = ChatState()
    classification = Classification(
        primary_category="incentive_rebate", confidence=0.95, eligibility_intent=True
    )
    services_ = services(lambda q, c: [], classification)
    services_.check_eligibility = lambda facts: (_ for _ in ()).throw(
        RuntimeError("private tool error")
    )
    handle_chat_message("Am I eligible?", services_, state)

    reply = handle_chat_message(
        "ZIP 94101, annual income $80,000, system size 5 kW, installer approved",
        services_,
        state,
    )

    assert reply.outcome == "error"
    assert "couldn't complete" in reply.answer
    assert "eligible" not in reply.answer.lower()
    assert "private tool error" not in reply.answer


def test_invalid_eligibility_result_fails_safely():
    state = ChatState()
    classification = Classification(
        primary_category="incentive_rebate", confidence=0.95, eligibility_intent=True
    )
    services_ = services(lambda q, c: [], classification)
    services_.check_eligibility = lambda facts: {"eligible": True}
    handle_chat_message("Am I eligible?", services_, state)

    reply = handle_chat_message(
        "ZIP 94101, annual income $80,000, system size 5 kW, installer approved",
        services_,
        state,
    )

    assert reply.outcome == "error"
    assert "couldn't complete" in reply.answer
    assert "eligible" not in reply.answer.lower()
