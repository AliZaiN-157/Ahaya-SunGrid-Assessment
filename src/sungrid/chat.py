import json
import logging
import os
import time
from contextvars import ContextVar
from uuid import uuid4
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from sungrid.eligibility import EligibilityFacts, EligibilityResult, collect_facts
from sungrid.taxonomy import Classification, SearchCategory


MIN_CONFIDENCE = 0.60
MIN_RELEVANCE = 0.35
UNSUPPORTED = "I couldn't find information about that in the SunGrid documents."
MAX_QUESTION_LENGTH = 2000
logger = logging.getLogger("sungrid.requests")
request_started = ContextVar("request_started", default=0.0)
request_metrics: ContextVar[dict[str, int | float] | None] = ContextVar(
    "request_metrics", default=None
)


def record_model_retry() -> None:
    metrics = request_metrics.get()
    if metrics is not None:
        metrics["retry_count"] += 1


def record_model_usage(usage, retries: int = 0) -> None:
    metrics = request_metrics.get()
    if metrics is None:
        return
    requests = int(getattr(usage, "requests", 0) or 0)
    metrics["model_requests"] += requests
    metrics["retry_count"] += retries
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        metrics[name] += int(getattr(usage, name, 0) or 0)
    cost = getattr(usage, "cost", None)
    if cost is not None:
        metrics["reported_cost_usd"] = float(
            metrics.get("reported_cost_usd", 0.0)
        ) + float(cost)


class RetrievedChunk(BaseModel):
    chunk_id: str | None = None
    document_title: str
    section_heading: str
    body: str
    score: float


class Source(BaseModel):
    document_title: str
    section_heading: str


class ChatReply(BaseModel):
    answer: str
    sources: list[Source]
    outcome: str = "answered"


class ChatState(BaseModel):
    eligibility_pending: bool = False
    eligibility_facts: EligibilityFacts = Field(default_factory=EligibilityFacts)


@dataclass
class ChatServices:
    classify: Callable[[str], Classification]
    search: Callable[[str, list[SearchCategory] | None], list[RetrievedChunk]]
    answer: Callable[[str, list[RetrievedChunk]], str]
    check_eligibility: Callable[[EligibilityFacts], EligibilityResult]


def handle_chat_message(
    question: str, services: ChatServices, state: ChatState | None = None
) -> ChatReply:
    state = state if state is not None else ChatState()
    request_id = str(uuid4())
    request_started.set(time.perf_counter())
    request_metrics.set(
        {
            "retry_count": 0,
            "model_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    if not question.strip():
        return _finish(
            request_id,
            ChatReply(
                answer="Please enter a question.", sources=[], outcome="rejected"
            ),
        )
    if len(question) > MAX_QUESTION_LENGTH:
        return _finish(
            request_id,
            ChatReply(
                answer="Please keep your question under 2,000 characters.",
                sources=[],
                outcome="rejected",
            ),
        )

    if state.eligibility_pending:
        return _handle_eligibility(question, services, state, request_id)

    try:
        classification = services.classify(question)
    except Exception as exc:
        return _finish(
            request_id,
            ChatReply(
                answer="I couldn't process that question right now. Please try again.",
                sources=[],
                outcome="error",
            ),
            error_type=type(exc).__name__,
        )
    if classification.primary_category == "non_relevant":
        return _finish(
            request_id,
            ChatReply(answer=UNSUPPORTED, sources=[], outcome="unsupported"),
            category="non_relevant",
            confidence=classification.confidence,
        )
    if classification.confidence < MIN_CONFIDENCE:
        return _finish(
            request_id,
            ChatReply(
                answer="I'm not sure which SunGrid topic you mean. Could you clarify what you're asking about?",
                sources=[],
                outcome="clarification",
            ),
            category=classification.primary_category,
            confidence=classification.confidence,
        )

    if classification.eligibility_intent:
        state.eligibility_pending = True
        state.eligibility_facts = collect_facts(state.eligibility_facts, question)
        return _eligibility_or_request_more(services, state, request_id, classification)

    search_categories: list[SearchCategory] = list(
        dict.fromkeys(
            category
            for category in (
                classification.primary_category,
                *classification.related_categories,
            )
            if category != "non_relevant"
        )
    )
    try:
        chunks = services.search(question, search_categories)
    except Exception as exc:
        return _finish(
            request_id,
            ChatReply(
                answer="I couldn't search the SunGrid documents right now. Please try again.",
                sources=[],
                outcome="error",
            ),
            category=classification.primary_category,
            searched_categories=search_categories,
            error_type=type(exc).__name__,
        )
    widened = not chunks or chunks[0].score < MIN_RELEVANCE
    if widened:
        try:
            chunks = services.search(question, None)
        except Exception as exc:
            return _finish(
                request_id,
                ChatReply(
                    answer="I couldn't search the SunGrid documents right now. Please try again.",
                    sources=[],
                    outcome="error",
                ),
                category=classification.primary_category,
                widened=True,
                searched_categories=search_categories,
                error_type=type(exc).__name__,
            )
    if not chunks or chunks[0].score < MIN_RELEVANCE:
        return _finish(
            request_id,
            ChatReply(answer=UNSUPPORTED, sources=[], outcome="unsupported"),
            category=classification.primary_category,
            searched_categories=search_categories,
            widened=widened,
        )

    try:
        answer = services.answer(question, chunks)
    except Exception as exc:
        return _finish(
            request_id,
            ChatReply(
                answer="I couldn't prepare an answer right now. Please try again.",
                sources=[],
                outcome="error",
            ),
            category=classification.primary_category,
            searched_categories=search_categories,
            widened=widened,
            error_type=type(exc).__name__,
        )
    if widened:
        answer = (
            "I broadened the search because the category search found no sufficiently relevant result.\n\n"
            + answer
        )
    sources = []
    seen_sources = set()
    for chunk in chunks:
        source_key = (chunk.document_title, chunk.section_heading)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(
                Source(document_title=source_key[0], section_heading=source_key[1])
            )
    return _finish(
        request_id,
        ChatReply(answer=answer, sources=sources),
        category=classification.primary_category,
        searched_categories=search_categories,
        confidence=classification.confidence,
        widened=widened,
        path="knowledge",
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        chunk_scores=[chunk.score for chunk in chunks],
    )


def _handle_eligibility(
    question: str, services: ChatServices, state: ChatState, request_id: str
) -> ChatReply:
    state.eligibility_facts = collect_facts(state.eligibility_facts, question)
    return _eligibility_or_request_more(services, state, request_id, None)


def _eligibility_or_request_more(
    services: ChatServices,
    state: ChatState,
    request_id: str,
    classification: Classification | None,
) -> ChatReply:
    missing = state.eligibility_facts.missing_labels()
    eligibility_values = state.eligibility_facts.model_dump()
    eligibility_fields = [
        name for name, value in eligibility_values.items() if value is not None
    ]
    missing_fields = [
        name for name, value in eligibility_values.items() if value is None
    ]
    category = classification.primary_category if classification else "incentive_rebate"
    confidence = classification.confidence if classification else None
    if missing:
        answer = (
            "To check rebate eligibility, please provide: " + ", ".join(missing) + "."
        )
        return _finish(
            request_id,
            ChatReply(answer=answer, sources=[], outcome="needs_more_input"),
            category=category,
            confidence=confidence,
            path="eligibility",
            eligibility_fields=eligibility_fields,
            missing_fields=missing_fields,
        )

    try:
        result = EligibilityResult.model_validate(
            services.check_eligibility(state.eligibility_facts)
        )
    except Exception as exc:
        state.eligibility_pending = False
        state.eligibility_facts = EligibilityFacts()
        return _finish(
            request_id,
            ChatReply(
                answer="I couldn't complete the eligibility check. Please try again later.",
                sources=[],
                outcome="error",
            ),
            category=category,
            confidence=confidence,
            path="eligibility",
            eligibility_fields=eligibility_fields,
            missing_fields=missing_fields,
            error_type=type(exc).__name__,
        )

    state.eligibility_pending = False
    state.eligibility_facts = EligibilityFacts()
    source = Source(
        document_title="SunGrid Cooperative — Incentive & Rebate Programs",
        section_heading="Rooftop Rebate Program",
    )
    if result.eligible:
        answer = f"The eligibility checks passed. Your estimated rebate is ${result.estimated_rebate_usd:,.2f}. This is an estimate, not a final award."
        return _finish(
            request_id,
            ChatReply(answer=answer, sources=[source], outcome="eligible"),
            category=category,
            confidence=confidence,
            path="eligibility",
            eligibility_fields=eligibility_fields,
            missing_fields=missing_fields,
        )
    answer = f"The household is not eligible: {result.reason} The rebate is $0. You may appeal within 30 days of the determination."
    return _finish(
        request_id,
        ChatReply(answer=answer, sources=[source], outcome="ineligible"),
        category=category,
        confidence=confidence,
        path="eligibility",
        eligibility_fields=eligibility_fields,
        missing_fields=missing_fields,
    )


def _finish(request_id: str, reply: ChatReply, **details) -> ChatReply:
    details.setdefault("path", "knowledge")
    event = {
        "event": "chat_request_completed",
        "request_id": request_id,
        "outcome": reply.outcome,
        "total_latency_ms": round((time.perf_counter() - request_started.get()) * 1000),
        "classifier_backend": os.getenv("CLASSIFIER_BACKEND", "llm"),
        "classifier_model": os.getenv("CLASSIFIER_MODEL_ID"),
        "answer_model": os.getenv("ANSWER_MODEL_ID"),
        **details,
    }
    if event["path"] == "knowledge":
        event["embedding_model"] = os.getenv(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
    if "category" in details:
        event["applied_filter"] = (
            None
            if details.get("widened")
            else details.get("searched_categories", details["category"])
        )
    metrics = request_metrics.get()
    if metrics is not None:
        event.update(metrics)
    logger.info(json.dumps(event, sort_keys=True))
    request_metrics.set(None)
    return reply
