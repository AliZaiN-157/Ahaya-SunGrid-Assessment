import pytest

from sungrid.chat import ChatServices, RetrievedChunk
from sungrid.eligibility import EligibilityResult
from sungrid.taxonomy import Classification


def terminal_module():
    try:
        from sungrid import terminal
    except ModuleNotFoundError:
        pytest.fail("The terminal chat entry point is missing.")
    return terminal


def scripted_input(messages: list[str]):
    remaining = iter(messages)
    return lambda _prompt: next(remaining)


def test_terminal_chat_prints_answer_and_sources_until_exit():
    terminal = terminal_module()
    outputs: list[str] = []
    services = ChatServices(
        classify=lambda _question: Classification(
            primary_category="billing_account",
            confidence=0.98,
            eligibility_intent=False,
        ),
        search=lambda _question, _categories: [
            RetrievedChunk(
                chunk_id="billing-due-date",
                document_title="SunGrid Billing FAQs",
                section_heading="When are bills issued?",
                body="Bills are due within 21 days.",
                score=0.92,
            )
        ],
        answer=lambda _question, _chunks: "Bills are due within 21 days.",
        check_eligibility=lambda _facts: pytest.fail(
            "Knowledge questions must not call the eligibility checker."
        ),
    )

    result = terminal.run_terminal_chat(
        services,
        input_fn=scripted_input(["When is my bill due?", "exit"]),
        output_fn=outputs.append,
    )

    rendered = "\n".join(outputs)
    assert result == 0
    assert "SunGrid: Bills are due within 21 days." in rendered
    assert "Sources:" in rendered
    assert "SunGrid Billing FAQs — When are bills issued?" in rendered
    assert rendered.endswith("Goodbye.")


def test_terminal_chat_keeps_eligibility_state_between_messages():
    terminal = terminal_module()
    outputs: list[str] = []
    classifications = iter(
        [
            Classification(
                primary_category="incentive_rebate",
                confidence=0.97,
                eligibility_intent=True,
            )
        ]
    )
    services = ChatServices(
        classify=lambda _question: next(classifications),
        search=lambda _question, _categories: pytest.fail(
            "Eligibility questions must not search documents."
        ),
        answer=lambda _question, _chunks: pytest.fail(
            "Eligibility questions must not use document answer composition."
        ),
        check_eligibility=lambda _facts: EligibilityResult(
            eligible=True,
            reason="All checks passed.",
            estimated_rebate_usd=2000,
        ),
    )

    result = terminal.run_terminal_chat(
        services,
        input_fn=scripted_input(
            [
                "Am I eligible for the rooftop rebate?",
                "ZIP 94101, annual income $80,000, system size 5 kW, installer approved",
                "quit",
            ]
        ),
        output_fn=outputs.append,
    )

    rendered = "\n".join(outputs)
    assert result == 0
    assert "please provide: ZIP code" in rendered
    assert "estimated rebate is $2,000.00" in rendered
    assert "Rooftop Rebate Program" in rendered


def test_terminal_chat_treats_end_of_input_as_a_clean_exit():
    terminal = terminal_module()
    outputs: list[str] = []

    def end_input(_prompt: str) -> str:
        raise EOFError

    services = ChatServices(
        classify=lambda _question: pytest.fail("No question should be classified."),
        search=lambda _question, _categories: [],
        answer=lambda _question, _chunks: "unused",
        check_eligibility=lambda _facts: pytest.fail(
            "No eligibility check should run."
        ),
    )

    result = terminal.run_terminal_chat(
        services, input_fn=end_input, output_fn=outputs.append
    )

    assert result == 0
    assert outputs[-1] == "Goodbye."


def test_terminal_main_reports_startup_failure(monkeypatch, capsys):
    terminal = terminal_module()

    def fail_to_index():
        raise RuntimeError("Document memory is unavailable.")

    monkeypatch.setattr(terminal, "ingest_documents", fail_to_index)

    result = terminal.main()

    assert result == 1
    assert "Startup error: Document memory is unavailable." in capsys.readouterr().out
