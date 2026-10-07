from collections.abc import Callable

from sungrid.chat import ChatServices, ChatState, handle_chat_message
from sungrid.runtime import create_chat_services, ingest_documents


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


def run_terminal_chat(
    services: ChatServices,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    state = ChatState()
    output_fn("SunGrid Copilot is ready. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            question = input_fn("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break

        if question.strip().lower() in {"quit", "exit"}:
            break

        try:
            reply = handle_chat_message(question, services, state)
        except KeyboardInterrupt:
            break

        output_fn(f"SunGrid: {reply.answer}")
        if reply.sources:
            output_fn("Sources:")
            for source in reply.sources:
                output_fn(
                    f"- {source.document_title} — {source.section_heading}"
                )

    output_fn("Goodbye.")
    return 0


def main() -> int:
    print("Preparing the SunGrid document library...")
    try:
        chunk_count = ingest_documents()
        services = create_chat_services()
    except (RuntimeError, ValueError) as exc:
        print(f"Startup error: {exc}")
        return 1

    print(f"Ready: {chunk_count} document sections indexed.")
    return run_terminal_chat(services)


if __name__ == "__main__":
    raise SystemExit(main())
