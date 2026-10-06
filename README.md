# SunGrid Cooperative Copilot

A small FastAPI service that answers questions from the fixed SunGrid documents and checks rooftop rebate eligibility.

Stack: Python 3.11, PydanticAI/Pydantic, OpenRouter embeddings and LLMs, and Docker Qdrant.

## Run locally

Prerequisites: Python 3.11, [uv](https://docs.astral.sh/uv/), and Docker Desktop.

1. From the repository root, install the app and test dependencies: `uv sync --extra dev`.
2. Set `OPENROUTER_API_KEY`, `CLASSIFIER_MODEL_ID`, and `ANSWER_MODEL_ID` in the existing `.env` file. `EMBEDDING_MODEL` defaults to `openai/text-embedding-3-small`; `AGENT_MAX_STEPS` defaults to `4`. The API reads `.env` directly.
3. Start Qdrant: `docker compose up -d qdrant`.
4. Start the API: `uv run uvicorn sungrid.api:app --reload`.

The API indexes the fixed `docs/` folder when it starts and reuses the index while it is current. Run the tests with `uv run pytest`.

## API

Open `http://localhost:8000/docs` to try the API.

- `POST /ingest` indexes the fixed `docs/` folder and returns the chunk count.
- `POST /agent/run` accepts `{"question": "When is my bill due?"}` and returns an answer with sources. Send the returned `chat_state` in the next request to continue a multi-turn eligibility check. It contains an opaque session ID; pending eligibility facts stay in process memory and are cleared when the check finishes. Restarting the API expires pending sessions.

## Document categories

The rebate refund and billing adjustment notice is primarily **Incentive & Rebate Programs** and is also tagged **Billing & Account**, so either category search can find it.

The assessment's Part C design write-up is in [PART_C_DESIGN.md](PART_C_DESIGN.md).
