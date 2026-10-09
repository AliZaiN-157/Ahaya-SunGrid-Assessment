import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sungrid.chat import ChatReply, ChatState, handle_chat_message
from sungrid.runtime import configure_logging, create_chat_services, ingest_documents


class ChatSession(BaseModel):
    session_id: str | None = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_state: ChatSession = Field(default_factory=ChatSession)


class AgentResponse(ChatReply):
    chat_state: ChatSession


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    ingest_documents()
    yield


app = FastAPI(title="SunGrid Copilot API", lifespan=lifespan)
app.state.eligibility_sessions = {}
MAX_ELIGIBILITY_SESSIONS = 100


@app.post("/ingest")
def ingest() -> dict[str, int]:
    try:
        return {"chunks_indexed": ingest_documents()}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(request: AgentRequest) -> AgentResponse:
    session_id = request.chat_state.session_id
    sessions: dict[str, ChatState] = app.state.eligibility_sessions
    state = sessions.get(session_id) if session_id else ChatState()
    if state is None:
        raise HTTPException(
            status_code=400,
            detail="This eligibility session expired. Please start the check again.",
        )

    try:
        reply = handle_chat_message(
            request.question, create_chat_services(), state
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response_state = ChatSession()
    if state.eligibility_pending:
        if session_id is None:
            if len(sessions) >= MAX_ELIGIBILITY_SESSIONS:
                sessions.pop(next(iter(sessions)))
            session_id = secrets.token_urlsafe(32)
        sessions[session_id] = state
        response_state.session_id = session_id
    elif session_id:
        sessions.pop(session_id, None)

    return AgentResponse(**reply.model_dump(), chat_state=response_state)
