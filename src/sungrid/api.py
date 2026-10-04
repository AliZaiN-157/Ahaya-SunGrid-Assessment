from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sungrid.chat import ChatReply, ChatState, handle_chat_message
from sungrid.runtime import create_chat_services, ingest_documents


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_state: ChatState = Field(default_factory=ChatState)


class AgentResponse(ChatReply):
    chat_state: ChatState


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ingest_documents()
    yield


app = FastAPI(title="SunGrid Copilot API", lifespan=lifespan)


@app.post("/ingest")
def ingest() -> dict[str, int]:
    try:
        return {"chunks_indexed": ingest_documents()}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(request: AgentRequest) -> AgentResponse:
    try:
        reply = handle_chat_message(
            request.question, create_chat_services(), request.chat_state
        )
        return AgentResponse(**reply.model_dump(), chat_state=request.chat_state)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
