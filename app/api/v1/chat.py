import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.bot import chat
from app.services.bot_hyde import chat_hyde
from app.services.history import clear_session

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


@router.post("")
def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = chat(req.question, session_id)
    result["session_id"] = session_id
    return result


@router.post("/hyde")
def chat_hyde_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = chat_hyde(req.question, session_id)
    result["session_id"] = session_id
    return result


@router.delete("/{session_id}")
def clear_chat(session_id: str):
    clear_session(session_id)
    return {"cleared": session_id}
