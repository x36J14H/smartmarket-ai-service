import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.bot import chat, chat_stream
from app.services.history import clear_session, get_history

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    stream: bool = False  # включить SSE streaming


@router.post("")
async def chat_endpoint(req: ChatRequest):
    """
    Основной чат-эндпоинт.

    Если stream=false (по умолчанию) — возвращает полный JSON-ответ:
    {
        "answer": "...",
        "sources": [...],
        "intent": "products",
        "session_id": "...",
        "needs_clarification": false,
        "clarification_question": null
    }

    Если stream=true — возвращает Server-Sent Events (SSE) поток:
    - {"type": "meta", "intent": "...", "sources": [...], ...}
    - {"type": "chunk", "text": "часть ответа"}
    - {"type": "chunk", "text": "ещё часть"}
    - {"type": "replace", "text": "полный ответ (если были нечистые ссылки)"}  — опционально
    - {"type": "done"}

    Каждое событие — отдельная JSON-строка, разделённые переводом строки.
    """
    session_id = req.session_id or str(uuid.uuid4())

    if req.stream:
        stream_iter = await chat_stream(req.question, session_id)
        return StreamingResponse(
            stream_iter,
            media_type="text/event-stream",
            headers={
                "X-Session-Id": session_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # отключаем буферизацию в nginx
            },
        )

    result = await chat(req.question, session_id)
    result["session_id"] = session_id
    return result


@router.post("/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """
    Алиас для streaming — всегда стримит, параметр stream игнорируется.
    Удобно для клиентов которые хотят явный URL для SSE.
    """
    session_id = req.session_id or str(uuid.uuid4())
    stream_iter = await chat_stream(req.question, session_id)
    return StreamingResponse(
        stream_iter,
        media_type="text/event-stream",
        headers={
            "X-Session-Id": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{session_id}/history")
def get_chat_history(session_id: str):
    """Получить историю диалога по session_id."""
    history = get_history(session_id)
    return {
        "session_id": session_id,
        "messages":   history,
        "count":      len(history),
    }


@router.delete("/{session_id}")
def clear_chat(session_id: str):
    """Очистить историю диалога."""
    clear_session(session_id)
    return {"cleared": session_id}
