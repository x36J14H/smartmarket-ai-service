from collections import deque
from datetime import datetime, timedelta

MAX_MESSAGES = 20
SESSION_TTL_MINUTES = 60

_store: dict[str, dict] = {}


def _now() -> datetime:
    return datetime.utcnow()


def get_history(session_id: str) -> list[dict]:
    _cleanup_expired()
    session = _store.get(session_id)
    if not session:
        return []
    session["last_active"] = _now()
    return list(session["messages"])


def add_messages(session_id: str, user_text: str, assistant_text: str) -> None:
    if session_id not in _store:
        _store[session_id] = {
            "messages": deque(maxlen=MAX_MESSAGES),
            "last_active": _now(),
            "created_at": _now(),
        }
    session = _store[session_id]
    session["messages"].append({"role": "user",      "content": user_text})
    session["messages"].append({"role": "assistant", "content": assistant_text})
    session["last_active"] = _now()


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)


def _cleanup_expired() -> None:
    cutoff = _now() - timedelta(minutes=SESSION_TTL_MINUTES)
    expired = [sid for sid, s in _store.items() if s["last_active"] < cutoff]
    for sid in expired:
        del _store[sid]
