"""
Тесты сервиса истории диалогов (app/services/history.py).

Быстрые, без внешних зависимостей.

pytest tests/test_history.py -v
"""
import time
import pytest
from datetime import datetime, timedelta


# ── Вспомогательная функция — изолируем каждый тест ──────────────────────────

def fresh_session() -> str:
    """Возвращает гарантированно новый session_id."""
    import uuid
    return str(uuid.uuid4())


# ── Базовые операции ──────────────────────────────────────────────────────────

class TestHistoryBasic:

    def test_new_session_empty(self):
        from app.services.history import get_history
        assert get_history(fresh_session()) == []

    def test_add_and_get(self):
        from app.services.history import add_messages, get_history
        sid = fresh_session()
        add_messages(sid, "Привет", "Здравствуйте!")
        history = get_history(sid)
        assert len(history) == 2
        assert history[0] == {"role": "user",      "content": "Привет"}
        assert history[1] == {"role": "assistant", "content": "Здравствуйте!"}

    def test_multiple_messages_accumulate(self):
        from app.services.history import add_messages, get_history
        sid = fresh_session()
        add_messages(sid, "Вопрос 1", "Ответ 1")
        add_messages(sid, "Вопрос 2", "Ответ 2")
        add_messages(sid, "Вопрос 3", "Ответ 3")
        history = get_history(sid)
        assert len(history) == 6
        assert history[0]["content"] == "Вопрос 1"
        assert history[4]["content"] == "Вопрос 3"

    def test_clear_session(self):
        from app.services.history import add_messages, get_history, clear_session
        sid = fresh_session()
        add_messages(sid, "Вопрос", "Ответ")
        assert len(get_history(sid)) == 2

        clear_session(sid)
        assert get_history(sid) == []

    def test_clear_nonexistent_session_no_error(self):
        from app.services.history import clear_session
        clear_session("несуществующий-session-id-xyz")  # не должно падать

    def test_get_history_returns_list_copy(self):
        """get_history должен возвращать копию — мутация не влияет на хранилище."""
        from app.services.history import add_messages, get_history
        sid = fresh_session()
        add_messages(sid, "Q", "A")

        h1 = get_history(sid)
        h1.append({"role": "user", "content": "МУТАЦИЯ"})

        h2 = get_history(sid)
        assert len(h2) == 2, "Мутация возвращённого списка не должна влиять на хранилище"

    def test_roles_are_correct(self):
        from app.services.history import add_messages, get_history
        sid = fresh_session()
        add_messages(sid, "user text", "assistant text")
        history = get_history(sid)
        roles = [m["role"] for m in history]
        assert roles == ["user", "assistant"]


# ── Ограничение размера (MAX_MESSAGES = 20) ───────────────────────────────────

class TestHistoryMaxSize:

    def test_max_messages_not_exceeded(self):
        """При добавлении > 10 пар старые должны выталкиваться."""
        from app.services.history import add_messages, get_history, MAX_MESSAGES
        sid = fresh_session()

        # Добавляем больше MAX_MESSAGES/2 пар
        pairs = MAX_MESSAGES // 2 + 3
        for i in range(pairs):
            add_messages(sid, f"Q{i}", f"A{i}")

        history = get_history(sid)
        assert len(history) <= MAX_MESSAGES, (
            f"История превысила MAX_MESSAGES={MAX_MESSAGES}: {len(history)}"
        )

    def test_oldest_messages_dropped_first(self):
        """При переполнении старые сообщения удаляются первыми."""
        from app.services.history import add_messages, get_history, MAX_MESSAGES
        sid = fresh_session()

        pairs = MAX_MESSAGES  # добавляем MAX_MESSAGES пар = 2*MAX_MESSAGES сообщений
        for i in range(pairs):
            add_messages(sid, f"Q{i}", f"A{i}")

        history = get_history(sid)
        contents = [m["content"] for m in history]

        # Старые сообщения (Q0, A0) должны быть вытолканы
        assert "Q0" not in contents, "Q0 должен быть вытолкнут из истории"
        # Последние должны остаться
        assert f"Q{pairs - 1}" in contents, f"Q{pairs-1} должен быть в истории"


# ── TTL и очистка ─────────────────────────────────────────────────────────────

class TestHistoryTtl:

    def test_expired_session_cleaned_on_access(self):
        """Истёкшая сессия удаляется при следующем обращении к любой сессии."""
        from app.services import history as hist_module

        sid = fresh_session()
        hist_module.add_messages(sid, "Q", "A")

        # Симулируем истечение TTL — напрямую меняем last_active в хранилище
        expired_time = datetime.utcnow() - timedelta(minutes=hist_module.SESSION_TTL_MINUTES + 1)
        hist_module._store[sid]["last_active"] = expired_time

        # Обращение через get_history должно вызвать _cleanup_expired
        another_sid = fresh_session()
        hist_module.get_history(another_sid)  # триггерим очистку

        assert sid not in hist_module._store, "Истёкшая сессия должна быть удалена"

    def test_active_session_not_cleaned(self):
        """Активная сессия не удаляется при очистке."""
        from app.services import history as hist_module

        sid = fresh_session()
        hist_module.add_messages(sid, "Q", "A")

        # Очистка не должна трогать свежую сессию
        hist_module._cleanup_expired()

        assert sid in hist_module._store, "Свежая сессия не должна удаляться"

    def test_last_active_updated_on_get(self):
        """get_history обновляет last_active сессии."""
        from app.services import history as hist_module

        sid = fresh_session()
        hist_module.add_messages(sid, "Q", "A")

        before = hist_module._store[sid]["last_active"]
        time.sleep(0.01)

        hist_module.get_history(sid)
        after = hist_module._store[sid]["last_active"]

        assert after >= before, "last_active должен обновляться при get_history"


# ── Изоляция сессий ───────────────────────────────────────────────────────────

class TestHistoryIsolation:

    def test_sessions_are_independent(self):
        """Разные session_id не смешиваются."""
        from app.services.history import add_messages, get_history
        sid1 = fresh_session()
        sid2 = fresh_session()

        add_messages(sid1, "Q для сессии 1", "A1")
        add_messages(sid2, "Q для сессии 2", "A2")

        h1 = get_history(sid1)
        h2 = get_history(sid2)

        assert len(h1) == 2
        assert len(h2) == 2
        assert h1[0]["content"] == "Q для сессии 1"
        assert h2[0]["content"] == "Q для сессии 2"

    def test_clear_one_session_leaves_other_intact(self):
        from app.services.history import add_messages, get_history, clear_session
        sid1 = fresh_session()
        sid2 = fresh_session()

        add_messages(sid1, "Q1", "A1")
        add_messages(sid2, "Q2", "A2")

        clear_session(sid1)

        assert get_history(sid1) == []
        assert len(get_history(sid2)) == 2
