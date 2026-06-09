"""
Тесты HTTP API через FastAPI TestClient (без запущенного сервера).

Проверяем:
- GET /health
- POST /api/v1/products  (upsert + deleted)
- DELETE /api/v1/products/{id}
- POST /api/v1/products/search
- POST /api/v1/faq
- POST /api/v1/navigation
- POST /api/v1/chat  (без streaming, с mock бота)
- POST /api/v1/chat  (с stream=true)
- POST /api/v1/chat/stream
- GET  /api/v1/chat/{session_id}/history
- DELETE /api/v1/chat/{session_id}
- GET /api/v1/settings/llm
- POST /api/v1/settings/llm/provider/{provider}
- POST /api/v1/settings/llm/active

pytest tests/test_api.py -v
"""
import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# ── Фикстура клиента ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    TestClient с замоканными Qdrant и embedding — чтобы не нужен был
    реальный Qdrant при тестировании API-слоя.
    """
    with patch("app.db.qdrant.get_client") as mock_qdrant, \
         patch("app.db.qdrant.ensure_collections"), \
         patch("app.ml_models.embedder.get_model"), \
         patch("app.ml_models.embedder.get_sparse_model"):

        mock_qdrant.return_value = MagicMock()

        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── Products API ──────────────────────────────────────────────────────────────

class TestProductsApi:

    def _mock_qdrant(self):
        """Контекстный менеджер для мока qdrant upsert/delete."""
        return patch("app.api.v1.products.upsert", return_value=2), \
               patch("app.api.v1.products.get_client")

    def test_upsert_products_success(self, client):
        with patch("app.api.v1.products.upsert", return_value=2), \
             patch("app.api.v1.products.get_client"):
            resp = client.post("/api/v1/products", json=[
                {
                    "id":             "550e8400-e29b-41d4-a716-446655440001",
                    "name":           "Samsung Galaxy S24",
                    "price":          89990.0,
                    "embedding_text": "Смартфон Samsung Galaxy S24 256ГБ",
                },
                {
                    "id":             "550e8400-e29b-41d4-a716-446655440002",
                    "name":           "iPhone 15",
                    "price":          79990.0,
                    "embedding_text": "Смартфон Apple iPhone 15 128ГБ",
                },
            ])
        assert resp.status_code == 200
        data = resp.json()
        print(f"\n  upsert_products: {data}")
        assert "inserted" in data
        assert "deleted" in data
        assert data["deleted"] == 0

    def test_upsert_products_with_soft_delete(self, client):
        with patch("app.api.v1.products.upsert", return_value=1), \
             patch("app.api.v1.products.get_client") as mock_qdrant_cls:
            mock_client = MagicMock()
            mock_qdrant_cls.return_value = mock_client

            resp = client.post("/api/v1/products", json=[
                {
                    "id":             "aaa00000-0000-0000-0000-000000000001",
                    "name":           "Обычный товар",
                    "embedding_text": "текст",
                    "deleted":        False,
                },
                {
                    "id":             "aaa00000-0000-0000-0000-000000000002",
                    "name":           "Удалённый товар",
                    "embedding_text": "текст",
                    "deleted":        True,
                },
            ])
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 1
        # delete должен был вызваться
        mock_client.delete.assert_called_once()

    def test_delete_product(self, client):
        with patch("app.api.v1.products.get_client") as mock_qdrant_cls:
            mock_client = MagicMock()
            mock_qdrant_cls.return_value = mock_client

            pid = "550e8400-e29b-41d4-a716-446655440001"
            resp = client.delete(f"/api/v1/products/{pid}")

        assert resp.status_code == 200
        assert resp.json()["deleted"] == pid

    def test_search_products(self, client):
        mock_hit = MagicMock()
        mock_hit.payload = {"source_id": "search-uuid-1", "text": "iPhone"}
        mock_hit.score = 0.9

        with patch("app.api.v1.products.search", return_value=[mock_hit, mock_hit]), \
             patch("app.api.v1.products.rerank", AsyncMock(return_value=[mock_hit])), \
             patch("app.api.v1.products.filter_available_ids",
                   AsyncMock(return_value={"search-uuid-1"})):
            resp = client.post("/api/v1/products/search", json={"query": "iPhone", "top_k": 5})

        assert resp.status_code == 200
        data = resp.json()
        print(f"\n  search_products: {data}")
        assert "ids" in data
        assert isinstance(data["ids"], list)

    def test_upsert_products_empty_list(self, client):
        with patch("app.api.v1.products.upsert", return_value=0), \
             patch("app.api.v1.products.get_client"):
            resp = client.post("/api/v1/products", json=[])
        assert resp.status_code == 200
        assert resp.json()["inserted"] == 0


# ── FAQ и Navigation API ──────────────────────────────────────────────────────

class TestKnowledgeApi:

    def test_upsert_faq(self, client):
        with patch("app.api.v1.knowledge.upsert", return_value=1), \
             patch("app.api.v1.knowledge.get_client"):
            resp = client.post("/api/v1/faq", json=[{
                "id":       "660e8400-e29b-41d4-a716-446655440001",
                "question": "Как оформить возврат?",
                "answer":   "В течение 14 дней",
                "category": "Возвраты",
                "deleted":  False,
            }])
        assert resp.status_code == 200
        data = resp.json()
        assert data["inserted"] == 1
        assert data["deleted"] == 0

    def test_upsert_faq_with_delete(self, client):
        with patch("app.api.v1.knowledge.upsert", return_value=0), \
             patch("app.api.v1.knowledge.get_client") as mock_qdrant_cls:
            mock_client = MagicMock()
            mock_qdrant_cls.return_value = mock_client

            resp = client.post("/api/v1/faq", json=[{
                "id":       "660e8400-e29b-41d4-a716-446655440002",
                "question": "Удалить?",
                "answer":   "Да",
                "deleted":  True,
            }])

        assert resp.status_code == 200
        mock_client.delete.assert_called_once()

    def test_delete_faq(self, client):
        with patch("app.api.v1.knowledge.get_client") as mock_qdrant_cls:
            mock_qdrant_cls.return_value = MagicMock()
            fid = "660e8400-e29b-41d4-a716-446655440001"
            resp = client.delete(f"/api/v1/faq/{fid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == fid

    def test_upsert_navigation(self, client):
        with patch("app.api.v1.knowledge.upsert", return_value=1), \
             patch("app.api.v1.knowledge.get_client"):
            resp = client.post("/api/v1/navigation", json=[{
                "id":          "770e8400-e29b-41d4-a716-446655440001",
                "title":       "Каталог смартфонов",
                "description": "Все смартфоны в наличии",
                "url":         "/catalog/smartphones",
                "deleted":     False,
            }])
        assert resp.status_code == 200
        assert resp.json()["inserted"] == 1

    def test_delete_navigation(self, client):
        with patch("app.api.v1.knowledge.get_client") as mock_qdrant_cls:
            mock_qdrant_cls.return_value = MagicMock()
            nid = "770e8400-e29b-41d4-a716-446655440001"
            resp = client.delete(f"/api/v1/navigation/{nid}")
        assert resp.status_code == 200


# ── Chat API ──────────────────────────────────────────────────────────────────

class TestChatApi:

    def _mock_chat_result(self, answer: str = "Тестовый ответ", intent: str = "products"):
        return {
            "answer":              answer,
            "sources":             [{"collection": "products", "score": 0.9,
                                     "text": "тест", "product_id": "test-uuid", "url": None}],
            "intent":              intent,
            "needs_clarification": False,
            "clarification_question": None,
        }

    def test_chat_returns_answer(self, client):
        with patch("app.api.v1.chat.chat", AsyncMock(return_value=self._mock_chat_result())):
            resp = client.post("/api/v1/chat", json={
                "question": "Есть ли у вас iPhone?",
            })
        assert resp.status_code == 200
        data = resp.json()
        print(f"\n  /chat response keys: {list(data.keys())}")
        assert "answer" in data
        assert "session_id" in data
        assert "intent" in data
        assert "sources" in data
        assert "needs_clarification" in data
        assert data["answer"] == "Тестовый ответ"

    def test_chat_generates_session_id_if_not_provided(self, client):
        with patch("app.api.v1.chat.chat", AsyncMock(return_value=self._mock_chat_result())):
            resp = client.post("/api/v1/chat", json={"question": "тест"})
        sid = resp.json()["session_id"]
        assert sid, "session_id должен быть в ответе"
        # Должен быть валидным UUID
        uuid.UUID(sid)

    def test_chat_respects_provided_session_id(self, client):
        custom_sid = "my-custom-session-123"
        with patch("app.api.v1.chat.chat", AsyncMock(return_value=self._mock_chat_result())):
            resp = client.post("/api/v1/chat", json={
                "question":   "тест",
                "session_id": custom_sid,
            })
        assert resp.json()["session_id"] == custom_sid

    def test_chat_clarification_response(self, client):
        """Уточняющий вопрос возвращается с needs_clarification=true."""
        clarify_result = {
            "answer":              "Какой бюджет? До 20 000 / выше",
            "sources":             [],
            "intent":              "products",
            "needs_clarification": True,
            "clarification_question": "Какой бюджет? До 20 000 / выше",
        }
        with patch("app.api.v1.chat.chat", AsyncMock(return_value=clarify_result)):
            resp = client.post("/api/v1/chat", json={"question": "посоветуй телефон"})
        data = resp.json()
        assert data["needs_clarification"] is True
        assert data["clarification_question"] is not None
        assert "бюджет" in data["answer"].lower() or "20 000" in data["answer"]

    def test_chat_stream_returns_event_stream(self, client):
        """POST /api/v1/chat с stream=true возвращает text/event-stream."""
        async def fake_stream(q, sid):
            yield json.dumps({"type": "meta", "intent": "products",
                               "sources": [], "needs_clarification": False,
                               "clarification_question": None}) + "\n"
            yield json.dumps({"type": "chunk", "text": "Ответ"}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        with patch("app.api.v1.chat.chat_stream", fake_stream):
            resp = client.post("/api/v1/chat", json={
                "question": "тест",
                "stream":   True,
            })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [l for l in resp.text.strip().splitlines() if l.strip()]
        events = [json.loads(l) for l in lines]
        types = [e["type"] for e in events]
        print(f"\n  Stream events: {types}")
        assert "meta" in types
        assert "chunk" in types
        assert "done" in types

    def test_chat_stream_endpoint(self, client):
        """POST /api/v1/chat/stream — отдельный streaming endpoint."""
        async def fake_stream(q, sid):
            yield json.dumps({"type": "meta", "intent": "info",
                               "sources": [], "needs_clarification": False,
                               "clarification_question": None}) + "\n"
            yield json.dumps({"type": "chunk", "text": "Доставка 2 дня"}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        with patch("app.api.v1.chat.chat_stream", fake_stream):
            resp = client.post("/api/v1/chat/stream", json={"question": "доставка"})

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_chat_history_endpoint(self, client):
        """GET /api/v1/chat/{session_id}/history возвращает историю."""
        from app.services.history import add_messages, clear_session

        sid = "test-history-api-session"
        clear_session(sid)
        add_messages(sid, "Вопрос", "Ответ")

        resp = client.get(f"/api/v1/chat/{sid}/history")
        assert resp.status_code == 200
        data = resp.json()
        print(f"\n  history response: {data}")
        assert data["session_id"] == sid
        assert "messages" in data
        assert "count" in data
        assert data["count"] == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Вопрос"

        clear_session(sid)

    def test_chat_history_empty_session(self, client):
        """Несуществующая сессия → пустая история, не ошибка."""
        resp = client.get("/api/v1/chat/nonexistent-session-xyz/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["messages"] == []

    def test_clear_chat_session(self, client):
        from app.services.history import add_messages, get_history, clear_session

        sid = "test-clear-api"
        add_messages(sid, "Q", "A")
        assert len(get_history(sid)) == 2

        resp = client.delete(f"/api/v1/chat/{sid}")
        assert resp.status_code == 200
        assert resp.json()["cleared"] == sid
        assert get_history(sid) == []


# ── Settings API ──────────────────────────────────────────────────────────────

class TestSettingsApi:

    def test_get_llm_settings(self, client):
        resp = client.get("/api/v1/settings/llm")
        assert resp.status_code == 200
        data = resp.json()
        print(f"\n  /settings/llm: {json.dumps(data, ensure_ascii=False, indent=2)}")
        assert "active_provider" in data
        assert "active_model" in data
        assert "providers" in data
        assert "source" in data

    def test_get_llm_settings_shows_lm_studio(self, client):
        """После подмены conftest'ом active_provider должен быть openai (LM Studio)."""
        resp = client.get("/api/v1/settings/llm")
        data = resp.json()
        print(f"\n  active_provider: {data['active_provider']}, model: {data['active_model']}")
        assert data["active_provider"] == "openai"
        assert data["active_model"] == "google/gemma-4-12b"

    def test_upsert_provider_credentials(self, client):
        resp = client.post("/api/v1/settings/llm/provider/openai", json={
            "api_key":  "test-key-for-tests",
            "base_url": "http://192.168.56.1:1234/v1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True
        assert data["provider"] == "openai"

    def test_upsert_provider_requires_at_least_one_field(self, client):
        resp = client.post("/api/v1/settings/llm/provider/openai", json={})
        assert resp.status_code == 400

    def test_set_provider_model(self, client):
        resp = client.post("/api/v1/settings/llm/provider/openai/model", json={
            "model": "google/gemma-4-12b"
        })
        assert resp.status_code == 200
        assert resp.json()["model"] == "google/gemma-4-12b"

    def test_switch_active_provider(self, client):
        # openai уже настроен через conftest
        resp = client.post("/api/v1/settings/llm/active", json={
            "provider": "openai",
            "model":    "google/gemma-4-12b",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_provider"] == "openai"
        assert data["active_model"] == "google/gemma-4-12b"

    def test_get_models_list(self, client):
        with patch("app.api.v1.settings._fetch_models_openai_compatible",
                   return_value=["google/gemma-4-12b", "qwen3-9b"]):
            resp = client.get("/api/v1/settings/llm/models/openai")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        print(f"\n  Модели OpenAI: {data['models']}")

    def test_reset_llm_settings(self, client):
        """DELETE /settings/llm сбрасывает настройки — но мы восстановим их в conftest."""
        # Этот тест пропускаем — он сотрёт наши LM Studio настройки для остальных тестов
        # Запускать последним если нужно
        pytest.skip("Пропускаем чтобы не сбросить LM Studio настройки для других тестов")
