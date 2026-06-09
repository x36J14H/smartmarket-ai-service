"""
Юнит-тесты логики бота — без реальных LLM и 1С вызовов.
Всё мокируется, тесты быстрые.

Проверяем:
- _format_product_chunk, _format_chunk, _format_1c_product
- _format_promotion, _format_order
- _collect_valid_urls
- _strip_hallucinated_links
- _extract_order_number
- _enrich_with_availability (с моком fetch_availability)
- _build_response для каждого intent
- chat() — полный pipeline с моками

pytest tests/test_bot_unit.py -v
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ── Вспомогательные объекты ───────────────────────────────────────────────────

def make_hit(collection: str, source_id: str = None, url: str = None,
             text: str = "Тестовый товар", score: float = 0.8,
             actual_price: float = None, actual_stock: int = None) -> MagicMock:
    """Создаёт fake ScoredPoint."""
    hit = MagicMock()
    hit.score = score
    hit.payload = {"collection": collection, "text": text}
    if source_id:
        hit.payload["source_id"] = source_id
    if url:
        hit.payload["url"] = url
    if actual_price is not None:
        hit.payload["actual_price"] = actual_price
    if actual_stock is not None:
        hit.payload["actual_stock"] = actual_stock
    return hit


# ── Форматирование чанков ─────────────────────────────────────────────────────

class TestFormatChunks:

    def test_format_product_chunk_full(self):
        from app.services.bot import _format_product_chunk
        payload = {
            "source_id": "abc-123",
            "actual_price": 29990.0,
            "actual_stock": 5,
            "text": "Наушники Sony WH-1000XM5",
        }
        result = _format_product_chunk(payload)
        print(f"\n  product chunk:\n{result}")
        assert "product_id: abc-123" in result
        assert "/product/abc-123" in result
        assert "29 990" in result
        assert "5 шт." in result
        assert "Наушники Sony" in result

    def test_format_product_chunk_no_price(self):
        from app.services.bot import _format_product_chunk
        payload = {"source_id": "xyz-999", "text": "Товар без цены"}
        result = _format_product_chunk(payload)
        assert "product_id: xyz-999" in result
        assert "цена" not in result
        assert "в наличии" not in result

    def test_format_product_chunk_no_source_id(self):
        from app.services.bot import _format_product_chunk
        payload = {"text": "Товар без UUID"}
        result = _format_product_chunk(payload)
        assert "product_id" not in result
        assert "Товар без UUID" in result

    def test_format_chunk_navigation(self):
        from app.services.bot import _format_chunk
        payload = {
            "collection": "navigation",
            "url": "/catalog/smartphones",
            "text": "Каталог смартфонов",
        }
        result = _format_chunk(payload)
        print(f"\n  nav chunk: {result}")
        assert "url: /catalog/smartphones" in result
        assert "Каталог смартфонов" in result

    def test_format_chunk_faq(self):
        from app.services.bot import _format_chunk
        payload = {"collection": "faq", "text": "Возврат в течение 14 дней"}
        result = _format_chunk(payload)
        assert result == "Возврат в течение 14 дней"

    def test_format_1c_product_full(self):
        from app.services.bot import _format_1c_product
        item = {
            "id": "onec-uuid-1",
            "name": "iPhone 15 Pro",
            "price": 119990.0,
            "inStock": 3,
            "brand": "Apple",
        }
        result = _format_1c_product(item)
        print(f"\n  1c product chunk:\n{result}")
        assert "product_id: onec-uuid-1" in result
        assert "/product/onec-uuid-1" in result
        assert "iPhone 15 Pro" in result
        assert "119 990" in result
        assert "3 шт." in result
        assert "Apple" in result

    def test_format_1c_product_zero_price(self):
        from app.services.bot import _format_1c_product
        item = {"id": "uuid-0", "name": "Товар", "price": 0, "inStock": 0}
        result = _format_1c_product(item)
        # price=0 → не выводим
        assert "цена" not in result

    def test_format_promotion_full(self):
        from app.services.bot import _format_promotion
        promo = {
            "title": "Скидки на iPhone",
            "description": "До 15% на все модели",
            "discount_percent": 15,
            "until": "2026-06-30",
            "category_slug": "iphone",
        }
        result = _format_promotion(promo)
        print(f"\n  promotion chunk:\n{result}")
        assert "Скидки на iPhone" in result
        assert "15%" in result
        assert "2026-06-30" in result
        assert "/catalog/iphone" in result

    def test_format_order_full(self):
        from app.services.bot import _format_order
        order = {
            "number": "12345",
            "status": "В пути",
            "delivery": {
                "date": "2026-06-10",
                "address": "Москва, Ленина 1",
                "tracking_url": "https://track.example.com/12345",
            },
            "total": 89990.0,
            "items": [
                {"name": "Samsung Galaxy S24", "quantity": 1},
            ],
        }
        result = _format_order(order)
        print(f"\n  order chunk:\n{result}")
        assert "12345" in result
        assert "В пути" in result
        assert "2026-06-10" in result
        assert "Москва" in result
        assert "track.example.com" in result
        assert "89 990" in result
        assert "Samsung Galaxy S24" in result

    def test_format_order_minimal(self):
        from app.services.bot import _format_order
        order = {"number": "999", "status": "Новый"}
        result = _format_order(order)
        assert "999" in result
        assert "Новый" in result


# ── URL-валидация ─────────────────────────────────────────────────────────────

class TestUrlValidation:

    def test_collect_valid_urls_products(self):
        from app.services.bot import _collect_valid_urls
        hits = [
            make_hit("products", source_id="uuid-1"),
            make_hit("products", source_id="uuid-2"),
        ]
        urls = _collect_valid_urls(hits)
        assert "/product/uuid-1" in urls
        assert "/product/uuid-2" in urls

    def test_collect_valid_urls_navigation(self):
        from app.services.bot import _collect_valid_urls
        hits = [make_hit("navigation", url="/catalog/phones")]
        urls = _collect_valid_urls(hits)
        assert "/catalog/phones" in urls

    def test_collect_valid_urls_extra_ids(self):
        from app.services.bot import _collect_valid_urls
        urls = _collect_valid_urls([], extra_ids=["onec-uuid-1", "onec-uuid-2"])
        assert "/product/onec-uuid-1" in urls
        assert "/product/onec-uuid-2" in urls

    def test_collect_valid_urls_ignores_faq(self):
        from app.services.bot import _collect_valid_urls
        hits = [make_hit("faq", text="Текст FAQ")]
        urls = _collect_valid_urls(hits)
        assert len(urls) == 0

    def test_strip_valid_links_untouched(self):
        from app.services.bot import _strip_hallucinated_links
        text = "Вот [iPhone 15](/product/abc-123) за 79 990 руб."
        valid = {"/product/abc-123"}
        result = _strip_hallucinated_links(text, valid)
        assert "[iPhone 15](/product/abc-123)" in result

    def test_strip_hallucinated_links_removed(self):
        from app.services.bot import _strip_hallucinated_links
        text = "Вот [Выдуманный товар](/product/fake-uuid) и [Реальный](/product/real-uuid)"
        valid = {"/product/real-uuid"}
        result = _strip_hallucinated_links(text, valid)
        print(f"\n  После очистки: {result}")
        assert "/product/fake-uuid" not in result
        assert "Выдуманный товар" in result  # текст остался, ссылка убрана
        assert "[Реальный](/product/real-uuid)" in result

    def test_strip_external_links_removed(self):
        from app.services.bot import _strip_hallucinated_links
        text = "Смотри на [сайте](https://example.com/product/123)"
        valid = {"/product/abc"}
        result = _strip_hallucinated_links(text, valid)
        assert "https://example.com" not in result
        assert "сайте" in result

    def test_strip_empty_valid_set(self):
        from app.services.bot import _strip_hallucinated_links
        text = "[Товар](/product/uuid) — хороший"
        result = _strip_hallucinated_links(text, set())
        assert "/product/uuid" not in result
        assert "Товар" in result


# ── Извлечение номера заказа ──────────────────────────────────────────────────

class TestExtractOrderNumber:

    def test_number_in_question(self):
        from app.services.bot import _extract_order_number
        n = _extract_order_number("где мой заказ 12345?", [])
        assert n == "12345"

    def test_number_in_history(self):
        from app.services.bot import _extract_order_number
        history = [
            {"role": "user", "content": "мой заказ 98765 где?"},
            {"role": "assistant", "content": "Заказ 98765 обрабатывается"},
        ]
        n = _extract_order_number("а когда будет?", history)
        assert n == "98765"

    def test_short_number_ignored(self):
        from app.services.bot import _extract_order_number
        # Трёхзначные числа не считаем номерами заказов
        n = _extract_order_number("заказ 123", [])
        assert n is None

    def test_no_number_returns_none(self):
        from app.services.bot import _extract_order_number
        n = _extract_order_number("где мой заказ?", [])
        assert n is None

    def test_long_number_extracted(self):
        from app.services.bot import _extract_order_number
        n = _extract_order_number("заказ номер 2024050001", [])
        assert n == "2024050001"


# ── _enrich_with_availability ─────────────────────────────────────────────────

class TestEnrichWithAvailability:

    @pytest.mark.asyncio
    async def test_enrich_filters_out_of_stock(self):
        from app.services import bot as bot_module
        hits = [
            make_hit("products", source_id="in-stock"),
            make_hit("products", source_id="out-of-stock"),
        ]
        availability = {"in-stock": {"price": 10000.0, "inStock": 5}}

        with patch.object(bot_module, "fetch_availability", AsyncMock(return_value=availability)):
            result = await bot_module._enrich_with_availability(hits)

        ids = [h.payload["source_id"] for h in result]
        assert "in-stock" in ids
        assert "out-of-stock" not in ids

    @pytest.mark.asyncio
    async def test_enrich_adds_price_and_stock(self):
        from app.services import bot as bot_module
        hits = [make_hit("products", source_id="abc-1")]
        availability = {"abc-1": {"price": 59990.0, "inStock": 3}}

        with patch.object(bot_module, "fetch_availability", AsyncMock(return_value=availability)):
            result = await bot_module._enrich_with_availability(hits)

        assert result[0].payload["actual_price"] == 59990.0
        assert result[0].payload["actual_stock"] == 3

    @pytest.mark.asyncio
    async def test_enrich_graceful_if_1c_down(self):
        """Если 1С недоступна (пустой dict) — все hits возвращаются как есть."""
        from app.services import bot as bot_module
        hits = [
            make_hit("products", source_id="x1"),
            make_hit("products", source_id="x2"),
        ]
        with patch.object(bot_module, "fetch_availability", AsyncMock(return_value={})):
            result = await bot_module._enrich_with_availability(hits)

        assert len(result) == 2  # ничего не отфильтровали

    @pytest.mark.asyncio
    async def test_enrich_non_product_hits_untouched(self):
        """FAQ и navigation хиты не трогаются."""
        from app.services import bot as bot_module
        hits = [
            make_hit("faq",        text="Возврат 14 дней"),
            make_hit("navigation", url="/delivery"),
        ]
        with patch.object(bot_module, "fetch_availability", AsyncMock(return_value={})):
            result = await bot_module._enrich_with_availability(hits)

        assert len(result) == 2


# ── _build_response — каждый intent ──────────────────────────────────────────

class TestBuildResponse:
    """
    Проверяем что _build_response правильно маршрутизирует по intent-у.
    LLM и 1С полностью мокируются.
    """

    def _analysis(self, intent: str, search_query: str = "запрос",
                  filters: dict = None, clarify: bool = False, clarify_q: str = None):
        return {
            "intent": intent,
            "search_query": search_query,
            "filters": filters or {},
            "needs_clarification": clarify,
            "clarification_question": clarify_q,
        }

    @pytest.mark.asyncio
    async def test_clarification_returned_immediately(self):
        """needs_clarification=True → ответ без поиска."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-clarify-1"
        clear_session(session)

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value=self._analysis(
            "products", clarify=True, clarify_q="Какой бюджет? До 20 000 / 20–50 000 / выше 50 000"
        ))):
            result = await bot_module._build_response("посоветуй телефон", session, [])

        assert "answer" in result
        assert result["needs_clarification"] is True
        assert "Какой бюджет" in result["answer"]
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_order_help_with_order_data(self):
        """intent=order_help + номер → запрашиваем 1С, формируем контекст."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-order-1"
        clear_session(session)

        order_data = {
            "number": "12345", "status": "В пути",
            "delivery": {"date": "2026-06-10", "address": "Москва", "tracking_url": ""},
            "total": 5000.0, "items": [],
        }
        with patch.object(bot_module, "analyze_query", AsyncMock(return_value=self._analysis("order_help"))), \
             patch.object(bot_module, "fetch_order_status", AsyncMock(return_value=order_data)):
            result = await bot_module._build_response("где заказ 12345?", session, [])

        assert "_context_chunks" in result
        chunks = result["_context_chunks"]
        assert any("12345" in c for c in chunks)
        assert any("В пути" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_order_help_not_found(self):
        """intent=order_help + 1С вернула None → контекст с сообщением."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-order-2"
        clear_session(session)

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value=self._analysis("order_help"))), \
             patch.object(bot_module, "fetch_order_status", AsyncMock(return_value=None)):
            result = await bot_module._build_response("где заказ 99999?", session, [])

        chunks = result["_context_chunks"]
        assert any("не найден" in c.lower() for c in chunks)

    @pytest.mark.asyncio
    async def test_promotions_with_1c_data(self):
        """intent=promotions + 1С вернула акции → контекст из акций."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-promos-1"
        clear_session(session)

        promos = [{
            "title": "Скидки на iPhone",
            "description": "До 15% на все модели",
            "discount_percent": 15,
            "until": "2026-06-30",
            "product_ids": ["promo-uuid-1"],
            "category_slug": "iphone",
        }]
        with patch.object(bot_module, "analyze_query", AsyncMock(return_value=self._analysis("promotions"))), \
             patch.object(bot_module, "fetch_promotions", AsyncMock(return_value=promos)):
            result = await bot_module._build_response("какие есть акции?", session, [])

        assert "_context_chunks" in result
        chunks = result["_context_chunks"]
        assert any("Скидки на iPhone" in c for c in chunks)
        assert "/product/promo-uuid-1" in result["_valid_urls"]

    @pytest.mark.asyncio
    async def test_promotions_fallback_to_faq(self):
        """intent=promotions + 1С недоступна → ищем в Qdrant faq."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-promos-2"
        clear_session(session)

        faq_hit = make_hit("faq", text="Акция: скидка 10% при оплате картой")

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value=self._analysis("promotions"))), \
             patch.object(bot_module, "fetch_promotions", AsyncMock(return_value=[])), \
             patch("app.services.bot.search", return_value=[faq_hit]):
            result = await bot_module._build_response("есть скидки?", session, [])

        assert "_context_chunks" in result
        assert any("скидка" in c.lower() or "акция" in c.lower() for c in result["_context_chunks"])

    @pytest.mark.asyncio
    async def test_info_searches_faq_and_nav(self):
        """intent=info → поиск по faq + navigation одновременно."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-info-1"
        clear_session(session)

        faq_hit = make_hit("faq", text="Возврат в 14 дней", score=0.9)
        nav_hit = make_hit("navigation", url="/returns", text="Оформление возвратов", score=0.85)

        search_results = {
            "faq": [faq_hit],
            "navigation": [nav_hit],
        }

        def mock_search(query, collection, top_k=None, **kwargs):
            return search_results.get(collection, [])

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value=self._analysis("info"))), \
             patch("app.services.bot.search", side_effect=mock_search):
            result = await bot_module._build_response("как вернуть товар?", session, [])

        assert "_context_chunks" in result
        chunks = result["_context_chunks"]
        assert any("14 дней" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_products_with_filters_calls_onec(self):
        """intent=products + фильтры → вызов fetch_catalog_search."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-products-filtered"
        clear_session(session)

        onec_items = [
            {"id": "onec-1", "name": "iPhone 15", "price": 79990.0, "inStock": 2},
        ]
        mock_search = AsyncMock(return_value=onec_items)
        qdrant_hits: list = []

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value=self._analysis(
            "products", filters={"brand": "Apple", "price_max": 100000}
        ))), \
             patch.object(bot_module, "fetch_catalog_search", mock_search), \
             patch("app.services.bot.search", return_value=qdrant_hits), \
             patch.object(bot_module, "fetch_availability", AsyncMock(return_value={})):
            result = await bot_module._build_response("iPhone до 100 тысяч", session, [])

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs.get("brand") == "Apple" or (
            call_kwargs.args and "Apple" in str(call_kwargs.args)
        )

        if "_context_chunks" in result:
            assert any("iPhone 15" in c for c in result["_context_chunks"])

    @pytest.mark.asyncio
    async def test_empty_context_returns_not_found(self):
        """Если контекст пустой → ответ 'не нашёл информации'."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-empty"
        clear_session(session)

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value=self._analysis("products"))), \
             patch("app.services.bot.search", return_value=[]), \
             patch.object(bot_module, "fetch_catalog_search", AsyncMock(return_value=[])), \
             patch.object(bot_module, "fetch_availability", AsyncMock(return_value={})):
            result = await bot_module._build_response("тест", session, [])

        assert "answer" in result
        assert "не нашёл" in result["answer"].lower() or "поддержк" in result["answer"].lower()


# ── chat() — полный pipeline ──────────────────────────────────────────────────

class TestChatPipeline:

    @pytest.mark.asyncio
    async def test_chat_returns_required_fields(self):
        """chat() всегда возвращает answer, sources, intent, session_id-совместимые поля."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-chat-fields"
        clear_session(session)

        hit = make_hit("products", source_id="uuid-test", text="Смартфон тест", actual_price=10000.0, actual_stock=1)

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value={
            "intent": "products", "search_query": "смартфон",
            "filters": {}, "needs_clarification": False, "clarification_question": None,
        })), \
             patch("app.services.bot.search", return_value=[hit]), \
             patch.object(bot_module, "fetch_catalog_search", AsyncMock(return_value=[])), \
             patch.object(bot_module, "fetch_availability", AsyncMock(return_value={
                 "uuid-test": {"price": 10000.0, "inStock": 1}
             })), \
             patch.object(bot_module, "ask", AsyncMock(return_value="Вот смартфон [Тест](/product/uuid-test)")):
            result = await bot_module.chat("смартфон", session)

        print(f"\n  chat() result keys: {list(result.keys())}")
        assert "answer" in result
        assert "sources" in result
        assert "intent" in result
        assert "needs_clarification" in result
        assert result["intent"] == "products"

    @pytest.mark.asyncio
    async def test_chat_saves_to_history(self):
        """chat() сохраняет вопрос и ответ в историю сессии."""
        from app.services import bot as bot_module
        from app.services.history import clear_session, get_history

        session = "test-history-save"
        clear_session(session)

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value={
            "intent": "info", "search_query": "доставка",
            "filters": {}, "needs_clarification": False, "clarification_question": None,
        })), \
             patch("app.services.bot.search", return_value=[
                 make_hit("faq", text="Доставка 1-3 дня")
             ]), \
             patch.object(bot_module, "ask", AsyncMock(return_value="Доставка занимает 1-3 рабочих дня")):
            await bot_module.chat("как долго доставляют?", session)

        history = get_history(session)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "как долго доставляют?"
        assert history[1]["role"] == "assistant"
        assert "1-3" in history[1]["content"]

    @pytest.mark.asyncio
    async def test_chat_strips_hallucinated_links(self):
        """chat() убирает выдуманные ссылки из ответа LLM."""
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-strip-links"
        clear_session(session)

        hit = make_hit("products", source_id="real-uuid", text="Реальный товар")

        # LLM придумала ссылку /product/fake-uuid которой нет в результатах
        fake_answer = (
            "[Реальный товар](/product/real-uuid) и "
            "[Выдуманный товар](/product/fake-uuid)"
        )

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value={
            "intent": "products", "search_query": "товар",
            "filters": {}, "needs_clarification": False, "clarification_question": None,
        })), \
             patch("app.services.bot.search", return_value=[hit]), \
             patch.object(bot_module, "fetch_catalog_search", AsyncMock(return_value=[])), \
             patch.object(bot_module, "fetch_availability", AsyncMock(return_value={
                 "real-uuid": {"price": 1000.0, "inStock": 1}
             })), \
             patch.object(bot_module, "ask", AsyncMock(return_value=fake_answer)):
            result = await bot_module.chat("товар", session)

        print(f"\n  После strip: {result['answer']}")
        assert "/product/real-uuid" in result["answer"]
        assert "/product/fake-uuid" not in result["answer"]
        assert "Выдуманный товар" in result["answer"]  # текст остался

    @pytest.mark.asyncio
    async def test_chat_stream_yields_meta_and_done(self):
        """chat_stream() всегда выдаёт meta-событие и done в конце."""
        import json
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-stream-1"
        clear_session(session)

        hit = make_hit("products", source_id="s-uuid", text="Стриминг товар")

        async def fake_stream(*a, **kw):
            yield "Вот "
            yield "ответ"

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value={
            "intent": "products", "search_query": "товар",
            "filters": {}, "needs_clarification": False, "clarification_question": None,
        })), \
             patch("app.services.bot.search", return_value=[hit]), \
             patch.object(bot_module, "fetch_catalog_search", AsyncMock(return_value=[])), \
             patch.object(bot_module, "fetch_availability", AsyncMock(return_value={
                 "s-uuid": {"price": 5000.0, "inStock": 2}
             })), \
             patch.object(bot_module, "ask_stream", AsyncMock(return_value=fake_stream())):
            stream = await bot_module.chat_stream("товар", session)

        events = []
        async for line in stream:
            events.append(json.loads(line.strip()))

        types = [e["type"] for e in events]
        print(f"\n  Stream events: {types}")
        assert "meta" in types, "Нет meta-события"
        assert "done" in types, "Нет done-события"
        assert "chunk" in types, "Нет chunk-событий"

        meta = next(e for e in events if e["type"] == "meta")
        assert "intent" in meta
        assert "sources" in meta

    @pytest.mark.asyncio
    async def test_chat_stream_clarification_single_chunk(self):
        """При уточняющем вопросе stream тоже возвращает meta + chunk + done."""
        import json
        from app.services import bot as bot_module
        from app.services.history import clear_session

        session = "test-stream-clarify"
        clear_session(session)

        with patch.object(bot_module, "analyze_query", AsyncMock(return_value={
            "intent": "products", "search_query": "телефон",
            "filters": {}, "needs_clarification": True,
            "clarification_question": "Какой бюджет? До 20 000 / выше",
        })):
            stream = await bot_module.chat_stream("посоветуй телефон", session)

        events = []
        async for line in stream:
            events.append(json.loads(line.strip()))

        types = [e["type"] for e in events]
        assert "meta" in types
        assert "chunk" in types
        assert "done" in types

        chunk_text = "".join(e["text"] for e in events if e["type"] == "chunk")
        assert "бюджет" in chunk_text.lower() or "20 000" in chunk_text
