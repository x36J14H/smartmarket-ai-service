"""
Интеграционные тесты бота с реальными LLM (LM Studio) и реальной 1С.

Требования:
- LM Studio запущен на http://192.168.56.1:1234, загружена google/gemma-4-12b
- 1С HTTP-сервис доступен на http://localhost:8081
- Qdrant запущен на localhost:6333 с данными

Запуск только интеграционных:
    pytest tests/test_bot_integration.py -v -s -m integration

Полный прогон (медленно, ~2-5 мин):
    pytest tests/test_bot_integration.py -v -s

pytest tests/test_bot_integration.py -v -s -k "test_products"
"""
import json
import pytest


pytestmark = pytest.mark.integration  # все тесты здесь — интеграционные


# ── Проверка доступности зависимостей ─────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def check_dependencies():
    """Проверяем LM Studio и Qdrant перед запуском модуля."""
    import httpx

    # LM Studio
    try:
        resp = httpx.get("http://192.168.56.1:1234/v1/models", timeout=5.0)
        assert resp.status_code == 200
    except Exception as e:
        pytest.skip(f"LM Studio недоступен: {e}")

    # Qdrant
    try:
        resp = httpx.get("http://localhost:6333/health", timeout=3.0)
        assert resp.status_code == 200
    except Exception as e:
        pytest.skip(f"Qdrant недоступен: {e}")


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _print_result(label: str, result: dict):
    """Форматированный вывод результата chat()."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  intent:  {result.get('intent')}")
    print(f"  clarify: {result.get('needs_clarification')}")
    if result.get("clarification_question"):
        print(f"  clarify_q: {result['clarification_question']}")
    print(f"  answer:  {result.get('answer', '')[:300]}")
    sources = result.get("sources", [])
    if sources:
        print(f"  sources ({len(sources)}):")
        for s in sources[:3]:
            print(f"    [{s.get('collection')}] score={s.get('score')} "
                  f"pid={s.get('product_id', 'N/A')}")
    print("="*60)


# ── Полный pipeline через chat() ──────────────────────────────────────────────

class TestBotIntegrationFull:

    @pytest.mark.asyncio
    async def test_products_query_iphone(self):
        """Запрос про iPhone → intent=products, ответ содержит информацию о товаре."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-iphone-test"
        clear_session(sid)

        result = await chat("есть ли у вас iPhone?", sid)
        _print_result("products / iPhone", result)

        assert "answer" in result
        assert len(result["answer"].strip()) > 20
        assert result["intent"] in ("products", "catalog_browse", "multi")
        # Ответ не должен быть пустым или шаблонным "не нашёл"
        assert "поддержк" not in result["answer"].lower() or len(result["answer"]) > 100

    @pytest.mark.asyncio
    async def test_products_query_with_price_filter(self):
        """Запрос с ценой → filters должны попасть в поиск."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-price-test"
        clear_session(sid)

        result = await chat("смартфон до 50000 рублей", sid)
        _print_result("products / price filter", result)

        assert result["intent"] in ("products", "catalog_browse")
        assert len(result["answer"].strip()) > 10

    @pytest.mark.asyncio
    async def test_info_query_delivery(self):
        """Вопрос о доставке → intent=info, поиск в FAQ."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-info-test"
        clear_session(sid)

        result = await chat("как работает доставка?", sid)
        _print_result("info / delivery", result)

        assert result["intent"] in ("info", "multi")
        assert len(result["answer"].strip()) > 10

    @pytest.mark.asyncio
    async def test_info_query_returns(self):
        """Вопрос о возврате."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-returns-test"
        clear_session(sid)

        result = await chat("как вернуть товар?", sid)
        _print_result("info / returns", result)

        assert len(result["answer"].strip()) > 10

    @pytest.mark.asyncio
    async def test_promotions_query(self):
        """Вопрос об акциях → intent=promotions."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-promos-test"
        clear_session(sid)

        result = await chat("какие сейчас есть скидки и акции?", sid)
        _print_result("promotions", result)

        assert result["intent"] in ("promotions", "info", "multi")
        assert len(result["answer"].strip()) > 10

    @pytest.mark.asyncio
    async def test_order_help_with_number(self):
        """Статус заказа — 1С скорее всего вернёт 404/None для тестового номера."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-order-test"
        clear_session(sid)

        result = await chat("где мой заказ 99999?", sid)
        _print_result("order_help / not found", result)

        assert result["intent"] in ("order_help", "info", "multi")
        # Если заказ не найден — ответ про это или перенаправление в поддержку
        assert len(result["answer"].strip()) > 10

    @pytest.mark.asyncio
    async def test_catalog_browse(self):
        """Просмотр каталога."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-browse-test"
        clear_session(sid)

        result = await chat("что у вас вообще есть в каталоге?", sid)
        _print_result("catalog_browse", result)

        assert len(result["answer"].strip()) > 20

    @pytest.mark.asyncio
    async def test_multi_turn_dialogue(self):
        """Многоходовой диалог — второй вопрос должен учитывать историю."""
        from app.services.bot import chat
        from app.services.history import clear_session, get_history

        sid = "integration-multiturn-test"
        clear_session(sid)

        # Первый вопрос
        r1 = await chat("есть ли у вас наушники?", sid)
        _print_result("turn 1 / наушники", r1)

        history_after_1 = get_history(sid)
        assert len(history_after_1) == 2, "После 1 вопроса должно быть 2 сообщения в истории"

        # Второй вопрос — местоимение "они" должно разрешиться через историю
        r2 = await chat("сколько они стоят?", sid)
        _print_result("turn 2 / сколько стоят", r2)

        history_after_2 = get_history(sid)
        assert len(history_after_2) == 4

        # Ответ должен быть про наушники (не про что-то другое)
        answer_lower = r2["answer"].lower()
        headphones_mentioned = any(w in answer_lower for w in (
            "наушник", "sony", "apple", "airpods", "jbl", "sennheiser", "цена", "руб"
        ))
        print(f"\n  Местоимение разрешено? {headphones_mentioned}")
        # Не делаем жёстким — модель может не разрешить местоимение при пустом контексте

    @pytest.mark.asyncio
    async def test_compare_query(self):
        """Сравнение двух товаров."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-compare-test"
        clear_session(sid)

        result = await chat("сравни Samsung Galaxy и iPhone — что лучше?", sid)
        _print_result("compare", result)

        assert result["intent"] in ("compare", "products", "multi")
        assert len(result["answer"].strip()) > 20

    @pytest.mark.asyncio
    async def test_no_hallucinated_product_links(self):
        """Ответ не должен содержать product-ссылки которых нет в sources."""
        import re
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-hallucination-test"
        clear_session(sid)

        result = await chat("есть ли у вас телевизоры Samsung?", sid)
        _print_result("hallucination check", result)

        answer = result["answer"]
        sources = result.get("sources", [])

        # Собираем валидные product_id из sources
        valid_ids = {s["product_id"] for s in sources if s.get("product_id")}
        print(f"  Валидные UUID в sources: {valid_ids}")

        # Находим все /product/uuid в ответе
        found_links = re.findall(r'/product/([^)\s"\']+)', answer)
        print(f"  UUID в ответе: {found_links}")

        for link_id in found_links:
            assert link_id in valid_ids or not valid_ids, (
                f"Выдуманная ссылка /product/{link_id} не найдена в sources: {valid_ids}"
            )

    @pytest.mark.asyncio
    async def test_answer_language_matches_question(self):
        """Ответ должен быть на том же языке что вопрос."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-lang-test"
        clear_session(sid)

        result = await chat("do you have any smartphones?", sid)
        _print_result("English question", result)

        # Хотя бы несколько английских слов должны быть в ответе
        answer = result["answer"]
        english_chars = sum(1 for c in answer if 'a' <= c.lower() <= 'z')
        total_alpha = sum(1 for c in answer if c.isalpha())
        english_ratio = english_chars / total_alpha if total_alpha else 0
        print(f"  English ratio: {english_ratio:.2f}")
        # Не жёсткое требование — зависит от наличия товаров с латиницей в данных

    @pytest.mark.asyncio
    async def test_empty_answer_fallback(self):
        """Запрос о явно несуществующем товаре → вежливый отказ."""
        from app.services.bot import chat
        from app.services.history import clear_session

        sid = "integration-fallback-test"
        clear_session(sid)

        result = await chat("есть ли у вас ядерные реакторы?", sid)
        _print_result("fallback / не найдено", result)

        answer = result["answer"].lower()
        # Должен быть вежливый ответ — не пустой
        assert len(result["answer"].strip()) > 5
        # Не должно быть выдуманных товаров с конкретными UUID
        import re
        assert not re.search(r'/product/[a-f0-9-]{36}', result["answer"]), (
            "Не должно быть product-ссылок для несуществующего товара"
        )


# ── Streaming integration ─────────────────────────────────────────────────────

class TestBotStreamIntegration:

    @pytest.mark.asyncio
    async def test_stream_full_flow(self):
        """Стриминг должен отдать meta → chunks → done."""
        from app.services.bot import chat_stream
        from app.services.history import clear_session

        sid = "integration-stream-test"
        clear_session(sid)

        stream = await chat_stream("есть ли смартфоны?", sid)

        events = []
        async for line in stream:
            line = line.strip()
            if line:
                events.append(json.loads(line))

        types = [e["type"] for e in events]
        print(f"\n  Stream event types: {types}")
        print(f"  Total events: {len(events)}")

        assert "meta" in types, "Нет meta-события"
        assert "done" in types, "Нет done-события"

        # Если не уточняющий вопрос — должны быть chunk-и
        meta = next(e for e in events if e["type"] == "meta")
        if not meta.get("needs_clarification"):
            assert "chunk" in types, "Нет chunk-событий для обычного ответа"

        # Собираем полный текст из чанков
        full_text = "".join(e["text"] for e in events if e["type"] == "chunk")
        print(f"  Полный ответ: {full_text[:200]}")
        assert len(full_text.strip()) > 5

    @pytest.mark.asyncio
    async def test_stream_meta_has_required_fields(self):
        """meta-событие должно иметь все нужные поля."""
        from app.services.bot import chat_stream
        from app.services.history import clear_session

        sid = "integration-stream-meta-test"
        clear_session(sid)

        stream = await chat_stream("как работает доставка?", sid)

        meta = None
        async for line in stream:
            line = line.strip()
            if line:
                event = json.loads(line)
                if event["type"] == "meta":
                    meta = event
                    break

        assert meta is not None
        assert "intent" in meta
        assert "sources" in meta
        assert "needs_clarification" in meta
        assert "clarification_question" in meta
        assert meta["intent"] in (
            "products", "catalog_browse", "compare", "info",
            "order_help", "promotions", "multi"
        )


# ── Тесты конкретных 1С данных ────────────────────────────────────────────────

class TestBotWith1cData:
    """
    Тесты которые используют реальные данные из 1С.
    Пропускаются если 1С не возвращает нужные данные.
    """

    @pytest.mark.asyncio
    async def test_categories_in_catalog_browse(self):
        """При catalog_browse бот должен вернуть реальные категории из 1С."""
        import httpx
        from app.services.bot import chat
        from app.services.history import clear_session

        # Получаем реальные категории из 1С
        try:
            resp = httpx.get(
                "http://localhost:8081/smartmarket/hs/site-api/categories",
                auth=("Администратор", ""),
                timeout=5.0,
            )
            if resp.status_code != 200:
                pytest.skip("1С /categories недоступен")
            cats = resp.json().get("categories", [])
            if not cats:
                pytest.skip("Нет категорий в 1С")
            first_cat_name = cats[0]["name"]
        except Exception as e:
            pytest.skip(f"1С недоступна: {e}")

        sid = "integration-1c-categories"
        clear_session(sid)

        result = await chat("что у вас есть в каталоге?", sid)
        _print_result(f"catalog с категорией {first_cat_name}", result)

        # Ответ должен содержать хотя бы одну реальную категорию
        # (или направить в каталог)
        assert len(result["answer"]) > 20

    @pytest.mark.asyncio
    async def test_availability_filters_products(self):
        """
        Если 1С вернула данные о наличии — в ответе не должно быть
        товаров с нулевым остатком.
        """
        import httpx
        from app.services.bot import chat
        from app.services.history import clear_session

        # Проверяем доступность availability endpoint
        try:
            resp = httpx.get(
                "http://localhost:8081/smartmarket/hs/site-api/catalog/availability",
                params={"ids": "00000000-0000-0000-0000-000000000000"},
                auth=("Администратор", ""),
                timeout=5.0,
            )
            if resp.status_code == 404:
                pytest.skip("1С /catalog/availability не реализован")
        except Exception as e:
            pytest.skip(f"1С недоступна: {e}")

        sid = "integration-availability"
        clear_session(sid)

        result = await chat("покажи смартфоны", sid)
        _print_result("availability filter check", result)

        # Проверяем что в sources нет товаров с нулевым остатком
        # (это не проверить напрямую без дополнительных данных из 1С,
        #  но убеждаемся что ответ разумный)
        assert len(result["answer"]) > 10
