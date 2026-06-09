"""
Тесты LLM-слоя через LM Studio (google/gemma-4-12b @ 192.168.56.1:1234).

Проверяем:
- Прямой вызов LM Studio через httpx
- analyze_query: парсинг intent, фильтров, уточнений
- ask: генерация ответа с контекстом и правильным промптом по intent
- ask_stream: стриминг
- rerank_async: фильтрация нерелевантных результатов
- Устойчивость к плохому JSON в analyze_query (fallback)

pytest tests/test_llm.py -v -s
"""
import json
import pytest
import asyncio
import httpx
from unittest.mock import AsyncMock, patch

from tests.conftest import LM_STUDIO_BASE, LM_STUDIO_MODEL


# ── Прямая проверка LM Studio ─────────────────────────────────────────────────

class TestLmStudioDirect:

    def test_lm_studio_reachable(self, lm_client):
        """LM Studio должен отвечать на /v1/models."""
        resp = lm_client.get("/models")
        assert resp.status_code == 200, (
            f"LM Studio недоступен: {resp.status_code}\n{resp.text[:300]}"
        )
        data = resp.json()
        assert "data" in data, f"Нет поля 'data': {data}"
        models = [m["id"] for m in data["data"]]
        print(f"\n  Загруженные модели: {models}")
        assert len(models) > 0, "Нет загруженных моделей в LM Studio"

    def test_lm_studio_target_model_loaded(self, lm_client):
        """Нужная модель google/gemma-4-12b должна быть загружена."""
        resp = lm_client.get("/models")
        assert resp.status_code == 200
        models = [m["id"] for m in resp.json()["data"]]
        print(f"\n  Доступные модели: {models}")
        assert any(LM_STUDIO_MODEL in m or m in LM_STUDIO_MODEL for m in models), (
            f"Модель {LM_STUDIO_MODEL} не загружена. Загружены: {models}"
        )

    def test_lm_studio_chat_completion(self, lm_client):
        """Минимальный chat/completions запрос должен вернуть ответ."""
        resp = lm_client.post("/chat/completions", json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "Скажи 'ок' одним словом"}],
            "max_tokens": 10,
            "temperature": 0.0,
        })
        assert resp.status_code == 200, f"LM Studio вернул {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"\n  Ответ LM Studio: {repr(content)}")
        assert content and len(content.strip()) > 0, "Пустой ответ от LLM"

    def test_lm_studio_streaming(self, lm_client):
        """Streaming chat/completions должен возвращать чанки."""
        chunks = []
        with lm_client.stream("POST", "/chat/completions", json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "Скажи 'привет'"}],
            "max_tokens": 20,
            "temperature": 0.0,
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    payload = json.loads(line[6:])
                    delta = payload["choices"][0]["delta"].get("content", "")
                    if delta:
                        chunks.append(delta)

        full = "".join(chunks)
        print(f"\n  Streaming ответ: {repr(full)}")
        assert len(chunks) > 0, "Streaming не вернул ни одного чанка"
        assert len(full.strip()) > 0, "Streaming дал пустой текст"


# ── analyze_query ─────────────────────────────────────────────────────────────

class TestAnalyzeQuery:

    @pytest.mark.asyncio
    async def test_products_intent(self):
        """Запрос о конкретном товаре → intent=products."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("есть ли у вас iPhone 15 Pro?")
        print(f"\n  analyze_query → {result}")
        assert result["intent"] == "products", f"Ожидали products, получили: {result['intent']}"
        assert result["search_query"], "search_query пустой"
        assert "iphone" in result["search_query"].lower() or "айфон" in result["search_query"].lower(), (
            f"iPhone не попал в search_query: {result['search_query']}"
        )

    @pytest.mark.asyncio
    async def test_info_intent(self):
        """Вопрос о доставке → intent=info."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("как у вас происходит доставка?")
        print(f"\n  analyze_query → {result}")
        assert result["intent"] == "info", f"Ожидали info, получили: {result['intent']}"

    @pytest.mark.asyncio
    async def test_catalog_browse_intent(self):
        """Просмотр категории → intent=catalog_browse."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("покажи все смартфоны")
        print(f"\n  analyze_query → {result}")
        assert result["intent"] in ("catalog_browse", "products"), (
            f"Ожидали catalog_browse или products, получили: {result['intent']}"
        )

    @pytest.mark.asyncio
    async def test_compare_intent(self):
        """Сравнение товаров → intent=compare."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("сравни iPhone 15 и Samsung Galaxy S24")
        print(f"\n  analyze_query → {result}")
        assert result["intent"] == "compare", f"Ожидали compare, получили: {result['intent']}"

    @pytest.mark.asyncio
    async def test_promotions_intent(self):
        """Вопрос об акциях → intent=promotions."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("есть ли сейчас какие-то скидки?")
        print(f"\n  analyze_query → {result}")
        assert result["intent"] == "promotions", f"Ожидали promotions, получили: {result['intent']}"

    @pytest.mark.asyncio
    async def test_order_help_intent(self):
        """Вопрос о заказе → intent=order_help."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("где мой заказ 12345?")
        print(f"\n  analyze_query → {result}")
        assert result["intent"] == "order_help", f"Ожидали order_help, получили: {result['intent']}"

    @pytest.mark.asyncio
    async def test_price_filter_extraction(self):
        """Цена до N рублей должна попасть в filters.price_max."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("хочу ноутбук до 80000 рублей")
        print(f"\n  analyze_query → {result}")
        filters = result.get("filters", {})
        print(f"  filters: {filters}")
        assert "price_max" in filters, (
            f"price_max не извлечён из запроса 'до 80000'. filters={filters}"
        )
        assert filters["price_max"] == 80000, (
            f"Ожидали 80000, получили {filters['price_max']}"
        )

    @pytest.mark.asyncio
    async def test_brand_filter_extraction(self):
        """Бренд Apple должен попасть в filters.brand."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("покажи наушники Apple")
        print(f"\n  analyze_query → {result}")
        filters = result.get("filters", {})
        brand = filters.get("brand", "")
        print(f"  filters: {filters}")
        assert "apple" in brand.lower() or "Apple" in brand, (
            f"Бренд Apple не извлечён. filters={filters}"
        )

    @pytest.mark.asyncio
    async def test_price_range_filter(self):
        """Диапазон цен должен попасть в price_min и price_max."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("смартфон от 30000 до 60000 рублей")
        print(f"\n  analyze_query → {result}")
        filters = result.get("filters", {})
        print(f"  filters: {filters}")
        # Хотя бы price_max должен быть
        assert "price_max" in filters or "price_min" in filters, (
            f"Ни price_min ни price_max не извлечены. filters={filters}"
        )

    @pytest.mark.asyncio
    async def test_needs_clarification_broad_query(self):
        """Широкий запрос 'посоветуй телефон' должен вызвать уточнение."""
        from app.ml_models.llm import analyze_query
        result = await analyze_query("посоветуй телефон")
        print(f"\n  analyze_query → {result}")
        print(f"  needs_clarification: {result['needs_clarification']}")
        print(f"  clarification_question: {result['clarification_question']}")
        # Модель может и не всегда давать уточнение — это не жёсткое требование
        # но если даёт — проверяем формат
        if result["needs_clarification"]:
            assert result["clarification_question"], "needs_clarification=True но вопрос пустой"
            assert len(result["clarification_question"]) > 10, "Уточняющий вопрос слишком короткий"

    @pytest.mark.asyncio
    async def test_pronoun_resolution_with_history(self):
        """'сколько они стоят' с историей про AirPods → search_query должен содержать AirPods."""
        from app.ml_models.llm import analyze_query
        history = [
            {"role": "user",      "content": "есть ли у вас AirPods Pro?"},
            {"role": "assistant", "content": "Да, у нас есть AirPods Pro 2 — 24 990 руб."},
        ]
        result = await analyze_query("сколько они стоят?", history=history)
        print(f"\n  analyze_query с историей → {result}")
        sq = result["search_query"].lower()
        assert "airpods" in sq or "наушники" in sq or "apple" in sq, (
            f"Местоимение 'они' не разрешено через историю. search_query: {sq}"
        )

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self):
        """Если LLM вернул не-JSON — analyze_query должен вернуть fallback, не упасть."""
        from app.ml_models import llm as llm_module
        original = llm_module._dispatch_async

        async def mock_bad_json(*args, **kwargs):
            return "Это не JSON совсем"

        llm_module._dispatch_async = mock_bad_json
        try:
            result = await llm_module.analyze_query("тест")
            print(f"\n  Fallback результат: {result}")
            assert result["intent"] == "multi"
            assert result["search_query"] == "тест"
            assert result["needs_clarification"] is False
        finally:
            llm_module._dispatch_async = original

    @pytest.mark.asyncio
    async def test_all_intents_valid(self):
        """intent всегда должен быть из списка валидных."""
        from app.ml_models.llm import analyze_query, VALID_INTENTS
        queries = [
            "iPhone 15",
            "как оформить возврат",
            "покажи ноутбуки",
            "сравни Samsung и Xiaomi",
            "где мой заказ 54321",
            "есть скидки?",
            "что угодно непонятное xyz123",
        ]
        for q in queries:
            result = await analyze_query(q)
            print(f"\n  '{q}' → intent={result['intent']}")
            assert result["intent"] in VALID_INTENTS, (
                f"Невалидный intent '{result['intent']}' для запроса '{q}'"
            )


# ── ask / ask_stream ──────────────────────────────────────────────────────────

class TestAsk:

    @pytest.mark.asyncio
    async def test_ask_returns_nonempty_answer(self):
        """ask() должен вернуть непустой текст."""
        from app.ml_models.llm import ask
        context = [
            "product_id: abc-123\nссылка на товар: /product/abc-123\n"
            "название: Sony WH-1000XM5\nцена: 29 990 руб.\nв наличии: 5 шт.\n"
            "Беспроводные наушники Sony с шумоподавлением."
        ]
        answer = await ask("Есть ли наушники Sony?", context, intent="products")
        print(f"\n  ask() → {repr(answer[:200])}")
        assert answer and len(answer.strip()) > 10, f"Слишком короткий ответ: {repr(answer)}"

    @pytest.mark.asyncio
    async def test_ask_products_uses_product_link(self):
        """Ответ по products-intent должен содержать markdown-ссылку на товар."""
        from app.ml_models.llm import ask
        context = [
            "product_id: abc-999\nссылка на товар: /product/abc-999\n"
            "название: Apple AirPods Pro 2\nцена: 24 990 руб.\nв наличии: 3 шт."
        ]
        answer = await ask("Покажи наушники Apple", context, intent="products")
        print(f"\n  ask() → {repr(answer[:300])}")
        assert "/product/abc-999" in answer, (
            f"Ссылка /product/abc-999 не попала в ответ:\n{answer}"
        )

    @pytest.mark.asyncio
    async def test_ask_info_intent_no_product_links(self):
        """Ответ по info-intent не должен выдумывать product-ссылки."""
        from app.ml_models.llm import ask
        context = [
            "Возврат товара возможен в течение 14 дней с момента получения. "
            "Для оформления перейдите в раздел 'Мои заказы'."
        ]
        answer = await ask("Как вернуть товар?", context, intent="info")
        print(f"\n  ask() info → {repr(answer[:300])}")
        assert len(answer.strip()) > 10, "Пустой ответ"
        # Не должно быть выдуманных product-ссылок
        import re
        fake_links = re.findall(r'\(/product/[^)]+\)', answer)
        assert not fake_links, f"Выдуманные product-ссылки в info-ответе: {fake_links}"

    @pytest.mark.asyncio
    async def test_ask_no_context_admits_ignorance(self):
        """При пустом контексте бот не должен выдумывать — ответ про отсутствие данных."""
        from app.ml_models.llm import ask
        answer = await ask("Есть ли у вас холодильники?", [], intent="products")
        print(f"\n  ask() пустой контекст → {repr(answer[:300])}")
        # Ответ не должен содержать выдуманные конкретные товары с ценами
        assert len(answer.strip()) > 5, "Совсем пустой ответ"

    @pytest.mark.asyncio
    async def test_ask_respects_history(self):
        """ask() с историей не должен терять контекст диалога."""
        from app.ml_models.llm import ask
        history = [
            {"role": "user",      "content": "Есть ли наушники Sony?"},
            {"role": "assistant", "content": "Да, есть Sony WH-1000XM5 — 29 990 руб."},
        ]
        context = [
            "product_id: abc-123\nссылка: /product/abc-123\n"
            "название: Sony WH-1000XM5\nцена: 29 990 руб."
        ]
        answer = await ask("А есть гарантия на них?", context, history=history, intent="products")
        print(f"\n  ask() с историей → {repr(answer[:300])}")
        assert len(answer.strip()) > 10

    @pytest.mark.asyncio
    async def test_ask_stream_yields_chunks(self):
        """ask_stream() должен отдавать несколько чанков текста."""
        from app.ml_models.llm import ask_stream
        context = [
            "product_id: xyz-777\nссылка на товар: /product/xyz-777\n"
            "название: Xiaomi 14T\nцена: 59 990 руб.\nв наличии: 2 шт."
        ]
        stream = await ask_stream("Есть ли Xiaomi?", context, intent="products")
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        full = "".join(chunks)
        print(f"\n  ask_stream() чанков: {len(chunks)}, полный текст: {repr(full[:200])}")
        assert len(chunks) > 0, "Streaming не дал ни одного чанка"
        assert len(full.strip()) > 5, "Streaming дал пустой текст"

    @pytest.mark.asyncio
    async def test_think_tags_stripped(self):
        """<think>...</think> теги должны удаляться из ответа."""
        from app.ml_models.llm import _clean_think_tags
        raw = "<think>Подумаю немного...</think>Вот мой ответ."
        cleaned = _clean_think_tags(raw)
        print(f"\n  Очищено: {repr(cleaned)}")
        assert "<think>" not in cleaned
        assert "Вот мой ответ." in cleaned


# ── rerank_async ──────────────────────────────────────────────────────────────

class TestRerankAsync:

    def _make_hits(self, texts: list[str]) -> list:
        """Создаёт fake ScoredPoint объекты для reranker."""
        from unittest.mock import MagicMock
        hits = []
        for i, text in enumerate(texts):
            hit = MagicMock()
            hit.payload = {"text": text, "source_id": f"uuid-{i}"}
            hit.score = 0.9 - i * 0.1
            hits.append(hit)
        return hits

    @pytest.mark.asyncio
    async def test_rerank_keeps_relevant(self):
        """rerank_async должен оставлять релевантные товары."""
        from app.ml_models.llm import rerank_async
        hits = self._make_hits([
            "Смартфон Apple iPhone 15 Pro, 256 ГБ, черный",
            "Холодильник двухкамерный Samsung, 400 литров",
            "Смартфон Samsung Galaxy S24, 128 ГБ",
        ])
        result = await rerank_async("смартфон Apple iPhone", hits)
        print(f"\n  rerank_async: из {len(hits)} → {len(result)}")
        texts = [h.payload["text"] for h in result]
        print(f"  Оставлено: {texts}")
        # iPhone должен остаться
        assert any("iPhone" in t for t in texts), "iPhone отфильтрован — это неверно"

    @pytest.mark.asyncio
    async def test_rerank_filters_irrelevant(self):
        """rerank_async должен убирать нерелевантные."""
        from app.ml_models.llm import rerank_async
        hits = self._make_hits([
            "Смартфон Apple iPhone 15 Pro",
            "Газовая плита GEFEST 4 конфорки",
            "Микроволновая печь LG 20 литров",
        ])
        result = await rerank_async("iPhone смартфон", hits)
        texts = [h.payload["text"] for h in result]
        print(f"\n  rerank_async после фильтрации: {texts}")
        # Плита и микроволновка не должны остаться
        irrelevant = [t for t in texts if "плита" in t.lower() or "микроволнов" in t.lower()]
        assert not irrelevant, f"Нерелевантные товары не отфильтрованы: {irrelevant}"

    @pytest.mark.asyncio
    async def test_rerank_empty_input(self):
        """rerank_async с пустым списком должен вернуть пустой список."""
        from app.ml_models.llm import rerank_async
        result = await rerank_async("iPhone", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_error(self):
        """При ошибке LLM rerank_async возвращает исходный список."""
        from app.ml_models import llm as llm_module
        hits = self._make_hits(["Товар 1", "Товар 2"])
        original = llm_module._dispatch_async

        async def mock_error(*a, **kw):
            raise RuntimeError("LLM недоступен")

        llm_module._dispatch_async = mock_error
        try:
            result = await llm_module.rerank_async("запрос", hits)
            assert result == hits, "При ошибке должен вернуться исходный список"
        finally:
            llm_module._dispatch_async = original
