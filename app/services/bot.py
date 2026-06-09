"""
Intent-routed RAG агент с поддержкой:
- Расширенных intent-ов (products, catalog_browse, compare, info, order_help, promotions, multi)
- Уточняющих вопросов когда запрос слишком широкий
- Структурированных фильтров из запроса (цена, бренд, категория)
- Параллельных запросов к Qdrant и 1С
- Streaming-ответов
- Специализированных промптов по intent-у
- Новых 1С-эндпоинтов (product detail, compare, orders, promotions)
"""
import re
import asyncio
import logging
from typing import AsyncIterator

from app.db.qdrant import search, search_all
from app.ml_models.llm import ask, ask_stream, analyze_query
from app.services.history import get_history, add_messages
from app.services.onec_client import (
    fetch_availability,
    fetch_catalog_search,
    fetch_compare,
    fetch_order_status,
    fetch_promotions,
)
from app.services.catalog import get_catalog_answer, get_browse_answer
from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Форматирование чанков для контекста ───────────────────────────────────────

def _format_product_chunk(payload: dict) -> str:
    """Форматирует payload товара в текстовый чанк для LLM."""
    lines = []
    source_id = payload.get("source_id")
    if source_id:
        lines.append(f"product_id: {source_id}")
        lines.append(f"ссылка на товар: /product/{source_id}")

    price = payload.get("actual_price")
    in_stock = payload.get("actual_stock")
    if price is not None:
        lines.append(f"цена: {price:,.0f} руб.".replace(",", " "))
    if in_stock is not None:
        lines.append(f"в наличии: {in_stock} шт.")

    text = payload.get("text", "")
    if text:
        lines.append(text)

    return "\n".join(lines)


def _format_chunk(payload: dict) -> str:
    """Форматирует payload хита в текстовый чанк в зависимости от коллекции."""
    collection = payload.get("collection", "")

    if collection == "products":
        return _format_product_chunk(payload)

    if collection == "navigation":
        url = payload.get("url", "")
        text = payload.get("text", "")
        return f"url: {url}\n{text}" if url else text

    return payload.get("text", "")


def _format_1c_product(item: dict) -> str:
    """Форматирует товар из 1С (catalog/search) в текстовый чанк."""
    lines = [
        f"product_id: {item['id']}",
        f"ссылка на товар: /product/{item['id']}",
        f"название: {item.get('name', '')}",
    ]
    price = item.get("price")
    in_stock = item.get("inStock")
    if price:
        lines.append(f"цена: {price:,.0f} руб.".replace(",", " "))
    if in_stock is not None:
        lines.append(f"в наличии: {in_stock} шт.")
    brand = item.get("brand")
    if brand:
        lines.append(f"бренд: {brand}")
    return "\n".join(lines)


def _format_promotion(promo: dict) -> str:
    """Форматирует акцию в текстовый чанк."""
    lines = [
        f"акция: {promo.get('title', '')}",
        f"описание: {promo.get('description', '')}",
    ]
    discount = promo.get("discount_percent")
    if discount:
        lines.append(f"скидка: {discount}%")
    until = promo.get("until")
    if until:
        lines.append(f"действует до: {until}")
    category_slug = promo.get("category_slug")
    if category_slug:
        lines.append(f"url: /catalog/{category_slug}")
    return "\n".join(lines)


def _format_order(order: dict) -> str:
    """Форматирует статус заказа в текстовый чанк."""
    lines = [
        f"номер заказа: {order.get('number', '')}",
        f"статус: {order.get('status', '')}",
    ]
    delivery = order.get("delivery", {})
    if delivery:
        if delivery.get("date"):
            lines.append(f"дата доставки: {delivery['date']}")
        if delivery.get("address"):
            lines.append(f"адрес доставки: {delivery['address']}")
        if delivery.get("tracking_url"):
            lines.append(f"ссылка отслеживания: {delivery['tracking_url']}")
    total = order.get("total")
    if total:
        lines.append(f"сумма заказа: {total:,.0f} руб.".replace(",", " "))
    items = order.get("items", [])
    if items:
        items_text = ", ".join(
            f"{i.get('name', '')} ×{i.get('quantity', 1)}"
            for i in items[:5]
        )
        lines.append(f"товары: {items_text}")
    return "\n".join(lines)


# ── Валидация ссылок ──────────────────────────────────────────────────────────

def _collect_valid_urls(hits: list, extra_ids: list[str] | None = None) -> set[str]:
    """Собирает все валидные URL из результатов поиска."""
    urls: set[str] = set()

    for hit in hits:
        if not hit.payload:
            continue
        collection = hit.payload.get("collection", "")
        source_id = hit.payload.get("source_id")
        if collection == "products" and source_id:
            urls.add(f"/product/{source_id}")
        elif collection == "navigation":
            url = hit.payload.get("url")
            if url:
                urls.add(url)

    # Дополнительные ID из 1С-запросов (catalog/search и т.д.)
    if extra_ids:
        for pid in extra_ids:
            urls.add(f"/product/{pid}")

    return urls


def _strip_hallucinated_links(text: str, valid_urls: set[str]) -> str:
    """
    Убирает markdown-ссылки с невалидными URL.
    [текст](url) → текст, если url не в valid_urls.
    """
    def replace(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        return f"[{label}]({url})" if url in valid_urls else label

    return re.sub(r"\[([^\]]+)\]\(([^)]*)\)", replace, text)


# ── Извлечение номера заказа из вопроса ──────────────────────────────────────

def _extract_order_number(question: str, history: list[dict]) -> str | None:
    """Ищет номер заказа в тексте — просто последовательность цифр."""
    # Ищем в текущем вопросе
    match = re.search(r"\b(\d{4,})\b", question)
    if match:
        return match.group(1)

    # Ищем в истории (последние 4 сообщения)
    for msg in reversed(history[-4:]):
        match = re.search(r"\b(\d{4,})\b", msg.get("content", ""))
        if match:
            return match.group(1)

    return None


# ── Обогащение hits данными из 1С ────────────────────────────────────────────

async def _enrich_with_availability(hits: list) -> list:
    """
    Запрашивает 1С по наличию и ценам, фильтрует недоступные товары,
    обогащает payload актуальными данными.
    """
    product_ids = [
        hit.payload.get("source_id")
        for hit in hits
        if hit.payload and hit.payload.get("collection") == "products"
    ]

    if not product_ids:
        return hits

    availability = await fetch_availability(product_ids)

    if not availability:
        # 1С недоступна — показываем все товары без фильтрации
        return hits

    # Фильтруем недоступные
    filtered = [
        hit for hit in hits
        if not hit.payload
           or hit.payload.get("collection") != "products"
           or hit.payload.get("source_id") in availability
    ]

    # Обогащаем payload актуальными ценой и остатком
    for hit in filtered:
        if not hit.payload:
            continue
        sid = hit.payload.get("source_id")
        if sid and sid in availability:
            hit.payload["actual_price"] = availability[sid]["price"]
            hit.payload["actual_stock"] = availability[sid]["inStock"]

    return filtered


# ── Основной pipeline ─────────────────────────────────────────────────────────

async def _build_response(
    question: str,
    session_id: str,
    history: list[dict],
    stream: bool = False,
) -> dict:
    """
    Строит полный ответ агента (без streaming-генерации).
    Возвращает dict с answer, sources, intent, needs_clarification.
    """

    # ── Шаг 1: анализ запроса ─────────────────────────────────────────────────
    analysis = await analyze_query(question, history)
    intent = analysis["intent"]
    search_query = analysis["search_query"]
    filters = analysis["filters"]
    needs_clarification = analysis["needs_clarification"]
    clarification_question = analysis["clarification_question"]

    # ── Уточняющий вопрос ─────────────────────────────────────────────────────
    if needs_clarification and clarification_question:
        add_messages(session_id, question, clarification_question)
        return {
            "answer":                 clarification_question,
            "sources":                [],
            "intent":                 intent,
            "needs_clarification":    True,
            "clarification_question": clarification_question,
        }

    # ── Шаг 2: intent-специфичный сбор контекста ─────────────────────────────

    context_chunks: list[str] = []
    hits: list = []
    extra_product_ids: list[str] = []

    # ── Помощь с заказом ──────────────────────────────────────────────────────
    if intent == "order_help":
        order_number = _extract_order_number(question, history)
        if order_number:
            order_data = await fetch_order_status(order_number)
            if order_data:
                context_chunks = [_format_order(order_data)]
            else:
                context_chunks = [
                    f"Заказ №{order_number} не найден. "
                    "Возможно номер введён неверно или заказ оформлен под другим аккаунтом."
                ]
        else:
            # Нет номера — ищем в FAQ как обычно
            hits = search(search_query, collection="faq", top_k=settings.top_k)
            for hit in hits:
                if hit.payload is not None:
                    hit.payload["collection"] = "faq"
            context_chunks = [_format_chunk(hit.payload) for hit in hits if hit.payload]

    # ── Акции ─────────────────────────────────────────────────────────────────
    elif intent == "promotions":
        promotions = await fetch_promotions()
        if promotions:
            context_chunks = [_format_promotion(p) for p in promotions]
            extra_product_ids = [
                pid
                for p in promotions
                for pid in p.get("product_ids", [])
            ]
        else:
            # 1С недоступна — ищем в FAQ
            hits = search(search_query, collection="faq", top_k=settings.top_k)
            for hit in hits:
                if hit.payload is not None:
                    hit.payload["collection"] = "faq"
            context_chunks = [_format_chunk(hit.payload) for hit in hits if hit.payload]

    # ── Информация о магазине ─────────────────────────────────────────────────
    elif intent == "info":
        k = settings.top_k
        faq_hits, nav_hits = await asyncio.gather(
            asyncio.to_thread(search, search_query, "faq",        k),
            asyncio.to_thread(search, search_query, "navigation", k),
        )
        for hit in faq_hits:
            if hit.payload is not None:
                hit.payload["collection"] = "faq"
        for hit in nav_hits:
            if hit.payload is not None:
                hit.payload["collection"] = "navigation"
        hits = sorted(faq_hits + nav_hits, key=lambda h: h.score, reverse=True)[:k]
        context_chunks = [_format_chunk(hit.payload) for hit in hits if hit.payload]

    # ── Просмотр каталога ─────────────────────────────────────────────────────
    elif intent == "catalog_browse":
        # Пробуем сначала поиск с фильтрами в 1С (если есть фильтры)
        onec_items: list[dict] = []
        if filters:
            onec_items = await fetch_catalog_search(
                category=filters.get("category"),
                brand=filters.get("brand"),
                price_min=filters.get("price_min"),
                price_max=filters.get("price_max"),
                limit=settings.top_k * 2,
            )

        if onec_items:
            context_chunks = [_format_1c_product(item) for item in onec_items]
            extra_product_ids = [item["id"] for item in onec_items]
        else:
            # Fallback: ссылка на категорию из каталога
            browse_answer = await get_browse_answer(search_query)
            if browse_answer:
                add_messages(session_id, question, browse_answer)
                return {
                    "answer":              browse_answer,
                    "sources":             [],
                    "intent":              intent,
                    "needs_clarification": False,
                    "clarification_question": None,
                }
            # 1С недоступна — ищем в Qdrant
            intent = "products"
            hits = search(search_query, collection="products", top_k=settings.top_k * 2)
            for hit in hits:
                if hit.payload is not None:
                    hit.payload["collection"] = "products"
            hits = await _enrich_with_availability(hits)
            context_chunks = [_format_chunk(hit.payload) for hit in hits if hit.payload]

    # ── Сравнение товаров ─────────────────────────────────────────────────────
    elif intent == "compare":
        # Ищем кандидатов в Qdrant параллельно с запросом к 1С
        qdrant_hits = await asyncio.to_thread(
            search, search_query, "products", settings.top_k * 2
        )
        for hit in qdrant_hits:
            if hit.payload is not None:
                hit.payload["collection"] = "products"

        # Пробуем получить данные для сравнения из 1С
        candidate_ids = [
            hit.payload["source_id"]
            for hit in qdrant_hits
            if hit.payload and hit.payload.get("source_id")
        ][:4]

        compare_data = None
        if len(candidate_ids) >= 2:
            compare_data = await fetch_compare(candidate_ids[:2])

        if compare_data:
            # Форматируем данные сравнения из 1С
            compare_chunks = []
            diff_fields = compare_data.get("diff_fields", [])
            if diff_fields:
                compare_chunks.append(f"Ключевые отличия: {', '.join(diff_fields)}")
            for product in compare_data.get("products", []):
                lines = [
                    f"product_id: {product['id']}",
                    f"ссылка на товар: /product/{product['id']}",
                    f"название: {product.get('name', '')}",
                ]
                chars = product.get("characteristics", {})
                for field in diff_fields:
                    val = chars.get(field)
                    if val:
                        lines.append(f"{field}: {val}")
                price = product.get("price")
                if price:
                    lines.append(f"цена: {price:,.0f} руб.".replace(",", " "))
                compare_chunks.append("\n".join(lines))
            context_chunks = compare_chunks
            extra_product_ids = candidate_ids
        else:
            # Fallback — используем Qdrant результаты
            hits = await _enrich_with_availability(qdrant_hits[:settings.top_k])
            context_chunks = [_format_chunk(hit.payload) for hit in hits if hit.payload]

    # ── Поиск товаров (основной путь) ─────────────────────────────────────────
    elif intent == "products":
        # Параллельно: Qdrant поиск + 1С поиск с фильтрами (если есть)
        qdrant_task = asyncio.to_thread(
            search, search_query, "products", settings.top_k * 2
        )
        async def _empty_list():
            return []

        onec_task = fetch_catalog_search(
            category=filters.get("category"),
            brand=filters.get("brand"),
            price_min=filters.get("price_min"),
            price_max=filters.get("price_max"),
            limit=settings.top_k,
        ) if filters else _empty_list()

        qdrant_hits, onec_items = await asyncio.gather(qdrant_task, onec_task)

        for hit in qdrant_hits:
            if hit.payload is not None:
                hit.payload["collection"] = "products"

        # Если 1С вернула результаты по фильтрам — используем их как приоритетный контекст
        if onec_items:
            extra_product_ids = [item["id"] for item in onec_items]
            onec_chunks = [_format_1c_product(item) for item in onec_items]

            # Дополняем результатами из Qdrant (могут содержать описания/характеристики)
            enriched_hits = await _enrich_with_availability(qdrant_hits[:settings.top_k])
            qdrant_chunks = [_format_chunk(hit.payload) for hit in enriched_hits if hit.payload]

            # 1С-результаты первыми — они точнее по фильтрам
            context_chunks = onec_chunks + qdrant_chunks
            hits = enriched_hits
        else:
            hits = await _enrich_with_availability(qdrant_hits)
            context_chunks = [_format_chunk(hit.payload) for hit in hits if hit.payload]

    # ── Смешанный / неопределённый запрос ────────────────────────────────────
    else:  # multi
        hits = await asyncio.to_thread(search_all, search_query)
        hits = await _enrich_with_availability(hits)
        context_chunks = [_format_chunk(hit.payload) for hit in hits if hit.payload]

    # ── Шаг 3: формируем sources для ответа ──────────────────────────────────
    sources = [
        {
            "collection": hit.payload.get("collection", intent),
            "score":      round(hit.score, 3),
            "text":       hit.payload.get("text", "")[:120],
            "product_id": hit.payload.get("source_id"),
            "url": (
                f"/product/{hit.payload.get('source_id')}"
                if hit.payload.get("collection") == "products" and hit.payload.get("source_id")
                else hit.payload.get("url")
            ),
        }
        for hit in hits
        if hit.payload
    ]

    if not context_chunks:
        answer = (
            "К сожалению, я не нашёл информации по вашему вопросу. "
            "Пожалуйста, свяжитесь с нашей поддержкой."
        )
        add_messages(session_id, question, answer)
        return {
            "answer":              answer,
            "sources":             [],
            "intent":              intent,
            "needs_clarification": False,
            "clarification_question": None,
        }

    return {
        "_context_chunks": context_chunks,
        "_valid_urls":     _collect_valid_urls(hits, extra_product_ids),
        "_sources":        sources,
        "_intent":         intent,
    }


# ── Публичные функции ─────────────────────────────────────────────────────────

async def chat(question: str, session_id: str) -> dict:
    """
    Основной entrypoint агента — возвращает полный ответ.
    """
    history = get_history(session_id)

    result = await _build_response(question, session_id, history)

    # Если ответ уже готов (уточнение, catalog, empty)
    if "answer" in result:
        return result

    context_chunks = result["_context_chunks"]
    valid_urls = result["_valid_urls"]
    sources = result["_sources"]
    intent = result["_intent"]

    answer = await ask(question, context_chunks, history=history, intent=intent)
    answer = _strip_hallucinated_links(answer, valid_urls)
    add_messages(session_id, question, answer)

    return {
        "answer":              answer,
        "sources":             sources,
        "intent":              intent,
        "needs_clarification": False,
        "clarification_question": None,
    }


async def chat_stream(question: str, session_id: str) -> AsyncIterator[str]:
    """
    Streaming entrypoint — возвращает AsyncIterator чанков текста.
    Сначала отдаёт метаданные (intent, sources) как первый JSON-чанк,
    затем стримит текст ответа.
    """
    import json

    history = get_history(session_id)

    result = await _build_response(question, session_id, history, stream=True)

    # Если ответ уже готов (уточнение, catalog, empty)
    if "answer" in result:
        # Отдаём как единый чанк
        async def _yield_ready():
            yield json.dumps({
                "type":                  "meta",
                "intent":                result.get("intent"),
                "sources":               result.get("sources", []),
                "needs_clarification":   result.get("needs_clarification", False),
                "clarification_question": result.get("clarification_question"),
            }) + "\n"
            yield json.dumps({"type": "chunk", "text": result["answer"]}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return _yield_ready()

    context_chunks = result["_context_chunks"]
    valid_urls = result["_valid_urls"]
    sources = result["_sources"]
    intent = result["_intent"]

    stream_iter = await ask_stream(question, context_chunks, history=history, intent=intent)

    async def _generate():
        # Сначала метаданные
        yield json.dumps({
            "type":    "meta",
            "intent":  intent,
            "sources": sources,
            "needs_clarification": False,
            "clarification_question": None,
        }) + "\n"

        # Стримим текст + собираем для post-processing
        full_text_parts = []
        async for chunk in stream_iter:
            full_text_parts.append(chunk)
            yield json.dumps({"type": "chunk", "text": chunk}) + "\n"

        # Пост-обработка полного текста
        full_text = "".join(full_text_parts)
        cleaned = _strip_hallucinated_links(full_text, valid_urls)

        # Если после очистки текст изменился — отправляем замену
        if cleaned != full_text:
            yield json.dumps({"type": "replace", "text": cleaned}) + "\n"

        add_messages(session_id, question, cleaned)
        yield json.dumps({"type": "done"}) + "\n"

    return _generate()
