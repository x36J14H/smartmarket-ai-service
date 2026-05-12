"""
Routed RAG: сначала классифицируем намерение, затем ищем только в нужных коллекциях.

Сценарии по intent:
- products → поиск только в коллекции products + фильтр остатков
- info     → поиск в faq + navigation одновременно (объединяем результаты)
- multi    → поиск по всем коллекциям (смешанный вопрос или неопределённое намерение)

Tool-вызовы (живые данные из 1С):
- catalog → вопрос об ассортименте/категориях → GET /categories
"""
import re
from app.db.qdrant import search, search_all
from app.ml_models.llm import ask, analyze_query, is_catalog_question
from app.services.history import get_history, add_messages
from app.services.availability import fetch_availability
from app.services.catalog import get_catalog_answer, get_browse_answer
from app.core.config import settings


def _format_chunk(payload: dict) -> str:
    """Форматирует payload хита в текстовый чанк для LLM-контекста."""
    source_id = payload.get("source_id")
    text = payload.get("text", "")
    collection = payload.get("collection", "")

    if collection == "products" and source_id:
        lines = [
            f"product_id: {source_id}",
            f"ссылка на товар: /product/{source_id}",
        ]
        # Актуальные данные из 1С (обогащаются в chat() после запроса к 1С)
        price    = payload.get("actual_price")
        in_stock = payload.get("actual_stock")
        if price is not None:
            lines.append(f"цена: {price:,.0f} руб.".replace(",", " "))
        if in_stock is not None:
            lines.append(f"в наличии: {in_stock} шт.")
        lines.append(text)
        return "\n".join(lines)

    if collection == "navigation":
        url = payload.get("url", "")
        if url:
            return f"url: {url}\n{text}"

    return text


def _collect_valid_urls(hits: list) -> set[str]:
    """Собирает все валидные url из payload хитов."""
    urls = set()
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
    return urls


def _strip_hallucinated_links(text: str, valid_urls: set[str]) -> str:
    """
    Убирает markdown-ссылки с невалидными url.
    [текст](url) → текст, если url не в valid_urls.
    """
    def replace(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        return f"[{label}]({url})" if url in valid_urls else label

    return re.sub(r"\[([^\]]+)\]\(([^)]*)\)", replace, text)


async def chat(question: str, session_id: str) -> dict:
    """
    Intent-routed RAG:
    1. Классифицируем вопрос → intent
    2. Ищем в нужных коллекциях
    3. Для products/multi — фильтруем по остаткам
    4. LLM формирует ответ
    """
    history = get_history(session_id)

    # Tool: вопрос об ассортименте/категориях — отвечаем детерминированно, без LLM
    if is_catalog_question(question):
        answer = await get_catalog_answer()
        if answer:
            add_messages(session_id, question, answer)
            return {"answer": answer, "sources": [], "intent": "catalog"}
        # 1С недоступна — падаем в обычный RAG

    # Один LLM-вызов: intent + оптимизированный поисковый запрос
    analysis = analyze_query(question, history)
    intent = analysis["intent"]
    search_query = analysis["search_query"]

    # catalog_browse: пользователь хочет посмотреть категорию в целом → редирект на каталог
    if intent == "catalog_browse":
        answer = await get_browse_answer(search_query)
        if answer:
            add_messages(session_id, question, answer)
            return {"answer": answer, "sources": [], "intent": "catalog_browse"}
        # 1С недоступна — ищем в Qdrant как обычно
        intent = "products"

    # Шаг 2: поиск в зависимости от intent
    if intent == "multi":
        hits = search_all(search_query)

    elif intent == "info":
        # faq + navigation — ищем в обеих коллекциях, объединяем и сортируем по score
        k = settings.top_k
        faq_hits = search(search_query, collection="faq", top_k=k)
        nav_hits = search(search_query, collection="navigation", top_k=k)
        for hit in faq_hits:
            if hit.payload is not None:
                hit.payload["collection"] = "faq"
        for hit in nav_hits:
            if hit.payload is not None:
                hit.payload["collection"] = "navigation"
        hits = sorted(faq_hits + nav_hits, key=lambda h: h.score, reverse=True)[:k]

    else:  # products
        # top_k * 2 — при поиске в одной коллекции нет конкуренции с другими
        hits = search(search_query, collection="products", top_k=settings.top_k * 2)
        for hit in hits:
            if hit.payload is not None:
                hit.payload["collection"] = "products"

    # Шаг 3: фильтр остатков для товаров + обогащение актуальными ценами из 1С
    if intent in ("products", "multi"):
        product_ids = [
            hit.payload.get("source_id")
            for hit in hits
            if hit.payload and hit.payload.get("collection") == "products"
        ]
        if product_ids:
            availability = await fetch_availability(product_ids)

            if availability:
                # 1С ответила — фильтруем недоступные и обогащаем payload
                hits = [
                    hit for hit in hits
                    if hit.payload and (
                        hit.payload.get("collection") != "products"
                        or hit.payload.get("source_id") in availability
                    )
                ]
                # Записываем актуальные цену и остаток прямо в payload хита
                for hit in hits:
                    if not hit.payload:
                        continue
                    sid = hit.payload.get("source_id")
                    if sid and sid in availability:
                        hit.payload["actual_price"] = availability[sid]["price"]
                        hit.payload["actual_stock"] = availability[sid]["inStock"]
            else:
                # 1С недоступна — показываем все найденные товары без фильтрации
                pass

    # Шаг 4: формируем контекст
    context_chunks = [
        _format_chunk(hit.payload)
        for hit in hits if hit.payload
    ]

    sources = [
        {
            "collection": hit.payload.get("collection", intent),
            "score": round(hit.score, 3),
            "text": hit.payload.get("text", "")[:120],
            "product_id": hit.payload.get("source_id"),
            "url": (
                f"/product/{hit.payload.get('source_id')}"
                if hit.payload.get("collection") == "products" and hit.payload.get("source_id")
                else hit.payload.get("url")
            ),
        }
        for hit in hits
    ]

    if not context_chunks:
        answer = (
            "К сожалению, я не нашёл информации по вашему вопросу. "
            "Пожалуйста, свяжитесь с нашей поддержкой."
        )
        add_messages(session_id, question, answer)
        return {"answer": answer, "sources": [], "intent": intent}

    # Шаг 5: LLM отвечает
    valid_urls = _collect_valid_urls(hits)
    answer = ask(question, context_chunks, history=history)
    answer = _strip_hallucinated_links(answer, valid_urls)
    add_messages(session_id, question, answer)

    return {"answer": answer, "sources": sources, "intent": intent}
