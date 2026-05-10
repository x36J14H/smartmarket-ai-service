from app.db.qdrant import search_all
from app.ml_models.llm import ask
from app.services.history import get_history, add_messages
from app.services.availability import filter_available_ids


def _format_chunk(payload: dict) -> str:
    """Форматирует payload хита в текстовый чанк для LLM-контекста."""
    source_id = payload.get("source_id")
    text = payload.get("text", "")
    if source_id and payload.get("collection") == "products":
        return f"product_id: {source_id}\nurl: /products/{source_id}\n{text}"
    return text


async def chat(question: str, session_id: str) -> dict:
    """RAG: история → поиск по всем коллекциям → фильтр остатков → LLM → сохранить историю."""
    history = get_history(session_id)
    hits = search_all(question)

    # Фильтруем товары без остатков — для FAQ/navigation пропускаем проверку
    product_ids = [
        hit.payload.get("source_id")
        for hit in hits
        if hit.payload and hit.payload.get("collection") == "products"
    ]
    if product_ids:
        available = await filter_available_ids(product_ids)
        hits = [
            hit for hit in hits
            if hit.payload and (
                hit.payload.get("collection") != "products"
                or hit.payload.get("source_id") in available
            )
        ]

    context_chunks = [
        _format_chunk(hit.payload)
        for hit in hits if hit.payload
    ]
    sources = [
        {
            "collection": hit.payload.get("collection", ""),
            "score": round(hit.score, 3),
            "text": hit.payload.get("text", "")[:120],
            "product_id": hit.payload.get("source_id"),
        }
        for hit in hits
    ]

    if not context_chunks:
        answer = (
            "К сожалению, я не нашёл информации по вашему вопросу. "
            "Пожалуйста, свяжитесь с нашей поддержкой."
        )
        add_messages(session_id, question, answer)
        return {"answer": answer, "sources": []}

    answer = ask(question, context_chunks, history=history)
    add_messages(session_id, question, answer)

    return {"answer": answer, "sources": sources}
