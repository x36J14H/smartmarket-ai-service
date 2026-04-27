from app.db.qdrant import search_all
from app.ml_models.llm import ask, hypothetical_answer
from app.services.history import get_history, add_messages


def chat_hyde(question: str, session_id: str) -> dict:
    """HyDE RAG: вопрос → гипотетический ответ → embed → Qdrant → LLM → ответ."""
    history = get_history(session_id)

    # Шаг 1: LLM генерирует гипотетический документ для поиска
    hyde_query = hypothetical_answer(question)

    # Шаг 2: ищем по гипотетическому документу, а не по сырому вопросу
    hits = search_all(hyde_query)

    context_chunks = [
        (f"product_id: {hit.payload.get('source_id')}\n" if hit.payload.get('source_id') else "")
        + hit.payload.get("text", "")
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
        return {"answer": answer, "sources": [], "hyde_query": hyde_query}

    # Шаг 3: LLM отвечает на основе найденного контекста
    answer = ask(question, context_chunks, history=history)
    add_messages(session_id, question, answer)

    return {
        "answer": answer,
        "sources": sources,
        "hyde_query": hyde_query,  # для отладки — видно что сгенерировала LLM
    }
