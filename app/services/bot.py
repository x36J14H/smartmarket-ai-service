from app.db.qdrant import search_all
from app.ml_models.llm import ask
from app.services.history import get_history, add_messages


def chat(question: str, session_id: str) -> dict:
    """RAG: история → поиск по всем коллекциям → LLM → сохранить историю."""
    history = get_history(session_id)
    hits = search_all(question)

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
        return {"answer": answer, "sources": []}

    answer = ask(question, context_chunks, history=history)
    add_messages(session_id, question, answer)

    return {"answer": answer, "sources": sources}
