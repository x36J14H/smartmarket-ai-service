"""
LLM-reranking для поиска товаров.

Используется только в /products/search — чат не нуждается в reranking,
там LLM сама разберётся с контекстом.
"""
import logging
from app.ml_models.llm import _dispatch

logger = logging.getLogger(__name__)

RERANK_PROMPT = """Ты — помощник для поиска товаров в интернет-магазине электроники.
Тебе дан запрос пользователя и список найденных товаров с номерами.
Верни ТОЛЬКО номера товаров, которые реально подходят под запрос, через запятую.
Пример ответа: 1,3,5
Если ни один не подходит — верни пустую строку.
Не добавляй никаких пояснений."""


def rerank(query: str, hits: list) -> list:
    """
    Отфильтровать нерелевантные результаты через LLM.

    hits — список ScoredPoint из Qdrant.
    Возвращает отфильтрованный список в том же порядке.
    """
    if not hits:
        return hits

    items = "\n".join(
        f"{i + 1}. {h.payload.get('text', '')[:120]}"
        for i, h in enumerate(hits)
        if h.payload
    )

    messages = [
        {"role": "system", "content": RERANK_PROMPT},
        {"role": "user", "content": f"Запрос: {query}\n\nТовары:\n{items}"},
    ]

    try:
        response = _dispatch(messages, temperature=0.0, max_tokens=64)
        indices = {
            int(x.strip()) - 1
            for x in response.split(",")
            if x.strip().isdigit()
        }
        reranked = [h for i, h in enumerate(hits) if i in indices]
        # Если LLM вернула пустой список — fallback на исходные результаты
        return reranked if reranked else hits
    except Exception as exc:
        logger.warning("reranking failed, returning original hits: %s", exc)
        return hits
