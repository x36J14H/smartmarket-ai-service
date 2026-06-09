"""
LLM-reranking для поиска товаров.

Используется только в /products/search — чат использует встроенный reranking
через async pipeline в bot.py.
"""
import logging
from app.ml_models.llm import rerank_async

logger = logging.getLogger(__name__)


async def rerank(query: str, hits: list) -> list:
    """
    Async LLM-reranking — фильтрует нерелевантные результаты поиска.
    hits — список ScoredPoint из Qdrant.
    Возвращает отфильтрованный список в том же порядке.
    """
    return await rerank_async(query, hits)
