"""
Проверка актуальных цен и остатков товаров через 1С.

Используется перед формированием контекста для LLM — чтобы бот не предлагал
товары которых нет в наличии или у которых не задана цена.
"""

import logging
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

# Максимум UUID за один запрос (ограничение 1С)
_BATCH_SIZE = 200


async def filter_available_ids(product_ids: list[str]) -> set[str]:
    """
    Запросить 1С и вернуть множество UUID товаров, которые есть в наличии
    (inStock > 0 и price > 0).

    Если ONEC_BASE_URL не задан или 1С недоступна — возвращает все переданные ID
    (graceful degradation: лучше показать возможно недоступный товар, чем ничего).
    """
    if not settings.onec_base_url:
        return set(product_ids)

    if not product_ids:
        return set()

    available: set[str] = set()

    # Батчами по 200 UUID
    for i in range(0, len(product_ids), _BATCH_SIZE):
        batch = product_ids[i : i + _BATCH_SIZE]
        ids_param = ",".join(batch)
        url = f"{settings.onec_base_url.rstrip('/')}/catalog/availability"

        try:
            auth = None
            if settings.onec_user:
                auth = (settings.onec_user, settings.onec_password)

            async with httpx.AsyncClient(timeout=5.0, auth=auth) as client:
                resp = await client.get(url, params={"ids": ids_param})
                resp.raise_for_status()
                data = resp.json()

            for item in data:
                if item.get("inStock", 0) > 0 and item.get("price", 0) > 0:
                    available.add(item["id"])

        except Exception as exc:
            # 1С недоступна — не ломаем чат, возвращаем батч как есть
            logger.warning("availability check failed (batch %d): %s", i // _BATCH_SIZE, exc)
            available.update(batch)

    return available
