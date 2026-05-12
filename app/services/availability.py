"""
Проверка актуальных цен и остатков товаров через 1С.

Используется перед формированием контекста для LLM — чтобы бот не предлагал
товары которых нет в наличии или у которых не задана цена, а также чтобы
передавать актуальные цены и остатки прямо в контекст.
"""

import logging
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

# Максимум UUID за один запрос (ограничение 1С)
_BATCH_SIZE = 200


async def fetch_availability(product_ids: list[str]) -> dict[str, dict]:
    """
    Запросить 1С и вернуть словарь {uuid: {"price": float, "inStock": int}}
    только для товаров, которые есть в наличии (inStock > 0 и price > 0).

    Если ONEC_BASE_URL не задан или 1С недоступна — возвращает пустой словарь
    (вызывающий код сам решает как деградировать).
    """
    if not settings.onec_base_url or not product_ids:
        return {}

    result: dict[str, dict] = {}

    for i in range(0, len(product_ids), _BATCH_SIZE):
        batch = product_ids[i : i + _BATCH_SIZE]
        ids_param = ",".join(batch)
        url = f"{settings.onec_base_url.rstrip('/')}/catalog/availability"

        try:
            auth = (settings.onec_user, settings.onec_password) if settings.onec_user else None

            async with httpx.AsyncClient(timeout=5.0, auth=auth) as client:
                resp = await client.get(url, params={"ids": ids_param})
                resp.raise_for_status()
                data = resp.json()

            for item in data:
                price    = item.get("price", 0) or 0
                in_stock = item.get("inStock", 0) or 0
                if in_stock > 0 and price > 0:
                    result[item["id"]] = {"price": price, "inStock": in_stock}

        except Exception as exc:
            logger.warning("availability check failed (batch %d): %s", i // _BATCH_SIZE, exc)

    return result


async def filter_available_ids(product_ids: list[str]) -> set[str]:
    """
    Обратная совместимость: вернуть только множество доступных UUID.

    Если 1С недоступна — возвращает все переданные ID
    (graceful degradation: лучше показать возможно недоступный товар, чем ничего).
    """
    if not settings.onec_base_url:
        return set(product_ids)

    data = await fetch_availability(product_ids)

    if not data:
        # 1С недоступна (fetch вернул пустой dict из-за ошибки) — деградируем
        return set(product_ids)

    return set(data.keys())
