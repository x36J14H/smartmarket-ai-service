"""
Единый HTTP-клиент для всех запросов к 1С.

Все эндпоинты 1С собраны здесь — меняешь URL один раз, работает везде.
Каждая функция деградирует gracefully: если 1С недоступна — возвращает None / пустой dict.
"""
import logging
import time
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Таймаут берётся из конфига (ONEC_TIMEOUT)
def _timeout() -> float:
    return settings.onec_timeout

# Максимум UUID за один запрос (ограничение 1С)
_AVAILABILITY_BATCH_SIZE = 200


def _base_url() -> str:
    return settings.onec_base_url.rstrip("/")


def _auth() -> tuple[str, str] | None:
    return (settings.onec_user, settings.onec_password) if settings.onec_user else None


def _is_configured() -> bool:
    return bool(settings.onec_base_url)


# ── Inline TTL-кэш (используется для каталога и акций) ───────────────────────

class _TtlCache:
    def __init__(self, ttl: float = 300.0):
        self._data: Any = None
        self._ts: float = 0.0
        self._ttl = ttl

    def get(self) -> Any | None:
        if self._data is not None and (time.monotonic() - self._ts) < self._ttl:
            return self._data
        return None

    def set(self, data: Any) -> None:
        self._data = data
        self._ts = time.monotonic()

    def invalidate(self) -> None:
        self._data = None
        self._ts = 0.0


_categories_cache = _TtlCache(ttl=300.0)   # 5 минут
_promotions_cache = _TtlCache(ttl=120.0)   # 2 минуты


# ── Availability — цены и остатки ─────────────────────────────────────────────

async def fetch_availability(product_ids: list[str]) -> dict[str, dict]:
    """
    GET /catalog/availability?ids=uuid1,uuid2,...

    Возвращает {uuid: {"price": float, "inStock": int}}
    только для товаров где inStock > 0 и price > 0.

    При недоступности 1С — возвращает пустой dict.
    """
    if not _is_configured() or not product_ids:
        return {}

    result: dict[str, dict] = {}

    for i in range(0, len(product_ids), _AVAILABILITY_BATCH_SIZE):
        batch = product_ids[i: i + _AVAILABILITY_BATCH_SIZE]
        ids_param = ",".join(batch)
        url = f"{_base_url()}/catalog/availability"

        try:
            async with httpx.AsyncClient(timeout=_timeout(), auth=_auth()) as client:
                resp = await client.get(url, params={"ids": ids_param})
                resp.raise_for_status()
                data = resp.json()

            for item in data:
                price = item.get("price", 0) or 0
                in_stock = item.get("inStock", 0) or 0
                if in_stock > 0 and price > 0:
                    result[item["id"]] = {"price": price, "inStock": in_stock}

        except Exception as exc:
            logger.warning("1С availability failed (batch %d): %s", i // _AVAILABILITY_BATCH_SIZE, exc)

    return result


async def filter_available_ids(product_ids: list[str]) -> set[str]:
    """
    Вернуть только множество доступных UUID.
    Если 1С недоступна — возвращает все ID (graceful degradation).
    """
    if not _is_configured():
        return set(product_ids)

    data = await fetch_availability(product_ids)

    if not data:
        # 1С недоступна — деградируем, показываем всё
        return set(product_ids)

    return set(data.keys())


# ── Categories — дерево категорий ─────────────────────────────────────────────

async def fetch_categories() -> dict | None:
    """
    GET /categories

    Возвращает дерево категорий с кэшированием (TTL 5 мин).
    None если 1С недоступна.
    """
    if not _is_configured():
        return None

    cached = _categories_cache.get()
    if cached is not None:
        return cached

    url = f"{_base_url()}/categories"
    try:
        async with httpx.AsyncClient(timeout=_timeout(), auth=_auth()) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        _categories_cache.set(data)
        return data

    except Exception as exc:
        logger.warning("1С categories fetch failed: %s", exc)
        return None


# ── Product detail — полные данные о товаре ───────────────────────────────────

async def fetch_product(product_id: str) -> dict | None:
    """
    GET /catalog/product/{uuid}

    Возвращает полные данные о товаре: характеристики, описание, изображения.
    None если 1С недоступна или товар не найден.

    Ожидаемый формат ответа от 1С:
    {
        "id": "uuid",
        "name": "Название",
        "price": 49990.0,
        "inStock": 5,
        "brand": "Apple",
        "category": "Смартфоны",
        "description": "...",
        "characteristics": {"Память": "256 ГБ", "Цвет": "Черный"},
        "images": ["/files/image1.jpg"]
    }
    """
    if not _is_configured():
        return None

    url = f"{_base_url()}/catalog/product/{product_id}"
    try:
        async with httpx.AsyncClient(timeout=_timeout(), auth=_auth()) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

    except Exception as exc:
        logger.warning("1С product detail failed (id=%s): %s", product_id, exc)
        return None


# ── Catalog search — поиск с фильтрами ───────────────────────────────────────

async def fetch_catalog_search(
    category: str | None = None,
    brand: str | None = None,
    price_min: int | None = None,
    price_max: int | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    GET /catalog/search?category=...&brand=...&price_min=...&price_max=...&limit=...

    Поиск товаров по фильтрам напрямую в 1С.
    Используется когда агент извлёк чёткие фильтры из запроса.

    Возвращает список товаров: [{"id": uuid, "name": str, "price": float, "inStock": int}]
    Пустой список если 1С недоступна или ничего не найдено.
    """
    if not _is_configured():
        return []

    params: dict = {"limit": limit}
    if category:
        params["category"] = category
    if brand:
        params["brand"] = brand
    if price_min is not None:
        params["price_min"] = price_min
    if price_max is not None:
        params["price_max"] = price_max

    url = f"{_base_url()}/catalog/search"
    try:
        async with httpx.AsyncClient(timeout=_timeout(), auth=_auth()) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("items", [])

    except Exception as exc:
        logger.warning("1С catalog/search failed: %s", exc)
        return []


# ── Compare — сравнение товаров ───────────────────────────────────────────────

async def fetch_compare(product_ids: list[str]) -> dict | None:
    """
    POST /catalog/compare
    body: {"ids": ["uuid1", "uuid2"]}

    Возвращает данные для сравнения:
    {
        "products": [{"id": uuid, "name": str, "characteristics": {...}}],
        "diff_fields": ["Память", "Процессор"]  — поля где товары отличаются
    }
    None если 1С недоступна.
    """
    if not _is_configured() or len(product_ids) < 2:
        return None

    url = f"{_base_url()}/catalog/compare"
    try:
        async with httpx.AsyncClient(timeout=_timeout(), auth=_auth()) as client:
            resp = await client.post(url, json={"ids": product_ids})
            resp.raise_for_status()
            return resp.json()

    except Exception as exc:
        logger.warning("1С catalog/compare failed: %s", exc)
        return None


# ── Orders — статус заказа ────────────────────────────────────────────────────

async def fetch_order_status(order_number: str) -> dict | None:
    """
    GET /orders/{order_number}/status

    Возвращает статус заказа:
    {
        "number": "12345",
        "status": "В пути",
        "status_code": "in_delivery",
        "items": [{"name": str, "quantity": int, "price": float}],
        "delivery": {"date": "2026-06-10", "address": "...", "tracking_url": "..."},
        "total": 49990.0
    }
    None если 1С недоступна или заказ не найден.
    """
    if not _is_configured():
        return None

    url = f"{_base_url()}/orders/{order_number}/status"
    try:
        async with httpx.AsyncClient(timeout=_timeout(), auth=_auth()) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

    except Exception as exc:
        logger.warning("1С orders/%s/status failed: %s", order_number, exc)
        return None


# ── Promotions — акции и скидки ───────────────────────────────────────────────

async def fetch_promotions() -> list[dict]:
    """
    GET /promotions/active

    Возвращает список активных акций с кэшированием (TTL 2 мин):
    [
        {
            "id": "uuid",
            "title": "Скидки на iPhone",
            "description": "До 15% скидки на все модели iPhone",
            "discount_percent": 15,
            "until": "2026-06-30",
            "product_ids": ["uuid1", "uuid2"],
            "category_slug": "smartphones"
        }
    ]
    Пустой список если 1С недоступна.
    """
    if not _is_configured():
        return []

    cached = _promotions_cache.get()
    if cached is not None:
        return cached

    url = f"{_base_url()}/promotions/active"
    try:
        async with httpx.AsyncClient(timeout=_timeout(), auth=_auth()) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("promotions", data) if isinstance(data, dict) else data

        _promotions_cache.set(items)
        return items

    except Exception as exc:
        logger.warning("1С promotions/active failed: %s", exc)
        return []


# ── Recommendations — рекомендации к товару ───────────────────────────────────

async def fetch_recommendations(product_id: str, limit: int = 5) -> list[dict]:
    """
    GET /catalog/recommendations?product_id={uuid}&limit={n}

    Возвращает список рекомендованных товаров:
    [{"id": uuid, "name": str, "price": float, "inStock": int}]
    Пустой список если 1С недоступна.
    """
    if not _is_configured():
        return []

    url = f"{_base_url()}/catalog/recommendations"
    try:
        async with httpx.AsyncClient(timeout=_timeout(), auth=_auth()) as client:
            resp = await client.get(url, params={"product_id": product_id, "limit": limit})
            resp.raise_for_status()
            data = resp.json()
            return data.get("items", [])

    except Exception as exc:
        logger.warning("1С recommendations failed (id=%s): %s", product_id, exc)
        return []
