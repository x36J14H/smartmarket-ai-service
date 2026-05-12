"""
Получение каталога категорий из 1С и форматирование в читаемый текст.

Эндпоинт: GET {ONEC_BASE_URL}/categories
Ответ: {"categories": [{"name": ..., "slug": ..., "subcategories": [...]}]}
"""

import logging
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

# TTL-кэш: не дёргаем 1С на каждый запрос
_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 300.0  # 5 минут


def _build_answer(data: dict) -> str:
    """
    Строит готовый ответ пользователю — markdown со ссылками.
    Этот текст отдаётся напрямую, без прохода через LLM.
    """
    lines = ["В нашем магазине представлены следующие категории товаров:\n"]

    for cat in data.get("categories", []):
        cat_name = cat.get("name", "")
        lines.append(f"**{cat_name}**")

        for sub in cat.get("subcategories", []):
            sub_name = sub.get("name", "")
            sub_slug = sub.get("slug", "")
            types = sub.get("types", [])

            if types:
                for t in types:
                    t_name = t.get("name", "")
                    t_slug = t.get("slug", "")
                    if t_slug:
                        lines.append(f"- [{t_name}](/catalog/{t_slug})")
                    else:
                        lines.append(f"- {t_name}")
            else:
                if sub_slug:
                    lines.append(f"- [{sub_name}](/catalog/{sub_slug})")
                else:
                    lines.append(f"- {sub_name}")

        lines.append("")  # пустая строка между категориями

    return "\n".join(lines).strip()


async def _fetch_raw() -> dict | None:
    """Запросить сырой JSON из 1С с кэшированием."""
    import time

    if not settings.onec_base_url:
        return None

    now = time.monotonic()
    if _cache["data"] is not None and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]

    url = f"{settings.onec_base_url.rstrip('/')}/categories"
    try:
        auth = (settings.onec_user, settings.onec_password) if settings.onec_user else None
        async with httpx.AsyncClient(timeout=5.0, auth=auth) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        _cache["data"] = data
        _cache["ts"] = now
        return data

    except Exception as exc:
        logger.warning("categories fetch failed: %s", exc)
        return None


async def get_catalog_answer() -> str | None:
    """
    Вернуть готовый markdown-ответ со списком категорий и ссылками.
    None если 1С недоступна.
    """
    data = await _fetch_raw()
    return _build_answer(data) if data else None


def _find_best_category(data: dict, question: str) -> tuple[str, str] | None:
    """
    Ищет наиболее подходящую категорию/подкатегорию по ключевым словам из вопроса.
    Возвращает (name, slug) или None если ничего не нашли.

    Приоритет: типы > подкатегории > категории верхнего уровня.
    """
    q = question.lower()

    # Собираем все варианты: (name, slug, priority)
    # priority: 3=тип, 2=подкатегория, 1=категория
    candidates: list[tuple[str, str, int]] = []

    for cat in data.get("categories", []):
        cat_name = cat.get("name", "")
        cat_slug = cat.get("slug", "")
        if cat_slug:
            candidates.append((cat_name, cat_slug, 1))

        for sub in cat.get("subcategories", []):
            sub_name = sub.get("name", "")
            sub_slug = sub.get("slug", "")
            if sub_slug:
                candidates.append((sub_name, sub_slug, 2))

            for t in sub.get("types", []):
                t_name = t.get("name", "")
                t_slug = t.get("slug", "")
                if t_slug:
                    candidates.append((t_name, t_slug, 3))

    # Ищем совпадение по словам из названия категории в вопросе
    best: tuple[str, str, int] | None = None
    for name, slug, priority in candidates:
        # Проверяем каждое слово из названия категории (длиннее 2 символов)
        words = [w.lower() for w in name.split() if len(w) > 2]
        if any(w in q for w in words):
            if best is None or priority > best[2]:
                best = (name, slug, priority)

    return (best[0], best[1]) if best else None


async def get_browse_answer(question: str) -> str | None:
    """
    Для вопроса типа "покажи смартфоны" — найти подходящую категорию
    и вернуть ответ со ссылкой на неё.
    Если категория не найдена — вернуть полный список (как get_catalog_answer).
    None если 1С недоступна.
    """
    data = await _fetch_raw()
    if not data:
        return None

    match = _find_best_category(data, question)
    if match:
        name, slug = match
        return (
            f"В нашем каталоге есть раздел [{name}](/catalog/{slug}). "
            f"Перейдите по ссылке чтобы посмотреть все товары в этой категории."
        )

    # Категория не распознана — показываем весь каталог
    return _build_answer(data)
