"""
Форматирование каталога категорий для ответа пользователю.
Данные получаем через onec_client.fetch_categories().
"""
from app.services.onec_client import fetch_categories


def _build_catalog_answer(data: dict) -> str:
    """Строит markdown-ответ со списком категорий и ссылками."""
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

        lines.append("")

    return "\n".join(lines).strip()


def _find_best_category(data: dict, query: str) -> tuple[str, str] | None:
    """
    Ищет наиболее подходящую категорию по ключевым словам из запроса.
    Приоритет: типы > подкатегории > категории верхнего уровня.
    Возвращает (name, slug) или None.
    """
    q = query.lower()
    candidates: list[tuple[str, str, int]] = []

    for cat in data.get("categories", []):
        if cat.get("slug"):
            candidates.append((cat["name"], cat["slug"], 1))
        for sub in cat.get("subcategories", []):
            if sub.get("slug"):
                candidates.append((sub["name"], sub["slug"], 2))
            for t in sub.get("types", []):
                if t.get("slug"):
                    candidates.append((t["name"], t["slug"], 3))

    best: tuple[str, str, int] | None = None
    for name, slug, priority in candidates:
        words = [w.lower() for w in name.split() if len(w) > 2]
        if any(w in q for w in words):
            if best is None or priority > best[2]:
                best = (name, slug, priority)

    return (best[0], best[1]) if best else None


async def get_catalog_answer() -> str | None:
    """Полный список категорий — markdown со ссылками. None если 1С недоступна."""
    data = await fetch_categories()
    return _build_catalog_answer(data) if data else None


async def get_browse_answer(query: str) -> str | None:
    """
    Для запроса типа "покажи смартфоны" — найти подходящую категорию и вернуть ссылку.
    Если категория не найдена — вернуть полный список.
    None если 1С недоступна.
    """
    data = await fetch_categories()
    if not data:
        return None

    match = _find_best_category(data, query)
    if match:
        name, slug = match
        return (
            f"В нашем каталоге есть раздел [{name}](/catalog/{slug}). "
            f"Перейдите по ссылке чтобы посмотреть все товары в этой категории."
        )

    return _build_catalog_answer(data)
