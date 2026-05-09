from fastapi import APIRouter
from pydantic import BaseModel
from app.db.qdrant import upsert, get_client, uuid_to_int64

router = APIRouter(tags=["knowledge"])


class FaqItem(BaseModel):
    """Одна FAQ-запись из 1С."""
    id: str                          # UUID из 1С
    question: str                    # вопрос — используется как заголовок
    answer: str                      # ответ
    category: str | None = None      # раздел FAQ, например "Доставка"
    deleted: bool = False

    def to_upsert_item(self) -> dict:
        numeric_id = uuid_to_int64(self.id)
        # Векторизуем вопрос + ответ вместе — так поиск работает лучше
        text = f"{self.question}\n{self.answer}"
        return {
            "id": numeric_id,
            "text": text,
            "source_id": self.id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
        }


class NavigationItem(BaseModel):
    """Один элемент навигации / раздел сайта из 1С."""
    id: str                          # UUID из 1С
    title: str                       # название раздела
    description: str                 # описание — что здесь можно найти/сделать
    url: str | None = None           # путь на сайте, например "/catalog/phones"
    category: str | None = None      # группа навигации, например "Каталог"
    deleted: bool = False

    def to_upsert_item(self) -> dict:
        numeric_id = uuid_to_int64(self.id)
        # Векторизуем заголовок + описание
        text = f"{self.title}\n{self.description}"
        return {
            "id": numeric_id,
            "text": text,
            "source_id": self.id,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "category": self.category,
        }


# ── FAQ ──────────────────────────────────────────────────────────────────────

@router.post("/faq")
def upsert_faq(items: list[FaqItem]):
    """Загрузить / обновить FAQ-записи из 1С."""
    to_insert = [i.to_upsert_item() for i in items if not i.deleted]
    deleted_ids = [uuid_to_int64(i.id) for i in items if i.deleted]

    if deleted_ids:
        get_client().delete(collection_name="faq", points_selector=deleted_ids)

    if not to_insert:
        return {"inserted": 0, "deleted": len(deleted_ids), "skipped": len(items) - len(deleted_ids)}

    count = upsert("faq", to_insert)
    return {"inserted": count, "deleted": len(deleted_ids), "skipped": len(items) - count - len(deleted_ids)}


@router.delete("/faq/{item_id}")
def delete_faq(item_id: str):
    """Удалить FAQ-запись по UUID из 1С."""
    numeric_id = uuid_to_int64(item_id)
    get_client().delete(collection_name="faq", points_selector=[numeric_id])
    return {"deleted": item_id}


# ── Navigation ────────────────────────────────────────────────────────────────

@router.post("/navigation")
def upsert_navigation(items: list[NavigationItem]):
    """Загрузить / обновить навигационные записи из 1С."""
    to_insert = [i.to_upsert_item() for i in items if not i.deleted]
    deleted_ids = [uuid_to_int64(i.id) for i in items if i.deleted]

    if deleted_ids:
        get_client().delete(collection_name="navigation", points_selector=deleted_ids)

    if not to_insert:
        return {"inserted": 0, "deleted": len(deleted_ids), "skipped": len(items) - len(deleted_ids)}

    count = upsert("navigation", to_insert)
    return {"inserted": count, "deleted": len(deleted_ids), "skipped": len(items) - count - len(deleted_ids)}


@router.delete("/navigation/{item_id}")
def delete_navigation(item_id: str):
    """Удалить навигационную запись по UUID из 1С."""
    numeric_id = uuid_to_int64(item_id)
    get_client().delete(collection_name="navigation", points_selector=[numeric_id])
    return {"deleted": item_id}
