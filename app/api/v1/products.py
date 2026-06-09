from fastapi import APIRouter
from pydantic import BaseModel
from app.db.qdrant import upsert, get_client, search, uuid_to_int64
from app.services.onec_client import filter_available_ids
from app.services.reranker import rerank

router = APIRouter(prefix="/products", tags=["products"])


class ProductItem(BaseModel):
    """Схема товара от 1С."""
    id: str              # UUID от 1С
    name: str
    price: float | None = None
    embedding_text: str  # готовый текст для векторизации
    deleted: bool = False

    def to_upsert_item(self) -> dict:
        numeric_id = uuid_to_int64(self.id)
        return {
            "id":        numeric_id,
            "text":      self.embedding_text,
            "source_id": self.id,
            "name":      self.name,
            "price":     self.price,
        }


@router.post("")
def upsert_products(items: list[ProductItem]):
    """
    Приём товаров из 1С.
    Поддерживает мягкое удаление через поле deleted=true.
    """
    to_insert = [p.to_upsert_item() for p in items if not p.deleted]
    deleted_ids = [uuid_to_int64(p.id) for p in items if p.deleted]

    if deleted_ids:
        get_client().delete(collection_name="products", points_selector=deleted_ids)

    count = upsert("products", to_insert) if to_insert else 0
    return {"inserted": count, "deleted": len(deleted_ids)}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    score_threshold: float = 0.45  # отсекаем явно нерелевантные результаты


@router.post("/search")
async def search_products(req: SearchRequest):
    """
    Семантический поиск товаров с LLM-reranking.

    Шаги:
    1. Гибридный поиск (dense + BM25) — берём top_k*2 кандидатов выше порога
    2. LLM-reranking — отсеивает нерелевантные товары (возвращает [] если ничего не подходит)
    3. Фильтр по наличию через 1С
    4. Возвращаем top_k UUID или пустой список если ничего не найдено
    """
    candidates = search(req.query, "products", top_k=req.top_k * 2, score_threshold=req.score_threshold)

    if not candidates:
        return {"ids": []}

    reranked = await rerank(req.query, candidates)

    if not reranked:
        return {"ids": []}

    all_ids = [p.payload.get("source_id") for p in reranked if p.payload]
    available = await filter_available_ids(all_ids)
    filtered = [p for p in reranked if p.payload and p.payload.get("source_id") in available]

    result_ids = [p.payload["source_id"] for p in filtered[: req.top_k]]

    return {"ids": result_ids}


@router.delete("/{product_id}")
def delete_product(product_id: str):
    """Удаление товара по UUID из 1С."""
    numeric_id = uuid_to_int64(product_id)
    get_client().delete(collection_name="products", points_selector=[numeric_id])
    return {"deleted": product_id}
