from fastapi import APIRouter
from pydantic import BaseModel
from app.db.qdrant import upsert, get_client, search, uuid_to_int64
from app.services.availability import filter_available_ids
from app.services.reranker import rerank

router = APIRouter(prefix="/products", tags=["products"])


class ProductItem(BaseModel):
    """Схема товара от 1С."""
    id: str              # UUID от 1С
    name: str
    price: float | None = None
    embedding_text: str  # готовый текст для векторизации

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
    """Приём товаров из 1С."""
    count = upsert("products", [p.to_upsert_item() for p in items])
    return {"inserted": count}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@router.post("/search")
async def search_products(req: SearchRequest):
    """
    Семантический поиск товаров с LLM-reranking.

    Шаги:
    1. Гибридный поиск (dense + BM25) — берём top_k*2 кандидатов
    2. LLM-reranking — отсеивает нерелевантные товары
    3. Фильтр по наличию через 1С
    4. Возвращаем top_k UUID
    """
    # Берём больше кандидатов для reranker, без порога по score —
    # reranker сам отсеет нерелевантное
    candidates = search(req.query, "products", top_k=req.top_k * 2, score_threshold=0.0)

    # LLM отсеивает нерелевантные
    reranked = rerank(req.query, candidates)

    # Фильтруем по наличию через 1С до обрезки до top_k —
    # чтобы не потерять позиции из-за недоступных товаров в начале списка
    all_ids = [p.payload.get("source_id") for p in reranked if p.payload]
    available = await filter_available_ids(all_ids)
    filtered = [p for p in reranked if p.payload and p.payload.get("source_id") in available]

    # Обрезаем до нужного количества уже после фильтра
    result_ids = [p.payload["source_id"] for p in filtered[: req.top_k]]

    return {"ids": result_ids}


@router.delete("/{product_id}")
def delete_product(product_id: str):
    """Удаление товара по UUID из 1С."""
    numeric_id = uuid_to_int64(product_id)
    get_client().delete(collection_name="products", points_selector=[numeric_id])
    return {"deleted": product_id}
