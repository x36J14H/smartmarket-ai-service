import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from app.db.qdrant import upsert, get_client, search

router = APIRouter(prefix="/products", tags=["products"])


class ProductItem(BaseModel):
    """Схема товара от 1С."""
    id: str              # UUID от 1С
    name: str
    price: float | None = None
    embedding_text: str  # готовый текст для векторизации

    def to_upsert_item(self) -> dict:
        numeric_id = uuid.UUID(self.id).int % (2**63)
        return {
            "id":        numeric_id,
            "text":      self.embedding_text,
            "source_id": self.id,
            "name":      self.name,
            "price":     self.price,
        }


@router.post("")
def upsert_products(items: list[ProductItem]):
    """Приём товаров от 1С."""
    count = upsert("products", [p.to_upsert_item() for p in items])
    return {"inserted": count}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@router.post("/search")
def search_products(req: SearchRequest):
    """Семантический поиск товаров по тексту. Возвращает список UUID товаров."""
    hits = search(req.query, "products", top_k=req.top_k)
    return {"ids": [p.payload.get("source_id") for p in hits if p.payload]}


@router.delete("/{product_id}")
def delete_product(product_id: str):
    """Удаление товара по UUID от 1С."""
    numeric_id = uuid.UUID(product_id).int % (2**63)
    get_client().delete(collection_name="products", points_selector=[numeric_id])
    return {"deleted": product_id}
