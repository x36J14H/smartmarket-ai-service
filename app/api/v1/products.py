import uuid
from typing import Any
from fastapi import APIRouter, Query
from pydantic import BaseModel
from app.db.qdrant import upsert, get_client, search

router = APIRouter(prefix="/products", tags=["products"])


class ProductItem(BaseModel):
    """Схема товара от 1С."""
    id: str                          # UUID от 1С
    name: str
    article: str | None = None
    slug: str | None = None
    description: str | None = None
    category: str | None = None
    category_slug: str | None = None
    subcategory: str | None = None
    subcategory_slug: str | None = None
    type: str | None = None
    type_slug: str | None = None
    brand: str | None = None
    brand_slug: str | None = None
    price: float | None = None
    in_stock: bool = True
    deleted: bool = False
    attributes: dict[str, Any] = {}
    images: list[str] = []
    embedding_text: str              # готовый текст для векторизации

    def to_upsert_item(self) -> dict:
        numeric_id = uuid.UUID(self.id).int % (2**63)
        return {
            "id": numeric_id,
            "text": self.embedding_text,
            "source_id": self.id,
            "name": self.name,
            "article": self.article,
            "slug": self.slug,
            "category": self.category,
            "subcategory": self.subcategory,
            "type": self.type,
            "brand": self.brand,
            "price": self.price,
            "in_stock": self.in_stock,
            "deleted": self.deleted,
            "attributes": self.attributes,
            "images": self.images,
        }


@router.post("")
def upsert_products(items: list[ProductItem]):
    """Приём товаров от 1С."""
    to_insert = [p.to_upsert_item() for p in items if not p.deleted]
    if not to_insert:
        return {"inserted": 0, "skipped": len(items)}
    count = upsert("products", to_insert)
    return {"inserted": count, "skipped": len(items) - count}


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
