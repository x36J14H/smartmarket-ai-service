import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import hashlib
import struct
import uuid as _uuid
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from app.core.config import settings
from app.ml_models.embedder import embed_one

# Коллекции: товары, навигация по сайту, FAQ о компании
COLLECTIONS = ["products", "navigation", "faq"]


def uuid_to_int64(source_id: str) -> int:
    """
    Конвертировать UUID из 1С в int64 для Qdrant.

    Используем SHA-1 от байт UUID, берём первые 8 байт и маскируем старший бит
    чтобы получить неотрицательный int63. Это гарантирует уникальность даже
    для UUID, которые отличаются ровно на 2**63 (при modulo такие UUID коллидируют).
    """
    uid_bytes = _uuid.UUID(source_id).bytes
    digest = hashlib.sha1(uid_bytes).digest()[:8]
    return struct.unpack(">Q", digest)[0] & 0x7FFFFFFFFFFFFFFF


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collections() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    for name in COLLECTIONS:
        if name in existing:
            # Проверяем размерность — если не совпадает, пересоздаём
            info = client.get_collection(name)
            current_dim = info.config.params.vectors.size
            if current_dim != settings.embed_dim:
                client.delete_collection(name)
                existing.discard(name)
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
            )


def upsert(collection: str, items: list[dict]) -> int:
    """items: [{"id": int, "text": str, **payload}]"""
    client = get_client()
    points = [
        PointStruct(
            id=item["id"],
            vector=embed_one(item["text"]),
            payload={k: v for k, v in item.items() if k != "id"},
        )
        for item in items
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


def search(query: str, collection: str, top_k: int = None) -> list[ScoredPoint]:
    client = get_client()
    k = top_k or settings.top_k
    return client.query_points(
        collection_name=collection,
        query=embed_one(query),
        limit=k,
    ).points


def search_all(query: str, top_k: int = None) -> list[ScoredPoint]:
    """Поиск по всем коллекциям, возвращает топ-k суммарно."""
    results = []
    for col in COLLECTIONS:
        results.extend(search(query, col, top_k=top_k or settings.top_k))
    results.sort(key=lambda p: p.score, reverse=True)
    return results[: top_k or settings.top_k]
