import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import hashlib
import struct
import uuid as _uuid
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, ScoredPoint, SparseVector,
    Prefetch, FusionQuery, Fusion,
)
from app.core.config import settings
from app.ml_models.embedder import embed_one, sparse_embed_one

# Коллекции: товары, навигация по сайту, FAQ о компании
COLLECTIONS = ["products", "navigation", "faq"]

# Имена векторов внутри коллекции
DENSE_VECTOR  = "dense"
SPARSE_VECTOR = "sparse"


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
            # Проверяем размерность dense-вектора — если не совпадает, пересоздаём
            info = client.get_collection(name)
            vectors = info.config.params.vectors
            # vectors может быть dict (named) или VectorParams (unnamed)
            if isinstance(vectors, dict):
                current_dim = vectors.get(DENSE_VECTOR, VectorParams(size=0, distance=Distance.COSINE)).size
            else:
                # Старая коллекция без named vectors — пересоздаём
                current_dim = None

            if current_dim != settings.embed_dim:
                client.delete_collection(name)
                existing.discard(name)

        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR: VectorParams(
                        size=settings.embed_dim,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False),
                    ),
                },
            )


def upsert(collection: str, items: list[dict]) -> int:
    """items: [{"id": int, "text": str, **payload}]"""
    client = get_client()
    points = []
    for item in items:
        text = item["text"]
        dense  = embed_one(text)
        sparse = sparse_embed_one(text)

        points.append(PointStruct(
            id=item["id"],
            vector={
                DENSE_VECTOR:  dense,
                SPARSE_VECTOR: SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            payload={k: v for k, v in item.items() if k != "id"},
        ))

    client.upsert(collection_name=collection, points=points)
    return len(points)


def search(query: str, collection: str, top_k: int = None, score_threshold: float = None) -> list[ScoredPoint]:
    """Гибридный поиск: dense (semantic) + sparse (BM25 keyword), fusion RRF."""
    client = get_client()
    k = top_k or settings.top_k
    threshold = score_threshold if score_threshold is not None else settings.score_threshold

    # prefetch_k — берём больше кандидатов для каждого вектора перед fusion
    prefetch_k = max(k * 3, 20)

    results = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=embed_one(query),
                using=DENSE_VECTOR,
                limit=prefetch_k,
            ),
            Prefetch(
                query=SparseVector(**_sparse_query(query)),
                using=SPARSE_VECTOR,
                limit=prefetch_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        score_threshold=threshold,
    ).points

    return results


def _sparse_query(text: str) -> dict:
    """Генерирует sparse вектор для запроса."""
    sv = sparse_embed_one(text)
    return {"indices": sv.indices.tolist(), "values": sv.values.tolist()}


def search_all(query: str, top_k: int = None, score_threshold: float = None) -> list[ScoredPoint]:
    """Гибридный поиск по всем коллекциям, возвращает топ-k суммарно."""
    threshold = score_threshold if score_threshold is not None else settings.score_threshold
    results = []
    for col in COLLECTIONS:
        hits = search(query, col, top_k=top_k or settings.top_k, score_threshold=threshold)
        # Проставляем имя коллекции в payload, чтобы bot.py мог корректно
        # форматировать чанки (ссылки на товары и разделы сайта)
        for hit in hits:
            if hit.payload is not None:
                hit.payload["collection"] = col
        results.extend(hits)
    results.sort(key=lambda p: p.score, reverse=True)
    return results[: top_k or settings.top_k]
