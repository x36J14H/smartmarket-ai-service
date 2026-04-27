from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db.qdrant import upsert, search, COLLECTIONS

router = APIRouter(prefix="/embed", tags=["embed"])


class UpsertItem(BaseModel):
    id: int
    text: str
    model_config = {"extra": "allow"}


class UpsertRequest(BaseModel):
    collection: str
    items: list[UpsertItem]


class SearchRequest(BaseModel):
    collection: str
    query: str
    top_k: int = 4


@router.post("/upsert")
def embed_upsert(req: UpsertRequest):
    if req.collection not in COLLECTIONS:
        raise HTTPException(400, f"Unknown collection. Use one of: {COLLECTIONS}")
    items = [item.model_dump() for item in req.items]
    count = upsert(req.collection, items)
    return {"inserted": count, "collection": req.collection}


@router.post("/search")
def embed_search(req: SearchRequest):
    if req.collection not in COLLECTIONS:
        raise HTTPException(400, f"Unknown collection. Use one of: {COLLECTIONS}")
    hits = search(req.query, req.collection, top_k=req.top_k)
    return {
        "results": [
            {"id": h.id, "score": round(h.score, 3), "payload": h.payload}
            for h in hits
        ]
    }
