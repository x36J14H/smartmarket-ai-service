import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.qdrant import ensure_collections
from app.api.v1 import chat, products, embed


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collections()
    yield


app = FastAPI(title="Shop Bot API", lifespan=lifespan)

app.include_router(chat.router,     prefix="/api/v1")
app.include_router(products.router, prefix="/api/v1")
app.include_router(embed.router,    prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
