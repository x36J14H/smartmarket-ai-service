import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import warnings
from pathlib import Path
from functools import lru_cache
from fastembed import TextEmbedding, SparseTextEmbedding
from app.core.config import settings

warnings.filterwarnings("ignore", category=UserWarning, module="fastembed")

# Кэш моделей в папке проекта
MODELS_CACHE = str(Path(__file__).resolve().parents[2] / "models_cache")

# Sparse-модель: BM25 — быстрый keyword-матч, не требует GPU
SPARSE_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    return TextEmbedding(settings.embed_model, cache_dir=MODELS_CACHE)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(SPARSE_MODEL_NAME, cache_dir=MODELS_CACHE)


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return [v.tolist() for v in model.embed(texts)]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def sparse_embed(texts: list[str]) -> list:
    model = get_sparse_model()
    return list(model.embed(texts))


def sparse_embed_one(text: str):
    return sparse_embed([text])[0]
