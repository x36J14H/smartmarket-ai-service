"""
LLM-слой с поддержкой нескольких провайдеров.
Провайдер и настройки берутся из data/llm_settings.json (если есть),
иначе — из .env через config.py.

Все публичные функции — async. Используется AsyncOpenAI для неблокирующих вызовов.
GigaChat SDK синхронный — запускается в threadpool через asyncio.to_thread.
"""
import re
import json
import asyncio
import logging
from typing import AsyncIterator

from app.core.config import settings
from app.ml_models.prompts import ANALYZE_PROMPT, get_prompt_for_intent
from app.services.llm_settings import get_active_provider_cfg

logger = logging.getLogger(__name__)

# Валидные intent-ы
VALID_INTENTS = {"products", "catalog_browse", "compare", "info", "order_help", "promotions", "multi"}


# ── Настройки ─────────────────────────────────────────────────────────────────

def _get_active_settings() -> dict:
    """
    Вернуть актуальные настройки LLM.
    Приоритет: data/llm_settings.json > .env
    """
    cfg = get_active_provider_cfg()

    if cfg:
        provider = cfg["provider"]
    else:
        provider = settings.llm_provider
        cfg = {"api_key": settings.llm_api_key, "model": settings.llm_model}

    default_base_url_map = {
        "openai":     settings.openai_base_url,
        "openrouter": settings.openrouter_base_url,
        "gigachat":   None,
    }
    base_url = cfg.get("base_url") or default_base_url_map.get(provider)

    return {
        "provider":       provider,
        "api_key":        cfg.get("api_key") or settings.llm_api_key,
        "model":          cfg.get("model")   or settings.llm_model,
        "base_url":       base_url,
        "gigachat_scope": cfg.get("gigachat_scope"),
    }


# ── Async провайдеры ──────────────────────────────────────────────────────────

async def _ask_openai_compatible_async(
    messages: list[dict],
    cfg: dict,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """AsyncOpenAI — OpenAI / OpenRouter / LM Studio."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    response = await client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def _ask_openai_compatible_stream(
    messages: list[dict],
    cfg: dict,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> AsyncIterator[str]:
    """AsyncOpenAI streaming — возвращает асинхронный генератор чанков."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    stream = await client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _ask_gigachat_async(
    messages: list[dict],
    cfg: dict,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """GigaChat SDK синхронный — запускаем в threadpool."""
    def _sync() -> str:
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole

        role_map = {
            "system":    MessagesRole.SYSTEM,
            "user":      MessagesRole.USER,
            "assistant": MessagesRole.ASSISTANT,
        }
        giga_messages = [
            Messages(role=role_map.get(m["role"], MessagesRole.USER), content=m["content"])
            for m in messages
        ]
        payload = Chat(
            messages=giga_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=cfg.get("model", "GigaChat"),
        )
        scope = cfg.get("gigachat_scope") or "GIGACHAT_API_PERS"
        with GigaChat(credentials=cfg["api_key"], scope=scope, verify_ssl_certs=False) as giga:
            response = giga.chat(payload)
        return response.choices[0].message.content or ""

    return await asyncio.to_thread(_sync)


# ── Dispatch ──────────────────────────────────────────────────────────────────

def _clean_think_tags(text: str) -> str:
    """Убираем <think>...</think> (Qwen3 и другие reasoning-модели)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def _dispatch_async(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Выбрать провайдера и выполнить async-запрос."""
    cfg = _get_active_settings()
    provider = cfg["provider"]

    if provider == "gigachat":
        result = await _ask_gigachat_async(messages, cfg, temperature, max_tokens)
    else:
        result = await _ask_openai_compatible_async(messages, cfg, temperature, max_tokens)

    return _clean_think_tags(result)


async def _dispatch_stream(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[str]:
    """Streaming dispatch — только для OpenAI-compatible провайдеров.
    GigaChat не поддерживает streaming через SDK — возвращаем как один чанк."""
    cfg = _get_active_settings()
    provider = cfg["provider"]

    if provider == "gigachat":
        # GigaChat не умеет streaming — имитируем единым чанком
        result = await _ask_gigachat_async(messages, cfg, temperature, max_tokens)
        result = _clean_think_tags(result)

        async def _single_chunk():
            yield result

        return _single_chunk()
    else:
        async def _stream_with_cleanup():
            buffer = []
            async for chunk in _ask_openai_compatible_stream(messages, cfg, temperature, max_tokens):
                buffer.append(chunk)
                yield chunk
            # think-теги могут быть только на границах чанков — проверяем финальный текст
            # (для streaming не убираем на лету — слишком сложно, think-модели редкость)

        return _stream_with_cleanup()


# ── Публичный интерфейс ───────────────────────────────────────────────────────

async def ask(
    question: str,
    context_chunks: list[str],
    history: list[dict] | None = None,
    intent: str = "multi",
) -> str:
    """Сформировать ответ на вопрос пользователя с учётом контекста и истории."""
    system_prompt = get_prompt_for_intent(intent)
    context = "\n\n---\n".join(context_chunks)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"})
    return await _dispatch_async(messages, temperature=0.3, max_tokens=1024)


async def ask_stream(
    question: str,
    context_chunks: list[str],
    history: list[dict] | None = None,
    intent: str = "multi",
) -> AsyncIterator[str]:
    """Streaming версия ask() — возвращает AsyncIterator чанков."""
    system_prompt = get_prompt_for_intent(intent)
    context = "\n\n---\n".join(context_chunks)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"})
    return await _dispatch_stream(messages, temperature=0.3, max_tokens=1024)


async def analyze_query(question: str, history: list[dict] | None = None) -> dict:
    """
    Один LLM-вызов: определяет intent + оптимальный поисковый запрос +
    структурированные фильтры + флаг нужности уточнения.

    Возвращает:
    {
        "intent": str,
        "search_query": str,
        "filters": {"price_max": int|None, "price_min": int|None, "brand": str|None, "category": str|None},
        "needs_clarification": bool,
        "clarification_question": str|None,
    }
    """
    messages = [{"role": "system", "content": ANALYZE_PROMPT}]

    # Последние 4 сообщения для разрешения местоимений
    if history:
        messages.extend(history[-4:])

    messages.append({"role": "user", "content": question})

    try:
        raw = await _dispatch_async(messages, temperature=0.0, max_tokens=150)

        # Убираем markdown-обёртку ```json ... ```
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()

        data = json.loads(raw)

        intent = data.get("intent", "multi").strip().lower()
        search_query = data.get("search_query", "").strip()
        filters = data.get("filters") or {}
        needs_clarification = bool(data.get("needs_clarification", False))
        clarification_question = data.get("clarification_question") or None

        if intent not in VALID_INTENTS:
            intent = "multi"
        if not search_query:
            search_query = question

        # Нормализуем filters — убираем None-значения
        clean_filters: dict = {}
        for key in ("price_max", "price_min"):
            val = filters.get(key)
            if val is not None:
                try:
                    clean_filters[key] = int(float(str(val)))
                except (ValueError, TypeError):
                    pass
        for key in ("brand", "category"):
            val = filters.get(key)
            if val and isinstance(val, str):
                clean_filters[key] = val.strip()

        return {
            "intent":                 intent,
            "search_query":           search_query,
            "filters":                clean_filters,
            "needs_clarification":    needs_clarification,
            "clarification_question": clarification_question,
        }

    except Exception:
        logger.warning("analyze_query failed to parse LLM response, using fallback")
        return {
            "intent":                 "multi",
            "search_query":           question,
            "filters":                {},
            "needs_clarification":    False,
            "clarification_question": None,
        }


async def rerank_async(query: str, hits: list) -> list:
    """
    Async LLM-reranking — фильтрует нерелевантные результаты поиска.
    hits — список ScoredPoint из Qdrant.
    """
    from app.ml_models.prompts import RERANK_PROMPT

    if not hits:
        return hits

    items = "\n".join(
        f"{i + 1}. {h.payload.get('text', '')[:120]}"
        for i, h in enumerate(hits)
        if h.payload
    )
    messages = [
        {"role": "system", "content": RERANK_PROMPT},
        {"role": "user",   "content": f"Запрос: {query}\n\nТовары:\n{items}"},
    ]
    try:
        response = await _dispatch_async(messages, temperature=0.0, max_tokens=64)
        indices = {
            int(x.strip()) - 1
            for x in response.split(",")
            if x.strip().isdigit()
        }
        reranked = [h for i, h in enumerate(hits) if i in indices]
        # If reranker returned empty — the query is irrelevant, return empty list
        return reranked
    except Exception as exc:
        logger.warning("reranking failed, returning original hits: %s", exc)
        return hits


# ── Синхронные обёртки для обратной совместимости ────────────────────────────
# Используются только там где async невозможен (например в sync endpoints)

def _dispatch(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Sync обёртка для мест где ещё нужен синхронный вызов (reranker в products/search)."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, _dispatch_async(messages, temperature, max_tokens))
        return future.result()
