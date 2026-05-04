"""
LLM-слой с поддержкой нескольких провайдеров.
Провайдер и настройки берутся из data/llm_settings.json (если есть),
иначе — из .env через config.py.
"""
import re
from app.core.config import settings
from app.ml_models.prompts import SYSTEM_PROMPT
from app.services.llm_settings import get_active_provider_cfg


def _get_active_settings() -> dict:
    """
    Вернуть актуальные настройки LLM.
    Приоритет: data/llm_settings.json > .env
    base_url всегда берётся из .env по имени провайдера.
    """
    cfg = get_active_provider_cfg()

    if cfg:
        provider = cfg["provider"]
    else:
        provider = settings.llm_provider
        cfg = {"api_key": settings.llm_api_key, "model": settings.llm_model}

    # base_url: приоритет — кастомный из настроек провайдера, затем дефолт из .env
    default_base_url_map = {
        "openai":     settings.openai_base_url,
        "openrouter": settings.openrouter_base_url,
        "gigachat":   None,  # SDK не использует base_url
    }
    base_url = cfg.get("base_url") or default_base_url_map.get(provider)

    return {
        "provider":       provider,
        "api_key":        cfg.get("api_key") or settings.llm_api_key,
        "model":          cfg.get("model")   or settings.llm_model,
        "base_url":       base_url,
        "gigachat_scope": cfg.get("gigachat_scope"),
    }


# ── Провайдеры ────────────────────────────────────────────────────────────────

def _ask_openai_compatible(
    messages: list[dict],
    cfg: dict,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """OpenAI / OpenRouter / LM Studio — все через openai-клиент."""
    from openai import OpenAI
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _ask_gigachat(
    messages: list[dict],
    cfg: dict,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """GigaChat через официальный SDK (pip install gigachat)."""
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

    # credentials — base64-строка из личного кабинета Сбера
    # scope: GIGACHAT_API_PERS | GIGACHAT_API_B2B | GIGACHAT_API_CORP
    scope = cfg.get("gigachat_scope") or "GIGACHAT_API_PERS"
    with GigaChat(credentials=cfg["api_key"], scope=scope, verify_ssl_certs=False) as giga:
        response = giga.chat(payload)

    return response.choices[0].message.content


# ── Публичный интерфейс ───────────────────────────────────────────────────────

def _dispatch(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Выбрать провайдера и выполнить запрос."""
    cfg = _get_active_settings()
    provider = cfg["provider"]

    if provider == "gigachat":
        result = _ask_gigachat(messages, cfg, temperature, max_tokens)
    else:
        # openai / openrouter — OpenAI-compatible (openai также поддерживает кастомный base_url)
        result = _ask_openai_compatible(messages, cfg, temperature, max_tokens)

    # Убираем <think>...</think> (Qwen3 и другие reasoning-модели)
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result


def ask(question: str, context_chunks: list[str], history: list[dict] = None) -> str:
    context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"})
    return _dispatch(messages, temperature=0.3, max_tokens=1024)


HYDE_PROMPT = """Ты — эксперт интернет-магазина электроники.
Напиши короткое описание товара или информации, которая идеально отвечала бы на вопрос пользователя.
Пиши как будто это текст из каталога или FAQ магазина — конкретно, без лишних слов.
Не отвечай на вопрос напрямую, просто опиши идеальный документ."""


def hypothetical_answer(question: str) -> str:
    """HyDE: генерирует гипотетический документ для улучшения векторного поиска."""
    messages = [
        {"role": "system", "content": HYDE_PROMPT},
        {"role": "user", "content": question},
    ]
    return _dispatch(messages, temperature=0.5, max_tokens=256)
