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
    context = "\n\n---\n".join(context_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"})
    return _dispatch(messages, temperature=0.3, max_tokens=1024)


# ── Переформулирование запроса с учётом истории ───────────────────────────────

# ── Каталог: keyword-детектор ─────────────────────────────────────────────────

# Ключевые слова для детектирования вопросов об ассортименте/категориях
_CATALOG_KEYWORDS = (
    "какие товары", "какой товар", "что у вас есть", "что продаёте", "что продаете",
    "что можно купить", "ассортимент", "категории", "каталог", "какие категории",
    "какие разделы", "что есть в магазине", "что вы продаёте", "что вы продаете",
    "какие продукты", "что имеется", "что в наличии есть",
)


def is_catalog_question(question: str) -> bool:
    """
    Быстрая keyword-проверка: спрашивает ли пользователь об ассортименте/категориях.
    Не использует LLM — работает мгновенно.
    """
    q = question.lower()
    return any(kw in q for kw in _CATALOG_KEYWORDS)


# ── Анализ запроса: intent + поисковый запрос (один LLM-вызов) ────────────────

_ANALYZE_PROMPT = """Ты — анализатор запросов для интернет-магазина электроники.
Тебе дан вопрос пользователя (и возможно история диалога).

Выполни ДВЕ задачи:

1. INTENT — определи категорию вопроса:
- products — ищет конкретный товар с параметрами (бренд, характеристики, цена, модель)
- catalog_browse — хочет посмотреть категорию в целом, без конкретных требований
- info — вопрос о работе магазина (доставка, возврат, гарантия, контакты)
- multi — смешанный вопрос (и товары, и информация)

Правило: если нет конкретных характеристик (размер, цена, модель, процессор) — это catalog_browse.

2. SEARCH_QUERY — сформулируй оптимальный поисковый запрос для векторной базы:
- Убери сленг, исправь опечатки, раскрой сокращения
- Замени местоимения (их, это, они) на конкретные сущности из истории диалога
- Оставь только ключевые слова для поиска: название, бренд, категория, характеристики
- Для info-запросов: ключевые слова темы (возврат, доставка, гарантия)
- Максимум 10 слов

Примеры:
Вопрос: "пачом планшеты эпл на чипе м4?" → intent: products, search_query: "планшет Apple iPad M4 цена"
Вопрос: "телек 4к самсунг" → intent: products, search_query: "телевизор Samsung 4K"
Вопрос: "какие наушники есть?" → intent: catalog_browse, search_query: "наушники"
Вопрос: "как вернуть товар?" → intent: info, search_query: "возврат товара условия"
Вопрос: "покажи ноуты" → intent: catalog_browse, search_query: "ноутбук"
Вопрос: "скок они стоят?" (история: обсуждали AirPods Pro) → intent: products, search_query: "Apple AirPods Pro цена"

Ответь СТРОГО в формате JSON (без markdown-обёртки):
{"intent": "...", "search_query": "..."}"""

_VALID_INTENTS = {"products", "catalog_browse", "info", "multi"}


def analyze_query(question: str, history: list[dict] | None = None) -> dict:
    """
    Один LLM-вызов: определяет intent + формирует оптимальный поисковый запрос.
    Возвращает {"intent": str, "search_query": str}.
    При ошибке парсинга — fallback на multi + оригинальный вопрос.
    """
    messages = [{"role": "system", "content": _ANALYZE_PROMPT}]

    # Добавляем последние 4 сообщения истории для контекста местоимений
    if history:
        messages.extend(history[-4:])

    messages.append({"role": "user", "content": question})

    try:
        raw = _dispatch(messages, temperature=0.0, max_tokens=80)

        # Убираем возможную markdown-обёртку ```json ... ```
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]  # убираем первую строку ```json
            raw = raw.rsplit("```", 1)[0]  # убираем последний ```
            raw = raw.strip()

        import json
        data = json.loads(raw)

        intent = data.get("intent", "multi").strip().lower()
        search_query = data.get("search_query", "").strip()

        if intent not in _VALID_INTENTS:
            intent = "multi"
        if not search_query:
            search_query = question

        return {"intent": intent, "search_query": search_query}

    except Exception:
        # Fallback: если JSON не распарсился — используем оригинал
        return {"intent": "multi", "search_query": question}


# Обратная совместимость для тестов
def classify_intent(question: str) -> str:
    """Обёртка для обратной совместимости — возвращает только intent."""
    return analyze_query(question)["intent"]


def rewrite_query(question: str, history: list[dict]) -> str:
    """Обёртка для обратной совместимости — возвращает только search_query."""
    return analyze_query(question, history)["search_query"]
