from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from app.services.llm_settings import (
    load_llm_settings,
    save_provider_credentials,
    save_provider_model,
    set_active_provider,
    delete_llm_settings,
)
from app.core.config import settings as env_settings

router = APIRouter(prefix="/settings", tags=["settings"])

Provider = Literal["openai", "openrouter", "gigachat"]

# Порядок отображения провайдеров
ALL_PROVIDERS: list[str] = ["gigachat", "openai", "openrouter"]

# Дефолтные модели — когда ключ не настроен или запрос к API не удался
DEFAULT_MODELS: dict[str, list[str]] = {
    "gigachat":   ["GigaChat", "GigaChat-2", "GigaChat-Plus", "GigaChat-Pro", "GigaChat-2-Pro", "GigaChat-Max", "GigaChat-2-Max"],
    "openai":     ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "openrouter": [
        "openai/gpt-4o-mini", "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-flash-1.5",
        "meta-llama/llama-3.1-8b-instruct:free",
    ],
}

# base_url по умолчанию из .env (используется если не задан в настройках провайдера)
DEFAULT_BASE_URLS: dict[str, str | None] = {
    "openai":     env_settings.openai_base_url,
    "openrouter": env_settings.openrouter_base_url,
    "gigachat":   None,
}


# ── Схемы запросов ────────────────────────────────────────────────────────────

class ProviderCredentials(BaseModel):
    """Учётные данные провайдера."""
    api_key: str | None = None  # опционально — можно обновить только base_url
    gigachat_scope: str | None = None  # только для GigaChat
    # Только для OpenAI — кастомный base_url (LM Studio, прокси, и т.д.)
    # Если не указан — используется значение из .env
    base_url: str | None = None


class ProviderModelRequest(BaseModel):
    """Выбор модели для провайдера."""
    model: str


class ActiveProviderRequest(BaseModel):
    """Переключение активного провайдера."""
    provider: Provider
    model: str


class ValidateProviderRequest(BaseModel):
    """Проверка валидности ключа провайдера."""
    provider: Provider


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _base_url_for(provider: str, cfg: dict | None = None) -> str | None:
    """
    Вернуть base_url для провайдера.
    Приоритет: кастомный из настроек провайдера > дефолт из .env
    """
    if cfg and cfg.get("base_url"):
        return cfg["base_url"]
    return DEFAULT_BASE_URLS.get(provider)


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@router.get("/llm")
def get_llm_settings() -> dict:
    """
    Текущее состояние: активный провайдер и все провайдеры с их настройками.
    Для каждого провайдера: настроен ли ключ, выбранная модель, актуальный base_url.
    """
    data = load_llm_settings()
    configured = data.get("providers", {})
    providers_info = {}

    for name in ALL_PROVIDERS:
        cfg = configured.get(name, {})
        has_key = bool(cfg.get("api_key"))
        entry: dict = {
            "key_configured": has_key,
            "api_key_hint":   _mask_key(cfg["api_key"]) if has_key else None,
            "selected_model": cfg.get("selected_model"),
            "base_url":       _base_url_for(name, cfg),
        }
        if cfg.get("gigachat_scope"):
            entry["gigachat_scope"] = cfg["gigachat_scope"]
        # Показываем флаг что base_url кастомный (не дефолтный)
        if name == "openai" and cfg.get("base_url"):
            entry["base_url_custom"] = True
        providers_info[name] = entry

    active = data.get("active_provider")
    return {
        "source":          "file" if active else "env",
        "active_provider": active or env_settings.llm_provider,
        "active_model":    data.get("active_model") or env_settings.llm_model,
        "providers":       providers_info,
    }


@router.post("/llm/provider/{provider}")
def upsert_provider(provider: Provider, req: ProviderCredentials) -> dict:
    """
    Сохранить/обновить настройки провайдера.
    Можно передать только api_key, только base_url, или оба поля сразу.
    Не переданные поля не затираются.
    """
    if not req.api_key and not req.base_url and not req.gigachat_scope:
        raise HTTPException(status_code=400, detail="Необходимо передать хотя бы одно поле: api_key, base_url или gigachat_scope")

    credentials: dict = {}
    if req.api_key:
        credentials["api_key"] = req.api_key
    if req.gigachat_scope:
        credentials["gigachat_scope"] = req.gigachat_scope
    if req.base_url and provider == "openai":
        credentials["base_url"] = req.base_url
    save_provider_credentials(provider, credentials)

    data = load_llm_settings()
    cfg = data.get("providers", {}).get(provider, {})
    return {
        "saved":    True,
        "provider": provider,
        "base_url": _base_url_for(provider, cfg),
    }


@router.post("/llm/provider/{provider}/model")
def set_provider_model(provider: Provider, req: ProviderModelRequest) -> dict:
    """
    Сохранить выбранную модель для провайдера.
    Не меняет активный провайдер — запоминает выбор для последующего переключения.
    """
    save_provider_model(provider, req.model)
    return {
        "saved":    True,
        "provider": provider,
        "model":    req.model,
    }


@router.post("/llm/active")
def switch_active_provider(req: ActiveProviderRequest) -> dict:
    """
    Переключить активный провайдер и модель.
    Провайдер должен быть предварительно настроен (иметь ключ).
    Также обновляет selected_model для этого провайдера.
    """
    data = load_llm_settings()
    configured = data.get("providers", {})

    if req.provider not in configured or not configured[req.provider].get("api_key"):
        raise HTTPException(
            status_code=400,
            detail=f"Провайдер '{req.provider}' не настроен. "
                   f"Сначала отправьте ключ: POST /api/v1/settings/llm/provider/{req.provider}",
        )

    set_active_provider(req.provider, req.model)
    cfg = configured[req.provider]
    return {
        "active_provider": req.provider,
        "active_model":    req.model,
        "base_url":        _base_url_for(req.provider, cfg),
    }


@router.delete("/llm")
def reset_llm_settings() -> dict:
    """Сбросить все настройки — удаляет файл, сервис возвращается к значениям из .env."""
    deleted = delete_llm_settings()
    return {
        "reset":   deleted,
        "message": "Настройки сброшены, используются значения из .env" if deleted
                   else "Файл настроек не найден, уже используется .env",
    }


@router.post("/llm/validate")
def validate_provider(req: ValidateProviderRequest) -> dict:
    """
    Проверить валидность ключа — делает минимальный тестовый запрос к провайдеру.
    Всегда возвращает HTTP 200; результат в поле valid.
    """
    data = load_llm_settings()
    cfg = data.get("providers", {}).get(req.provider)

    if not cfg or not cfg.get("api_key"):
        raise HTTPException(
            status_code=400,
            detail=f"Провайдер '{req.provider}' не настроен. "
                   f"Сначала отправьте ключ: POST /api/v1/settings/llm/provider/{req.provider}",
        )

    try:
        if req.provider == "gigachat":
            _validate_gigachat(cfg)
        else:
            _validate_openai_compatible(req.provider, cfg)
    except Exception as e:
        return {"valid": False, "provider": req.provider, "error": str(e)}

    return {"valid": True, "provider": req.provider, "error": None}


@router.get("/llm/models/{provider}")
def get_provider_models(provider: Provider) -> dict:
    """
    Получить список моделей провайдера.
    Если ключ настроен — запрашивает реальный список у провайдера.
    Если ключ не настроен или запрос не удался — возвращает встроенный дефолтный список.
    GigaChat всегда возвращает дефолт (их API не поддерживает список моделей).
    """
    data = load_llm_settings()
    cfg = data.get("providers", {}).get(provider, {})
    selected = cfg.get("selected_model")

    if not cfg.get("api_key"):
        return {
            "provider":       provider,
            "source":         "default",
            "selected_model": selected,
            "models":         DEFAULT_MODELS.get(provider, []),
        }

    if provider == "gigachat":
        try:
            models = _fetch_models_gigachat(cfg)
            return {
                "provider":       provider,
                "source":         "api",
                "selected_model": selected,
                "models":         models,
            }
        except Exception:
            return {
                "provider":       provider,
                "source":         "default",
                "selected_model": selected,
                "models":         DEFAULT_MODELS.get(provider, []),
            }

    try:
        models = _fetch_models_openai_compatible(provider, cfg)
        return {
            "provider":       provider,
            "source":         "api",
            "selected_model": selected,
            "models":         models,
        }
    except Exception:
        return {
            "provider":       provider,
            "source":         "default",
            "selected_model": selected,
            "models":         DEFAULT_MODELS.get(provider, []),
        }


# ── Внутренние функции ────────────────────────────────────────────────────────

def _validate_openai_compatible(provider: str, cfg: dict) -> None:
    """Проверка OpenAI / OpenRouter — способ зависит от провайдера."""
    from openai import OpenAI
    client = OpenAI(api_key=cfg["api_key"], base_url=_base_url_for(provider, cfg))

    if provider == "openrouter":
        # OpenRouter отдаёт models.list() без авторизации — проверяем через chat
        client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
    else:
        # OpenAI и совместимые — список моделей достаточен
        client.models.list()


def _validate_gigachat(cfg: dict) -> None:
    """Проверка GigaChat — получаем OAuth-токен."""
    from gigachat import GigaChat
    scope = cfg.get("gigachat_scope") or "GIGACHAT_API_PERS"
    with GigaChat(credentials=cfg["api_key"], scope=scope, verify_ssl_certs=False) as giga:
        giga.get_token()


def _fetch_models_gigachat(cfg: dict) -> list[str]:
    """Получить список chat-моделей GigaChat через SDK."""
    from gigachat import GigaChat
    scope = cfg.get("gigachat_scope") or "GIGACHAT_API_PERS"
    with GigaChat(credentials=cfg["api_key"], scope=scope, verify_ssl_certs=False) as giga:
        response = giga.get_models()
    # SDK не возвращает поле type — исключаем embedding-модели по имени
    return [
        m.id_ for m in response.data
        if "embed" not in m.id_.lower()
    ]


def _fetch_models_openai_compatible(provider: str, cfg: dict) -> list[str]:
    """Получить список моделей через OpenAI-совместимый API."""
    from openai import OpenAI
    client = OpenAI(api_key=cfg["api_key"], base_url=_base_url_for(provider, cfg))
    response = client.models.list()
    model_ids = [m.id for m in response.data]

    if provider == "openrouter":
        model_ids = [
            m for m in model_ids
            if not any(s in m for s in (":embed", "moderation", "whisper", "tts", "dall-e", "sora"))
        ]
        model_ids.sort()

    return model_ids
