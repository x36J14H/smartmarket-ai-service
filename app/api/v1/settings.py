from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from app.services.llm_settings import (
    load_llm_settings,
    save_provider_credentials,
    set_active_provider,
    delete_llm_settings,
)
from app.core.config import settings as env_settings

router = APIRouter(prefix="/settings", tags=["settings"])

Provider = Literal["openai", "openrouter", "lmstudio", "gigachat"]


# ── Схемы запросов ────────────────────────────────────────────────────────────

class ProviderCredentials(BaseModel):
    """Учётные данные одного провайдера. Отправляются один раз и хранятся."""
    api_key: str
    # Только для GigaChat: GIGACHAT_API_PERS | GIGACHAT_API_B2B | GIGACHAT_API_CORP
    gigachat_scope: str | None = None


class ActiveProviderRequest(BaseModel):
    """Переключение активного провайдера. API-ключ не нужен."""
    provider: Provider
    model: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _base_url_for(provider: str) -> str | None:
    return {
        "openai":     env_settings.openai_base_url,
        "openrouter": env_settings.openrouter_base_url,
        "lmstudio":   env_settings.lmstudio_base_url,
        "gigachat":   None,
    }.get(provider)


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@router.get("/llm")
def get_llm_settings() -> dict:
    """Текущее состояние: активный провайдер и список настроенных провайдеров."""
    data = load_llm_settings()
    providers_info = {}

    for name, cfg in data.get("providers", {}).items():
        providers_info[name] = {
            "configured":   True,
            "api_key_hint": _mask_key(cfg.get("api_key", "")),
            "base_url":     _base_url_for(name),
        }
        if cfg.get("gigachat_scope"):
            providers_info[name]["gigachat_scope"] = cfg["gigachat_scope"]

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
    Сохранить/обновить учётные данные провайдера.
    Вызывается один раз при первоначальной настройке или смене ключа.
    """
    credentials = {"api_key": req.api_key}
    if req.gigachat_scope:
        credentials["gigachat_scope"] = req.gigachat_scope
    save_provider_credentials(provider, credentials)
    return {
        "saved":    True,
        "provider": provider,
        "base_url": _base_url_for(provider),
    }


@router.post("/llm/active")
def switch_active_provider(req: ActiveProviderRequest) -> dict:
    """
    Переключить активный провайдер и модель.
    Провайдер должен быть предварительно настроен через /llm/provider/{provider}.
    """
    data = load_llm_settings()
    configured = data.get("providers", {})

    if req.provider not in configured:
        raise HTTPException(
            status_code=400,
            detail=f"Провайдер '{req.provider}' не настроен. "
                   f"Сначала отправьте ключ на POST /api/v1/settings/llm/provider/{req.provider}",
        )

    set_active_provider(req.provider, req.model)
    return {
        "active_provider": req.provider,
        "active_model":    req.model,
        "base_url":        _base_url_for(req.provider),
    }


@router.delete("/llm")
def reset_llm_settings() -> dict:
    """Сбросить все настройки на дефолт из .env."""
    deleted = delete_llm_settings()
    return {
        "reset":   deleted,
        "message": "Настройки сброшены, используются значения из .env" if deleted
                   else "Файл настроек не найден, уже используется .env",
    }
