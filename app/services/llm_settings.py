"""
Хранение настроек LLM в JSON-файле.
Структура файла:
{
  "active_provider": "gigachat",
  "active_model": "GigaChat-2",
  "providers": {
    "gigachat": {
      "api_key": "...",
      "gigachat_scope": "GIGACHAT_API_PERS",
      "selected_model": "GigaChat-2"
    },
    "openai": {
      "api_key": "sk-...",
      "base_url": "https://api.openai.com/v1",  // опционально — кастомный эндпоинт
      "selected_model": "gpt-4o-mini"
    },
    "openrouter": {
      "api_key": "sk-or-...",
      "selected_model": "openai/gpt-4o-mini"
    }
  }
}
"""
import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "llm_settings.json"


def _ensure_data_dir() -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_llm_settings() -> dict:
    """Загрузить весь файл настроек. Если нет — вернуть пустой dict."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    _ensure_data_dir()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_active_provider_cfg() -> dict | None:
    """
    Вернуть конфиг активного провайдера: {api_key, base_url?, gigachat_scope?, model}
    Возвращает None если файла нет или активный провайдер не настроен.
    """
    data = load_llm_settings()
    provider = data.get("active_provider")
    if not provider:
        return None
    cfg = data.get("providers", {}).get(provider)
    if not cfg:
        return None
    return {
        "provider": provider,
        "model":    data.get("active_model"),
        **cfg,
    }


def save_provider_credentials(provider: str, credentials: dict) -> None:
    """Сохранить/обновить учётные данные провайдера, не трогая selected_model."""
    data = load_llm_settings()
    if "providers" not in data:
        data["providers"] = {}
    existing = data["providers"].get(provider, {})
    updated = {**existing, **credentials}
    data["providers"][provider] = updated
    _save(data)


def save_provider_model(provider: str, model: str) -> None:
    """Сохранить выбранную модель для провайдера (не меняет активный провайдер)."""
    data = load_llm_settings()
    if "providers" not in data:
        data["providers"] = {}
    if provider not in data["providers"]:
        data["providers"][provider] = {}
    data["providers"][provider]["selected_model"] = model
    _save(data)


def set_active_provider(provider: str, model: str) -> None:
    """Переключить активный провайдер и модель. Также обновляет selected_model провайдера."""
    data = load_llm_settings()
    data["active_provider"] = provider
    data["active_model"] = model
    if "providers" not in data:
        data["providers"] = {}
    if provider not in data["providers"]:
        data["providers"][provider] = {}
    data["providers"][provider]["selected_model"] = model
    _save(data)


def delete_llm_settings() -> bool:
    """Удалить файл настроек (сброс на дефолт из .env)."""
    if SETTINGS_PATH.exists():
        SETTINGS_PATH.unlink()
        return True
    return False
