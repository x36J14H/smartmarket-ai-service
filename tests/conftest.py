"""
Общие фикстуры и конфигурация pytest.

LLM  → LM Studio  http://192.168.56.1:1234/v1  google/gemma-4-12b
1С   → http://localhost:8081/smartmarket/hs/site-api  (Администратор / без пароля)
"""
import os
import json
import pytest
import httpx

# ── Переключаем llm_settings.json на LM Studio ────────────────────────────────
# Делаем это ДО любого импорта app.* чтобы settings подхватила нужные значения.

_LM_STUDIO_SETTINGS = {
    "active_provider": "openai",
    "active_model": "google/gemma-4-12b",
    "providers": {
        "openai": {
            "api_key": "lm-studio",
            "base_url": "http://192.168.56.1:1234/v1",
            "selected_model": "google/gemma-4-12b",
        }
    },
}

_SETTINGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "llm_settings.json"
)
_ORIGINAL_SETTINGS: dict | None = None


def pytest_configure(config):
    """Подменяем llm_settings.json перед запуском сессии."""
    global _ORIGINAL_SETTINGS
    os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
    if os.path.exists(_SETTINGS_PATH):
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            _ORIGINAL_SETTINGS = json.load(f)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(_LM_STUDIO_SETTINGS, f, ensure_ascii=False, indent=2)
    print(f"\n[conftest] LLM переключён на LM Studio: google/gemma-4-12b @ http://192.168.56.1:1234/v1")


def pytest_unconfigure(config):
    """Восстанавливаем оригинальный llm_settings.json после сессии."""
    if _ORIGINAL_SETTINGS is not None:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(_ORIGINAL_SETTINGS, f, ensure_ascii=False, indent=2)
        print(f"\n[conftest] llm_settings.json восстановлен")


# ── Константы ─────────────────────────────────────────────────────────────────

ONEC_BASE = "http://localhost:8081/smartmarket/hs/site-api"
ONEC_AUTH = ("Администратор", "")

LM_STUDIO_BASE = "http://192.168.56.1:1234/v1"
LM_STUDIO_MODEL = "google/gemma-4-12b"

SERVICE_BASE = "http://localhost:8000"


# ── Фикстуры ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def onec_client():
    """Синхронный httpx-клиент к 1С (Basic auth)."""
    with httpx.Client(base_url=ONEC_BASE, auth=ONEC_AUTH, timeout=10.0) as client:
        yield client


@pytest.fixture(scope="session")
def lm_client():
    """Синхронный httpx-клиент к LM Studio."""
    with httpx.Client(base_url=LM_STUDIO_BASE, timeout=60.0) as client:
        yield client


@pytest.fixture(scope="session")
def api_client():
    """Клиент к запущенному AI-сервису (должен быть запущен отдельно)."""
    with httpx.Client(base_url=SERVICE_BASE, timeout=60.0) as client:
        yield client


@pytest.fixture(scope="session")
def categories_data(onec_client):
    """Загружает каталог категорий из 1С один раз на всю сессию."""
    resp = onec_client.get("/categories")
    assert resp.status_code == 200, f"1С /categories вернул {resp.status_code}"
    return resp.json()


@pytest.fixture(scope="session")
def products_sample(onec_client):
    """
    Возвращает первые несколько UUID товаров через /catalog/availability
    используя тестовый UUID чтобы убедиться что endpoint работает.
    Если 1С не имеет /catalog/availability — пропускаем.
    """
    # Сначала пробуем любой известный UUID из availability
    test_uuid = "00000000-0000-0000-0000-000000000000"
    resp = onec_client.get("/catalog/availability", params={"ids": test_uuid})
    if resp.status_code == 404:
        pytest.skip("1С endpoint /catalog/availability не реализован")
    return resp
