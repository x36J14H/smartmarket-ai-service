"""
Тесты onec_client.py — реальные HTTP-запросы к 1С + юнит-тесты TTL-кэша.

Реальные HTTP-тесты помечены @pytest.mark.integration — запускаются только
если 1С доступна. Юнит-тесты кэша и graceful degradation работают всегда.

pytest tests/test_onec_client.py -v                        # только юниты
pytest tests/test_onec_client.py -v -m integration        # + реальная 1С
"""
import time
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from tests.conftest import ONEC_BASE, ONEC_AUTH


# ── TTL-кэш ───────────────────────────────────────────────────────────────────

class TestTtlCache:

    def _make_cache(self, ttl: float = 10.0):
        from app.services.onec_client import _TtlCache
        return _TtlCache(ttl=ttl)

    def test_initially_empty(self):
        cache = self._make_cache()
        assert cache.get() is None

    def test_set_and_get(self):
        cache = self._make_cache()
        cache.set({"key": "value"})
        assert cache.get() == {"key": "value"}

    def test_list_data(self):
        cache = self._make_cache()
        cache.set([1, 2, 3])
        assert cache.get() == [1, 2, 3]

    def test_expired_returns_none(self):
        cache = self._make_cache(ttl=0.05)  # 50ms
        cache.set("data")
        assert cache.get() == "data"
        time.sleep(0.1)
        assert cache.get() is None, "Данные должны истечь после TTL"

    def test_invalidate(self):
        cache = self._make_cache()
        cache.set("data")
        assert cache.get() == "data"
        cache.invalidate()
        assert cache.get() is None

    def test_overwrite(self):
        cache = self._make_cache()
        cache.set("first")
        cache.set("second")
        assert cache.get() == "second"

    def test_zero_cache_not_returned(self):
        """Нулевые значения (0, False, "") кэшируются корректно."""
        cache = self._make_cache()
        cache.set(0)
        # 0 — это данные, не None → должен вернуться
        # Но наш get() проверяет `if self._data is not None`
        # Поэтому 0 возвращается корректно
        assert cache.get() == 0

    def test_empty_list_not_cached_as_miss(self):
        """Пустой список — данные, не None."""
        cache = self._make_cache()
        cache.set([])
        # [] is not None → должен вернуться
        result = cache.get()
        assert result == []


# ── Graceful degradation — 1С не настроена ────────────────────────────────────

class TestOnecClientNotConfigured:
    """
    Когда ONEC_BASE_URL пустой — все функции должны деградировать gracefully.
    Мокируем settings.onec_base_url = "".
    """

    @pytest.mark.asyncio
    async def test_fetch_availability_returns_empty(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.fetch_availability(["uuid-1", "uuid-2"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_availability_empty_ids(self):
        from app.services import onec_client
        # Пустой список → всегда пустой dict, независимо от конфига
        result = await onec_client.fetch_availability([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_filter_available_ids_returns_all_if_not_configured(self):
        from app.services import onec_client
        ids = ["a", "b", "c"]
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.filter_available_ids(ids)
        assert result == set(ids)

    @pytest.mark.asyncio
    async def test_fetch_categories_returns_none(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.fetch_categories()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_product_returns_none(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.fetch_product("any-uuid")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_catalog_search_returns_empty(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.fetch_catalog_search(brand="Apple")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_compare_returns_none(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.fetch_compare(["uuid-1", "uuid-2"])
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_compare_needs_two_ids(self):
        """fetch_compare с одним id → None без HTTP-запроса."""
        from app.services import onec_client
        result = await onec_client.fetch_compare(["only-one"])
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_order_status_returns_none(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.fetch_order_status("12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_promotions_returns_empty(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.fetch_promotions()
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_recommendations_returns_empty(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=False):
            result = await onec_client.fetch_recommendations("any-uuid")
        assert result == []


# ── Graceful degradation — HTTP ошибки ───────────────────────────────────────

class TestOnecClientHttpErrors:
    """Симулируем сетевые ошибки через mock httpx."""

    @pytest.mark.asyncio
    async def test_availability_connection_error_returns_empty(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch("app.services.onec_client.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                httpx.ConnectError("refused")
            )
            result = await onec_client.fetch_availability(["uuid-1"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_categories_timeout_returns_none(self):
        from app.services import onec_client
        # Сначала сбрасываем кэш
        onec_client._categories_cache.invalidate()
        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch("app.services.onec_client.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                httpx.TimeoutException("timeout")
            )
            result = await onec_client.fetch_categories()
        assert result is None

    @pytest.mark.asyncio
    async def test_order_404_returns_none(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch("app.services.onec_client.httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp
            result = await onec_client.fetch_order_status("99999")
        assert result is None

    @pytest.mark.asyncio
    async def test_product_404_returns_none(self):
        from app.services import onec_client
        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch("app.services.onec_client.httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp
            result = await onec_client.fetch_product("unknown-uuid")
        assert result is None

    @pytest.mark.asyncio
    async def test_filter_available_ids_degrades_if_availability_empty(self):
        """Если fetch_availability вернул {} из-за ошибки — возвращаем все IDs."""
        from app.services import onec_client
        ids = ["a", "b", "c"]
        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch.object(onec_client, "fetch_availability", AsyncMock(return_value={})):
            result = await onec_client.filter_available_ids(ids)
        assert result == set(ids)


# ── Кэширование categories и promotions ──────────────────────────────────────

class TestOnecClientCaching:

    @pytest.mark.asyncio
    async def test_categories_cached_on_second_call(self):
        """Второй вызов fetch_categories не делает HTTP-запрос."""
        from app.services import onec_client
        onec_client._categories_cache.invalidate()

        fake_data = {"categories": [{"name": "Смартфоны", "slug": "phones", "subcategories": []}]}

        call_count = 0

        async def mock_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=fake_data)
            return resp

        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch("app.services.onec_client.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = mock_get

            r1 = await onec_client.fetch_categories()
            r2 = await onec_client.fetch_categories()

        assert r1 == fake_data
        assert r2 == fake_data
        assert call_count == 1, f"HTTP-запрос сделан {call_count} раз, ожидали 1 (кэш)"

    @pytest.mark.asyncio
    async def test_promotions_cached(self):
        """Второй вызов fetch_promotions не делает HTTP-запрос."""
        from app.services import onec_client
        onec_client._promotions_cache.invalidate()

        fake_promos = [{"title": "Акция 1", "discount_percent": 10}]
        call_count = 0

        async def mock_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value={"promotions": fake_promos})
            return resp

        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch("app.services.onec_client.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = mock_get

            r1 = await onec_client.fetch_promotions()
            r2 = await onec_client.fetch_promotions()

        assert r1 == fake_promos
        assert r2 == fake_promos
        assert call_count == 1, f"HTTP-запрос сделан {call_count} раз (ожидали 1)"


# ── Парсинг availability ──────────────────────────────────────────────────────

class TestAvailabilityParsing:

    @pytest.mark.asyncio
    async def test_filters_zero_price(self):
        """Товар с price=0 не попадает в результат."""
        from app.services import onec_client
        fake_response = [
            {"id": "uuid-1", "price": 10000.0, "inStock": 5},
            {"id": "uuid-2", "price": 0, "inStock": 3},      # нет цены
            {"id": "uuid-3", "price": 5000.0, "inStock": 0}, # нет в наличии
        ]

        async def mock_get(*a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=fake_response)
            return resp

        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch("app.services.onec_client.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = mock_get
            result = await onec_client.fetch_availability(["uuid-1", "uuid-2", "uuid-3"])

        assert "uuid-1" in result, "uuid-1 должен быть (цена и сток ненулевые)"
        assert "uuid-2" not in result, "uuid-2 не должен быть (price=0)"
        assert "uuid-3" not in result, "uuid-3 не должен быть (inStock=0)"

    @pytest.mark.asyncio
    async def test_batch_splitting(self):
        """Большой список UUID разбивается на батчи по 200."""
        from app.services import onec_client

        call_params = []

        async def mock_get(url, params=None, **kw):
            call_params.append(params)
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=[])
            return resp

        uuids = [f"uuid-{i:04d}" for i in range(450)]  # 450 UUID → 3 батча

        with patch.object(onec_client, "_is_configured", return_value=True), \
             patch("app.services.onec_client.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = mock_get
            await onec_client.fetch_availability(uuids)

        assert len(call_params) == 3, (
            f"Ожидали 3 HTTP-запроса для 450 UUID, получили {len(call_params)}"
        )
        # Первый батч — 200 UUID
        ids_batch1 = call_params[0]["ids"].split(",")
        assert len(ids_batch1) == 200
        # Последний батч — 50 UUID
        ids_batch3 = call_params[2]["ids"].split(",")
        assert len(ids_batch3) == 50


# ── Интеграционные тесты с реальной 1С ───────────────────────────────────────

@pytest.mark.integration
class TestOnecClientIntegration:
    """
    Реальные запросы к 1С. Запускаются с флагом -m integration.
    Требуют доступности 1С на localhost:8081.
    """

    @pytest.mark.asyncio
    async def test_fetch_categories_real(self):
        from app.services.onec_client import fetch_categories, _categories_cache
        _categories_cache.invalidate()
        data = await fetch_categories()
        print(f"\n  Реальный fetch_categories: {type(data)}")
        if data is None:
            pytest.skip("1С недоступна")
        assert "categories" in data
        cats = data["categories"]
        assert isinstance(cats, list)
        print(f"  Категорий: {len(cats)}")

    @pytest.mark.asyncio
    async def test_fetch_availability_real(self, onec_client):
        """Тест с реальным UUID из каталога."""
        from app.services.onec_client import fetch_availability

        # Получаем реальный UUID из 1С
        resp = onec_client.get("/catalog/search", params={"limit": 1})
        if resp.status_code == 404:
            pytest.skip("❌ /catalog/search не реализован")
        items = resp.json().get("items", [])
        if not items:
            pytest.skip("Нет товаров в /catalog/search")

        real_uuid = items[0]["id"]
        result = await fetch_availability([real_uuid])
        print(f"\n  Availability для {real_uuid}: {result}")
        # Может быть пустым если товар без цены/остатка — это валидно
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_fetch_promotions_real(self):
        from app.services.onec_client import fetch_promotions, _promotions_cache
        _promotions_cache.invalidate()
        result = await fetch_promotions()
        print(f"\n  Реальный fetch_promotions: {result}")
        if result:
            assert isinstance(result, list)
            for p in result:
                assert "title" in p or "description" in p
        else:
            print("  (акций нет или endpoint не реализован)")
