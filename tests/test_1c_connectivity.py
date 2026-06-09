"""
Тесты связи с 1С — проверяем все известные эндпоинты.
Запускаются первыми, не зависят от LLM и Qdrant.

pytest tests/test_1c_connectivity.py -v
"""
import pytest
import httpx

from tests.conftest import ONEC_BASE, ONEC_AUTH


# ── Вспомогательная функция ───────────────────────────────────────────────────

def get(path: str, **params) -> httpx.Response:
    with httpx.Client(base_url=ONEC_BASE, auth=ONEC_AUTH, timeout=8.0) as c:
        return c.get(path, params=params if params else None)


def post(path: str, json_body: dict) -> httpx.Response:
    with httpx.Client(base_url=ONEC_BASE, auth=ONEC_AUTH, timeout=8.0) as c:
        return c.post(path, json=json_body)


# ── Базовое подключение ───────────────────────────────────────────────────────

class TestOnecConnectivity:

    def test_categories_endpoint_responds(self):
        """GET /categories должен отвечать 200."""
        resp = get("/categories")
        assert resp.status_code == 200, (
            f"1С /categories вернул {resp.status_code}:\n{resp.text[:300]}"
        )

    def test_categories_has_valid_structure(self):
        """Ответ /categories должен содержать ключ categories со списком."""
        resp = get("/categories")
        assert resp.status_code == 200
        data = resp.json()
        print(f"\n  Структура categories: {list(data.keys())}")
        assert "categories" in data, f"Нет ключа 'categories' в ответе: {list(data.keys())}"
        cats = data["categories"]
        assert isinstance(cats, list), f"categories должен быть списком, получили {type(cats)}"
        assert len(cats) > 0, "categories пустой — нет данных в 1С?"
        print(f"  Категорий верхнего уровня: {len(cats)}")
        for cat in cats:
            assert "name" in cat, f"Категория без name: {cat}"

    def test_categories_has_subcategories(self):
        """Категории должны содержать подкатегории с name и slug."""
        resp = get("/categories")
        data = resp.json()
        cats = data.get("categories", [])
        has_subs = any(cat.get("subcategories") for cat in cats)
        if not has_subs:
            pytest.skip("Подкатегорий нет — возможно данные ещё не загружены")
        for cat in cats:
            for sub in cat.get("subcategories", []):
                assert "name" in sub, f"Подкатегория без name: {sub}"
                print(f"  Подкатегория: {sub['name']} (slug={sub.get('slug', 'N/A')})")

    def test_categories_print_full_tree(self):
        """Выводим полное дерево категорий для отладки."""
        resp = get("/categories")
        data = resp.json()
        print("\n  === Дерево категорий 1С ===")
        for cat in data.get("categories", []):
            print(f"  [{cat.get('name')}]")
            for sub in cat.get("subcategories", []):
                slug = sub.get("slug", "")
                print(f"    - {sub.get('name')} (slug={slug})")
                for t in sub.get("types", []):
                    print(f"      · {t.get('name')} (slug={t.get('slug', '')})")
        assert True  # просто вывод, не падаем


# ── Availability ──────────────────────────────────────────────────────────────

class TestOnecAvailability:

    def test_availability_endpoint_exists(self):
        """GET /catalog/availability должен отвечать (200 или корректную ошибку)."""
        resp = get("/catalog/availability", ids="00000000-0000-0000-0000-000000000000")
        # 200 или 404 — оба означают что endpoint работает
        assert resp.status_code in (200, 204, 404), (
            f"Неожиданный статус: {resp.status_code}\n{resp.text[:300]}"
        )
        print(f"\n  /catalog/availability статус: {resp.status_code}")

    def test_availability_returns_list(self):
        """Ответ /catalog/availability должен быть JSON-массивом."""
        resp = get("/catalog/availability", ids="00000000-0000-0000-0000-000000000000")
        if resp.status_code == 404:
            pytest.skip("Endpoint не реализован")
        if resp.status_code == 204:
            return  # пустой — ок
        data = resp.json()
        print(f"\n  Ответ availability: {data}")
        assert isinstance(data, list), f"Ожидали список, получили {type(data)}: {data}"

    def test_availability_item_structure(self, onec_client):
        """Если availability вернул товары — проверяем структуру элементов."""
        # Сначала получаем хоть какой-нибудь UUID из каталога
        cats_resp = onec_client.get("/categories")
        if cats_resp.status_code != 200:
            pytest.skip("Нет доступа к /categories")

        # Пробуем получить реальный UUID через catalog/search если есть
        search_resp = onec_client.get("/catalog/search", params={"limit": 1})
        if search_resp.status_code != 200:
            pytest.skip("Endpoint /catalog/search не реализован — нет UUID для теста")

        items = search_resp.json().get("items", [])
        if not items:
            pytest.skip("Нет товаров в /catalog/search")

        real_uuid = items[0]["id"]
        resp = onec_client.get("/catalog/availability", params={"ids": real_uuid})
        assert resp.status_code == 200
        data = resp.json()
        print(f"\n  Availability для {real_uuid}: {data}")
        if data:
            item = data[0]
            assert "id" in item, f"Нет поля 'id': {item}"
            assert "price" in item or "inStock" in item, f"Нет price/inStock: {item}"


# ── Новые эндпоинты ───────────────────────────────────────────────────────────

class TestOnecNewEndpoints:
    """
    Проверяем эндпоинты которые мы добавили в onec_client.py.
    Если 1С их ещё не реализовала — тесты пропускаются с пояснением.
    """

    def test_catalog_search_endpoint(self):
        """GET /catalog/search — поиск по фильтрам."""
        resp = get("/catalog/search", limit=5)
        print(f"\n  /catalog/search статус: {resp.status_code}")
        if resp.status_code == 404:
            pytest.skip("❌ /catalog/search не реализован в 1С (нужно добавить)")
        assert resp.status_code == 200, f"Статус {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        print(f"  Ответ: {data}")
        assert "items" in data, f"Нет ключа 'items': {list(data.keys())}"
        print(f"  Товаров найдено: {len(data['items'])}")
        if data["items"]:
            item = data["items"][0]
            assert "id" in item, f"Нет 'id' у товара: {item}"
            assert "name" in item or "price" in item, f"Нет name/price: {item}"

    def test_catalog_search_with_brand_filter(self):
        """GET /catalog/search?brand=Apple — фильтр по бренду."""
        resp = get("/catalog/search", brand="Apple", limit=5)
        if resp.status_code == 404:
            pytest.skip("❌ /catalog/search не реализован")
        assert resp.status_code == 200
        data = resp.json()
        items = data.get("items", [])
        print(f"\n  Apple товаров: {len(items)}")
        for item in items:
            print(f"    {item.get('name', 'N/A')} — {item.get('price', 'N/A')} руб.")

    def test_catalog_search_with_price_filter(self):
        """GET /catalog/search?price_max=50000 — фильтр по цене."""
        resp = get("/catalog/search", price_max=50000, limit=5)
        if resp.status_code == 404:
            pytest.skip("❌ /catalog/search не реализован")
        assert resp.status_code == 200
        data = resp.json()
        items = data.get("items", [])
        print(f"\n  Товаров до 50 000 руб.: {len(items)}")
        for item in items:
            price = item.get("price", 0)
            print(f"    {item.get('name')} — {price} руб.")
            if price:
                assert price <= 50000, f"Фильтр не сработал: {item['name']} стоит {price}"

    def test_product_detail_endpoint(self, onec_client):
        """GET /catalog/product/{uuid} — детали товара."""
        # Сначала получаем UUID из search
        search_resp = onec_client.get("/catalog/search", params={"limit": 1})
        if search_resp.status_code == 404:
            pytest.skip("❌ /catalog/search не реализован — нет UUID для теста /catalog/product")
        items = search_resp.json().get("items", [])
        if not items:
            pytest.skip("Нет товаров в /catalog/search")

        uuid = items[0]["id"]
        resp = onec_client.get(f"/catalog/product/{uuid}")
        print(f"\n  /catalog/product/{uuid} статус: {resp.status_code}")
        if resp.status_code == 404:
            pytest.skip(f"❌ /catalog/product/{{uuid}} не реализован в 1С")
        assert resp.status_code == 200
        data = resp.json()
        print(f"  Данные товара: {list(data.keys())}")
        assert "id" in data
        assert "name" in data
        print(f"  Товар: {data['name']}, цена: {data.get('price')}, бренд: {data.get('brand')}")

    def test_compare_endpoint(self, onec_client):
        """POST /catalog/compare — сравнение двух товаров."""
        search_resp = onec_client.get("/catalog/search", params={"limit": 2})
        if search_resp.status_code == 404:
            pytest.skip("❌ /catalog/search не реализован — нет UUID для теста /catalog/compare")
        items = search_resp.json().get("items", [])
        if len(items) < 2:
            pytest.skip("Нет достаточно товаров для сравнения (нужно >= 2)")

        uuids = [items[0]["id"], items[1]["id"]]
        resp = onec_client.post("/catalog/compare", json={"ids": uuids})
        print(f"\n  /catalog/compare статус: {resp.status_code}")
        if resp.status_code == 404:
            pytest.skip("❌ /catalog/compare не реализован в 1С")
        assert resp.status_code == 200
        data = resp.json()
        print(f"  Ключи ответа: {list(data.keys())}")
        assert "products" in data, f"Нет 'products' в ответе: {data}"

    def test_order_status_endpoint(self, onec_client):
        """GET /orders/{number}/status — статус заказа."""
        # Пробуем с тестовым номером
        resp = onec_client.get("/orders/99999/status")
        print(f"\n  /orders/99999/status статус: {resp.status_code}")
        if resp.status_code == 404:
            # 404 может означать "заказ не найден" (ок) или "endpoint не существует"
            # Пробуем ещё раз с другим форматом
            resp2 = onec_client.get("/orders/1/status")
            if resp2.status_code == 404:
                pytest.skip("❌ /orders/{n}/status не реализован или нет тестового заказа")
            return
        if resp.status_code in (200, 404):
            print(f"  Ответ: {resp.text[:200]}")
            return
        pytest.fail(f"Неожиданный статус: {resp.status_code}: {resp.text[:200]}")

    def test_promotions_endpoint(self):
        """GET /promotions/active — активные акции."""
        resp = get("/promotions/active")
        print(f"\n  /promotions/active статус: {resp.status_code}")
        if resp.status_code == 404:
            pytest.skip("❌ /promotions/active не реализован в 1С")
        assert resp.status_code == 200
        data = resp.json()
        print(f"  Ответ: {data}")
        promotions = data.get("promotions", data) if isinstance(data, dict) else data
        print(f"  Акций найдено: {len(promotions)}")

    def test_recommendations_endpoint(self, onec_client):
        """GET /catalog/recommendations — рекомендации к товару."""
        search_resp = onec_client.get("/catalog/search", params={"limit": 1})
        if search_resp.status_code == 404:
            pytest.skip("❌ /catalog/search не реализован — нет UUID для теста")
        items = search_resp.json().get("items", [])
        if not items:
            pytest.skip("Нет товаров в /catalog/search")

        uuid = items[0]["id"]
        resp = onec_client.get("/catalog/recommendations", params={"product_id": uuid, "limit": 3})
        print(f"\n  /catalog/recommendations статус: {resp.status_code}")
        if resp.status_code == 404:
            pytest.skip("❌ /catalog/recommendations не реализован в 1С")
        assert resp.status_code == 200
        data = resp.json()
        items_rec = data.get("items", [])
        print(f"  Рекомендаций: {len(items_rec)}")
