import requests

product = [{
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "name": "Ноутбук ASUS VivoBook 15",
    "article": "NB-ASUS-001",
    "slug": "noutbuk-asus-vivobook-15",
    "description": "Мощный ноутбук для работы и учёбы с процессором Intel Core i7",
    "category": "Электроника",
    "category_slug": "elektronika",
    "subcategory": "Компьютеры",
    "subcategory_slug": "kompyutery",
    "type": "Ноутбуки",
    "type_slug": "noutbuki",
    "brand": "ASUS",
    "brand_slug": "asus",
    "price": 75000.0,
    "in_stock": True,
    "deleted": False,
    "attributes": {
        "Процессор": "Intel Core i7",
        "ОЗУ": "16 GB",
        "Накопитель": "512 GB SSD",
        "Диагональ": "15.6 дюймов",
        "Цвет": "Серебристый"
    },
    "images": [
        "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
        "a9b8c7d6-e5f4-3210-abcd-ef0987654321"
    ],
    "embedding_text": "Ноутбук ASUS VivoBook 15. Мощный ноутбук для работы и учёбы с процессором Intel Core i7. Категория: Электроника. Подкатегория: Компьютеры. Тип: Ноутбуки. Бренд: ASUS. Процессор: Intel Core i7. ОЗУ: 16 GB. Накопитель: 512 GB SSD. Диагональ: 15.6 дюймов. Цвет: Серебристый"
}]

r = requests.post("http://localhost:8000/products", json=product)
print("Status:", r.status_code)
print("Response:", r.json())
