# Интеграция AI-сервиса с 1С

## Уже есть в 1С
- Константа `AIServiceBaseURL` — адрес сервиса, например `http://localhost:8000`
- Модуль `ВекторнаяБДИнтеграция` — отправка товаров

---

## Справочник по API — что и куда слать

### Проверка соединения
```
GET /health
```
Ответ: `{"status": "ok"}` — сервис работает.

---

### Товары — POST /api/v1/products

Массив объектов. Обязательные поля: `id`, `name`, `embedding_text`.
Остальные — опциональные, хранятся в векторной БД как метаданные.

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Смартфон Samsung Galaxy S24",
    "price": 89990.00,
    "embedding_text": "Смартфон Samsung Galaxy S24. Бренд: Samsung. Категория: Смартфоны. Флагманский смартфон с AI-функциями. Память: 256 ГБ. Цена: 89990 руб."
  }
]
```

**Удаление товара** — передать тот же объект с `"deleted": true`, или DELETE-запрос:
```
DELETE /api/v1/products/550e8400-e29b-41d4-a716-446655440000
```

**Поле `embedding_text`** — самое важное. Именно по нему идёт поиск. Формировать в 1С: собрать название, бренд, категорию, описание, характеристики в одну строку через точку или перенос строки. Чем полнее — тем лучше поиск.

**Ответ:**
```json
{"inserted": 1, "skipped": 0}
```

---

### FAQ — POST /api/v1/faq

Массив объектов. Обязательные: `id`, `question`, `answer`.

```json
[
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "question": "Как оформить возврат товара?",
    "answer": "Возврат оформляется в течение 14 дней с момента покупки. Перейдите в раздел «Мои заказы», выберите нужный заказ и нажмите «Оформить возврат». Товар должен быть в оригинальной упаковке.",
    "category": "Возвраты",
    "deleted": false
  },
  {
    "id": "660e8400-e29b-41d4-a716-446655440002",
    "question": "Сколько стоит доставка?",
    "answer": "Доставка по городу — 300 руб., при заказе от 5000 руб. — бесплатно. Доставка в регионы — от 500 руб., срок 3-7 дней.",
    "category": "Доставка",
    "deleted": false
  }
]
```

Сервис сам объединяет `question` + `answer` для векторизации — отдельный `embedding_text` не нужен.

**Ответ:**
```json
{"inserted": 2, "deleted": 0, "skipped": 0}
```

---

### Навигация — POST /api/v1/navigation

Массив объектов. Обязательные: `id`, `title`, `description`.

```json
[
  {
    "id": "770e8400-e29b-41d4-a716-446655440001",
    "title": "Каталог смартфонов",
    "description": "Раздел с полным каталогом смартфонов. Здесь можно выбрать смартфон по бренду, цене, характеристикам. Доступна фильтрация и сравнение моделей.",
    "url": "/catalog/smartphones",
    "category": "Каталог",
    "deleted": false
  },
  {
    "id": "770e8400-e29b-41d4-a716-446655440002",
    "title": "Корзина и оформление заказа",
    "description": "Страница корзины. Здесь можно проверить выбранные товары, применить промокод, выбрать способ доставки и оплаты, оформить заказ.",
    "url": "/cart",
    "category": "Сервис",
    "deleted": false
  }
]
```

**Описание** — пишется вручную менеджером. Чем подробнее описано что можно сделать на странице — тем точнее бот направит покупателя.

**Ответ:**
```json
{"inserted": 2, "deleted": 0, "skipped": 0}
```

---

### Настройки LLM

#### Сохранить ключ провайдера — POST /api/v1/settings/llm/provider/{provider}

Провайдер в URL: `openai`, `openrouter`, `lmstudio`, `gigachat`

Для OpenAI / OpenRouter / LM Studio:
```json
{
  "api_key": "sk-..."
}
```

Для GigaChat (дополнительно нужен scope):
```json
{
  "api_key": "base64_credentials_из_личного_кабинета_сбера",
  "gigachat_scope": "GIGACHAT_API_PERS"
}
```

Значения `gigachat_scope`:
- `GIGACHAT_API_PERS` — физические лица
- `GIGACHAT_API_B2B` — бизнес
- `GIGACHAT_API_CORP` — корпоративный

**Ответ:**
```json
{"saved": true, "provider": "gigachat", "base_url": null}
```

---

#### Переключить активный провайдер — POST /api/v1/settings/llm/active

Ключ не нужен — провайдер должен быть уже настроен выше.

```json
{
  "provider": "gigachat",
  "model": "GigaChat"
}
```

Доступные модели по провайдерам:
- GigaChat: `GigaChat`, `GigaChat-Pro`, `GigaChat-Max`
- OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`
- OpenRouter: любая строка из каталога openrouter.ai
- LM Studio: название модели как показано в интерфейсе LM Studio

**Ответ:**
```json
{"active_provider": "gigachat", "active_model": "GigaChat", "base_url": null}
```

---

#### Получить текущие настройки — GET /api/v1/settings/llm

```json
{
  "source": "file",
  "active_provider": "gigachat",
  "active_model": "GigaChat",
  "providers": {
    "gigachat": {
      "configured": true,
      "api_key_hint": "NjM4...ZjRk",
      "base_url": null,
      "gigachat_scope": "GIGACHAT_API_PERS"
    },
    "openai": {
      "configured": true,
      "api_key_hint": "sk-p...ef12",
      "base_url": "https://api.openai.com/v1"
    }
  }
}
```

`source: "env"` — настройки берутся из конфига сервиса (файл настроек не создан).
`source: "file"` — настройки из файла, переданного через API.

---

#### Сбросить настройки — DELETE /api/v1/settings/llm

Удаляет файл настроек, сервис возвращается к значениям из своего конфига.

---

### Чат — POST /api/v1/chat

Используется фронтендом (Next.js), но можно вызывать и из 1С для тестирования.

```json
{
  "question": "Есть ли у вас наушники Sony?",
  "session_id": "550e8400-e29b-41d4-a716-446655440099"
}
```

`session_id` — опциональный. Если не передать — сервис создаст новый. Передавать один и тот же ID для сохранения истории диалога.

**Ответ:**
```json
{
  "answer": "Да, у нас есть наушники Sony. Например, [Sony WH-1000XM5](/products/550e8400-...) — накладные с шумоподавлением, цена 29 990 руб.",
  "sources": [
    {
      "collection": "products",
      "score": 0.921,
      "text": "Наушники Sony WH-1000XM5. Бренд: Sony...",
      "product_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  ],
  "session_id": "550e8400-e29b-41d4-a716-446655440099"
}
```

`sources` — список источников из которых бот взял информацию. Полезно для отладки.

---

## Порядок первого запуска

1. Убедиться что сервис доступен: `GET /health`
2. Отправить ключ провайдера: `POST /api/v1/settings/llm/provider/gigachat`
3. Установить активный провайдер: `POST /api/v1/settings/llm/active`
4. Загрузить товары: `POST /api/v1/products`
5. Загрузить FAQ: `POST /api/v1/faq`
6. Загрузить навигацию: `POST /api/v1/navigation`
7. Проверить чат: `POST /api/v1/chat`
