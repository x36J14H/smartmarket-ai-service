# Интеграция AI-сервиса с 1С

## Уже есть в 1С
- Константа `AIServiceBaseURL` — адрес сервиса, например `http://localhost:8000`
- Модуль `ВекторнаяБДИнтеграция` — отправка товаров

---

## Справочник по API

### Проверка соединения

```
GET /health
```
```json
{"status": "ok"}
```

---

### Товары — POST /api/v1/products

Массив объектов. Обязательные поля: `id`, `name`, `embedding_text`.

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

**Поле `embedding_text`** — самое важное, именно по нему идёт поиск. Собирать в 1С: название + бренд + категория + описание + характеристики в одну строку. Чем полнее — тем лучше поиск.

**Удаление** — передать объект с `"deleted": true` или DELETE-запрос:
```
DELETE /api/v1/products/550e8400-e29b-41d4-a716-446655440000
```

```json
{"inserted": 1, "skipped": 0}
```

---

### FAQ — POST /api/v1/faq

Обязательные поля: `id`, `question`, `answer`. Сервис сам объединяет их для векторизации.

```json
[
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "question": "Как оформить возврат товара?",
    "answer": "Возврат оформляется в течение 14 дней. Перейдите в «Мои заказы», выберите заказ, нажмите «Оформить возврат».",
    "category": "Возвраты",
    "deleted": false
  }
]
```

```json
{"inserted": 1, "deleted": 0, "skipped": 0}
```

---

### Навигация — POST /api/v1/navigation

Обязательные поля: `id`, `title`, `description`.

```json
[
  {
    "id": "770e8400-e29b-41d4-a716-446655440001",
    "title": "Каталог смартфонов",
    "description": "Раздел с полным каталогом смартфонов. Выбор по бренду, цене, характеристикам. Фильтрация и сравнение моделей.",
    "url": "/catalog/smartphones",
    "category": "Каталог",
    "deleted": false
  }
]
```

**Описание** пишется вручную менеджером. Чем подробнее — тем точнее бот направит покупателя.

```json
{"inserted": 1, "deleted": 0, "skipped": 0}
```

---

### Чат — POST /api/v1/chat

```json
{
  "question": "Есть ли у вас наушники Sony?",
  "session_id": "550e8400-e29b-41d4-a716-446655440099"
}
```

`session_id` — опциональный. Передавать один и тот же ID для сохранения истории диалога.

```json
{
  "answer": "Да, у нас есть наушники Sony WH-1000XM5 — накладные с шумоподавлением, цена 29 990 руб.",
  "sources": [
    {
      "collection": "products",
      "score": 0.921,
      "text": "Наушники Sony WH-1000XM5...",
      "product_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  ],
  "session_id": "550e8400-e29b-41d4-a716-446655440099"
}
```

---

## Настройки LLM

### Провайдеры

Поддерживаются три провайдера:

| Провайдер | Описание |
|-----------|----------|
| `gigachat` | GigaChat от Сбера. Работает из России без VPN. |
| `openai` | OpenAI API. Поддерживает кастомный `base_url` — можно подключить LM Studio, прокси или любой OpenAI-совместимый сервер. |
| `openrouter` | OpenRouter — агрегатор моделей (GPT, Claude, Gemini и др.). Работает из России. |

### Концепция хранения

Каждый провайдер хранит независимо:
- **ключ** (`api_key`) — сохраняется один раз
- **выбранную модель** (`selected_model`) — запоминается для каждого провайдера отдельно
- **кастомный base_url** (`base_url`) — только для OpenAI, опционально

Активный провайдер — тот, который сейчас отвечает на запросы чата.
Переключение мгновенное — ключи и модели уже сохранены.

---

### GET /api/v1/settings/llm — текущее состояние

```json
{
  "source": "file",
  "active_provider": "gigachat",
  "active_model": "GigaChat-2",
  "providers": {
    "gigachat": {
      "key_configured": true,
      "api_key_hint": "OTYz...YQ==",
      "selected_model": "GigaChat-2",
      "base_url": null,
      "gigachat_scope": "GIGACHAT_API_PERS"
    },
    "openai": {
      "key_configured": true,
      "api_key_hint": "sk-p...1XoA",
      "selected_model": "gpt-4o-mini",
      "base_url": "https://api.openai.com/v1"
    },
    "openrouter": {
      "key_configured": false,
      "api_key_hint": null,
      "selected_model": null,
      "base_url": "https://openrouter.ai/api/v1"
    }
  }
}
```

`source: "env"` — настройки из конфига сервиса (файл не создан).
`source: "file"` — настройки из файла, переданного через API.
`key_configured: false` — ключ не сохранён, переключиться на этот провайдер нельзя.
`base_url_custom: true` — у OpenAI задан кастомный адрес (не дефолтный).

---

### POST /api/v1/settings/llm/provider/{provider} — сохранить ключ

Провайдер в URL: `gigachat`, `openai`, `openrouter`.
Не меняет активный провайдер и не сбрасывает выбранную модель.

**GigaChat:**
```json
{
  "api_key": "base64_credentials_из_личного_кабинета_сбера",
  "gigachat_scope": "GIGACHAT_API_PERS"
}
```
`gigachat_scope`: `GIGACHAT_API_PERS` — физлица, `GIGACHAT_API_B2B` — бизнес, `GIGACHAT_API_CORP` — корпоративный.

**OpenAI (стандартный):**
```json
{"api_key": "sk-..."}
```

**OpenAI с кастомным base_url** (LM Studio, прокси, любой OpenAI-совместимый сервер):
```json
{
  "api_key": "lm-studio",
  "base_url": "http://192.168.1.100:1234/v1"
}
```

**OpenRouter:**
```json
{"api_key": "sk-or-..."}
```

```json
{"saved": true, "provider": "openai", "base_url": "https://api.openai.com/v1"}
```

---

### POST /api/v1/settings/llm/provider/{provider}/model — выбрать модель

Сохраняет выбранную модель для провайдера. Не меняет активный провайдер.

```json
{"model": "gpt-4o-mini"}
```

```json
{"saved": true, "provider": "openai", "model": "gpt-4o-mini"}
```

---

### POST /api/v1/settings/llm/active — переключить активный провайдер

Провайдер должен иметь сохранённый ключ. Также обновляет `selected_model` провайдера.

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini"
}
```

```json
{
  "active_provider": "openai",
  "active_model": "gpt-4o-mini",
  "base_url": "https://api.openai.com/v1"
}
```

---

### GET /api/v1/settings/llm/models/{provider} — список моделей

Провайдер в URL: `gigachat`, `openai`, `openrouter`.

Если ключ настроен — запрашивает реальный список у провайдера (`source: "api"`).
Если ключ не настроен или запрос не удался — возвращает встроенный список (`source: "default"`).
GigaChat всегда возвращает дефолт.

В ответе также `selected_model` — текущий выбор для этого провайдера.

```
GET /api/v1/settings/llm/models/openai
```
```json
{
  "provider": "openai",
  "source": "api",
  "selected_model": "gpt-4o-mini",
  "models": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "..."]
}
```

---

### POST /api/v1/settings/llm/validate — проверить ключ

Делает минимальный тестовый запрос к провайдеру. Всегда возвращает HTTP 200.

```json
{"provider": "openai"}
```

```json
{"valid": true, "provider": "openai", "error": null}
```

```json
{"valid": false, "provider": "openai", "error": "Connection refused"}
```

---

### DELETE /api/v1/settings/llm — сбросить все настройки

Удаляет файл настроек. Сервис возвращается к значениям из своего конфига.

```json
{"reset": true, "message": "Настройки сброшены, используются значения из .env"}
```

---

## Типичные сценарии

### Первоначальная настройка провайдера

1. `GET /api/v1/settings/llm` — загрузить текущее состояние
2. `GET /api/v1/settings/llm/models/{provider}` — получить список моделей
3. `POST /api/v1/settings/llm/provider/{provider}` — сохранить ключ
4. `POST /api/v1/settings/llm/validate` — проверить ключ (кнопка «Проверить»)
5. `POST /api/v1/settings/llm/provider/{provider}/model` — выбрать модель
6. `POST /api/v1/settings/llm/active` — сделать провайдер активным

### Переключение между настроенными провайдерами

1. `GET /api/v1/settings/llm` — видим все провайдеры, у каждого `selected_model` уже сохранена
2. `POST /api/v1/settings/llm/active` — переключить одним запросом

### Подключить LM Studio вместо OpenAI

```json
POST /api/v1/settings/llm/provider/openai
{
  "api_key": "lm-studio",
  "base_url": "http://localhost:1234/v1"
}
```
Затем переключить активный провайдер на `openai` с нужной моделью.

---

## Порядок первого запуска

1. `GET /health` — убедиться что сервис доступен
2. `POST /api/v1/settings/llm/provider/gigachat` — сохранить ключ
3. `POST /api/v1/settings/llm/active` — установить активный провайдер
4. `POST /api/v1/products` — загрузить товары
5. `POST /api/v1/faq` — загрузить FAQ
6. `POST /api/v1/navigation` — загрузить навигацию
7. `POST /api/v1/chat` — проверить чат
