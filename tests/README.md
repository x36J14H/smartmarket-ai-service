# Тесты SmartMarket AI Service

## Структура

```
tests/
├── conftest.py              — общие фикстуры, переключение LLM на LM Studio
├── test_history.py          — тесты истории диалогов (14 тестов, без внешних зависимостей)
├── test_onec_client.py      — тесты клиента 1С: TTL-кэш, graceful degradation, HTTP (37 тестов)
├── test_bot_unit.py         — юнит-тесты логики бота с моками LLM/1С/Qdrant (34 теста)
├── test_llm.py              — тесты LLM-слоя: analyze_query, ask, stream, rerank (25 тестов)
├── test_1c_connectivity.py  — проверка всех 1С эндпоинтов (15 тестов)
├── test_api.py              — тесты HTTP API через FastAPI TestClient (27 тестов)
└── test_bot_integration.py  — интеграционные тесты бота с реальными LLM+1С (18 тестов)
```

## Требования

```
pip install -r requirements-test.txt
```

LM Studio должен быть запущен с моделью `google/gemma-4-12b` на `http://192.168.56.1:1234`

## Запуск

### Только быстрые юниты (без LLM и 1С, ~2 сек)
```bash
pytest tests/test_history.py tests/test_bot_unit.py tests/test_onec_client.py
```

### Проверка 1С (нужна только 1С)
```bash
pytest tests/test_1c_connectivity.py -v
```

### Тесты API (без внешних сервисов)
```bash
pytest tests/test_api.py -v
```

### Тесты LLM (нужен LM Studio)
```bash
pytest tests/test_llm.py -v -s
```

### Интеграционные тесты бота (нужны LLM + 1С + Qdrant, ~2-5 мин)
```bash
pytest tests/test_bot_integration.py -v -s
```

### Все тесты
```bash
pytest
```

### Только помеченные integration
```bash
pytest -m integration -v -s
```

### Конкретный класс или тест
```bash
pytest tests/test_llm.py::TestAnalyzeQuery -v -s
pytest tests/test_bot_unit.py::TestChatPipeline::test_chat_strips_hallucinated_links -v -s
```

## Маркеры

- `@pytest.mark.integration` — тест требует реальных внешних сервисов
- `@pytest.mark.slow` — тест занимает более 10 секунд

## Что проверяется

### Юнит-тесты (быстрые, без внешних сервисов)
- Форматирование чанков: товары, навигация, FAQ, акции, заказы
- Валидация и очистка галлюцинированных ссылок
- Извлечение номера заказа из текста и истории
- Enrichment с данными availability (фильтрация + обогащение payload)
- Pipeline _build_response для каждого intent (products, info, order_help, promotions, compare, catalog_browse)
- Полный chat() pipeline: сохранение истории, strip ссылок, структура ответа
- Streaming chat_stream(): meta → chunks → done
- TTL-кэш в onec_client (set/get/expire/invalidate)
- Graceful degradation при недоступной 1С
- HTTP-ошибки (Connection refused, Timeout, 404)
- Кэширование categories и promotions
- Парсинг availability (фильтрация нулей, батчинг)
- История: добавление, очистка, TTL, изоляция сессий, ограничение размера

### Интеграционные тесты (нужны внешние сервисы)
- Прямые запросы к LM Studio (модели, chat, streaming)
- analyze_query для всех интентов с реальной LLM
- Извлечение фильтров (price_max, price_min, brand)
- Уточняющие вопросы
- Разрешение местоимений через историю
- ask() с intent-специфичными промптами
- ask_stream() стриминг
- rerank_async() фильтрация нерелевантных
- Все 1С эндпоинты (/categories, /availability, /catalog/search, ...)
- Полный pipeline бота: products, info, promotions, order_help, compare, catalog_browse
- Многоходовой диалог
- Проверка галлюцинированных ссылок
- Стриминг с реальной LLM
