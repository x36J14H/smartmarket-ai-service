# smartmarket-ai-service

AI-слой для интернет-магазина. Сервис отвечает за RAG-чат с покупателями, семантический поиск товаров и ИИ-рекомендации — подключается к основному приложению через REST API.

## Стек

- **FastAPI** — REST API
- **Qdrant** — векторное хранилище товаров
- **fastembed** — локальные эмбеддинги (multilingual-e5-large)
- **OpenAI-compatible LLM** — генерация ответов (LM Studio, OpenAI, и др.)

## Запуск

**Установка зависимостей**
```bash
pip install -r requirements.txt
```

**Заполнение env**
```bash
cp .env.example .env
```

**Запуск Qdrant**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Запуск сервера**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API — `http://localhost:8000`, документация — `http://localhost:8000/docs`.

## Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/v1/chat` | Чат с ботом |
| POST | `/api/v1/embed` | Загрузить товары в векторную БД |
| GET/POST | `/api/v1/products` | Управление товарами |
| GET | `/health` | Проверка состояния |
