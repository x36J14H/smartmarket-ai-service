from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embedding
    embed_model: str = "intfloat/multilingual-e5-large"
    embed_dim: int = 1024

    # LLM — дефолтный провайдер и ключ (используются если нет data/llm_settings.json)
    llm_provider: str = "openai"
    llm_api_key: str = "sk-..."
    llm_model: str = "gpt-4o-mini"

    # base_url для провайдеров
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # GigaChat не нужен — SDK сам знает адрес

    # 1С backend URL (для проверки остатков)
    onec_base_url: str = ""
    onec_user: str = ""
    onec_password: str = ""

    # RAG
    top_k: int = 6
    score_threshold: float = 0.60  # минимальный score для включения в результаты


settings = Settings()
