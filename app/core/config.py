from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embedding
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    # LLM (OpenAI-compatible)
    llm_api_key: str = "sk-..."
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # RAG
    top_k: int = 4


settings = Settings()
