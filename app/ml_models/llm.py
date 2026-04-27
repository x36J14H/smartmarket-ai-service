import re
from openai import OpenAI
from functools import lru_cache
from app.core.config import settings
from app.ml_models.prompts import SYSTEM_PROMPT


@lru_cache(maxsize=1)
def get_llm() -> OpenAI:
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


def ask(question: str, context_chunks: list[str], history: list[dict] = None) -> str:
    context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"})
    response = get_llm().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content
    # Убираем <think>...</think> блок (Qwen3 и другие reasoning модели)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer


HYDE_PROMPT = """Ты — эксперт интернет-магазина электроники.
Напиши короткое описание товара или информации, которая идеально отвечала бы на вопрос пользователя.
Пиши как будто это текст из каталога или FAQ магазина — конкретно, без лишних слов.
Не отвечай на вопрос напрямую, просто опиши идеальный документ."""


def hypothetical_answer(question: str) -> str:
    """HyDE: генерирует гипотетический документ для улучшения векторного поиска."""
    messages = [
        {"role": "system", "content": HYDE_PROMPT},
        {"role": "user", "content": question},
    ]
    response = get_llm().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.5,
        max_tokens=256,
    )
    result = response.choices[0].message.content
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result
