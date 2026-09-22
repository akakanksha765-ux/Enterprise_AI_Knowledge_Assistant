"""Ollama client."""

from ollama import Client

from src.config import settings
from src.metrics import (
    LLM_REQUESTS,
    LLM_ERRORS,
)

ollama_client = Client(
    host=settings.ollama_host,
)


def check_ollama():
    """Check whether Ollama is running."""

    try:
        ollama_client.list()
        return True
    except Exception:
        return False



def generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
) -> str:
    """Generate response from Ollama."""

    LLM_REQUESTS.inc()

    try:
        response = ollama_client.chat(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": (
                    settings.temperature
                    if temperature is None
                    else temperature
                ),
                "top_k": settings.top_k,
                "top_p": settings.top_p,
                "num_predict": settings.max_tokens,
            },
        )

        return response["message"]["content"]

    except Exception:
        LLM_ERRORS.inc()
        raise