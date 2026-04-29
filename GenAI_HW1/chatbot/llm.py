import os

from openai import OpenAI


def get_openai_client() -> OpenAI:
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_BASE_URL")

    if not api_key or not base_url:
        raise RuntimeError("請先在 .env 中設定 CUSTOM_API_KEY 與 CUSTOM_BASE_URL。")

    return OpenAI(api_key=api_key, base_url=base_url)


def stream_chat_completion(client: OpenAI, model: str, messages: list[dict], temperature: float, max_tokens: int):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        yield chunk.choices[0].delta.content or ""
