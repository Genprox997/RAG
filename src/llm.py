"""
LLM & Embedding clients (OpenAI-compatible).
- chat(): non-streaming
- stream_chat(): generator yielding text deltas
- embed(): batch embedding
"""
import time
from typing import Iterator

from openai import OpenAI

import config
from src.cache import default_cache


def _client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def _chat_impl(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    s = config.get_settings()
    client = _client(s.llm_base_url, s.llm_api_key)
    kwargs = {
        "model": s.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    json_mode: bool = False,
    _cache: bool = True,
) -> str:
    """Non-streaming chat completion. Returns the assistant message content.

    Results are cached by (messages, model, temperature, max_tokens, json_mode)
    when ENABLE_CACHE is on, so repeated identical prompts skip the API call.
    """
    s = config.get_settings()
    if _cache:
        return default_cache().cached(
            "chat",
            (messages, s.llm_model, temperature, max_tokens, json_mode),
            lambda: _chat_impl(
                messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
            ),
        )
    return _chat_impl(
        messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
    )


def stream_chat(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Streaming chat completion. Yields text deltas."""
    s = config.get_settings()
    client = _client(s.llm_base_url, s.llm_api_key)
    kwargs = {
        "model": s.llm_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    for chunk in client.chat.completions.create(**kwargs):
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


_LOCAL_MODEL = None


def _local_embedder():
    """Lazily load the offline embedding model (fastembed)."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from fastembed import TextEmbedding

        s = config.get_settings()
        _LOCAL_MODEL = TextEmbedding(model_name=s.embed_model)
    return _LOCAL_MODEL


def _embed_impl(texts: list[str], task: str | None = None) -> list[list[float]]:
    """Actual embedding call (no caching)."""
    s = config.get_settings()
    # API / model may reject empty strings
    texts = [t if t.strip() else " " for t in texts]

    if s.embed_provider == "local":
        model = _local_embedder()
        kwargs = {"task": task} if task else {}
        vecs = model.embed(texts, **kwargs)
        return [list(map(float, v)) for v in vecs]

    # ---- cloud (OpenAI-compatible) ----
    client = _client(s.embed_base_url, s.embed_api_key)
    out: list[list[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(
            model=s.embed_model, input=texts[i : i + batch]
        )
        out.extend([d.embedding for d in resp.data])
        time.sleep(0.05)
    return out


def embed(texts: list[str], task: str | None = None, _cache: bool = True) -> list[list[float]]:
    """Batch embedding.

    - provider == 'local'  -> offline fastembed model (no API, no cost)
    - provider == 'cloud'  -> OpenAI-compatible embeddings endpoint

    `task` is only used by the local backend (e.g. bge needs
    'retrieval.query' vs 'retrieval.passage' to add the right prefix).

    Results are cached by (texts, provider, model, task) when ENABLE_CACHE is on,
    so re-ingestion / repeated queries skip the (potentially costly) embedding call.
    """
    s = config.get_settings()
    if _cache:
        return default_cache().cached(
            "embed",
            (tuple(texts), s.embed_provider, s.embed_model, task),
            lambda: _embed_impl(texts, task),
        )
    return _embed_impl(texts, task)
