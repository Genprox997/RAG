"""
Project configuration. All secrets come from environment variables (.env).
Everything is OpenAI-compatible so you can plug in DeepSeek / Qwen / OpenAI / etc.
"""
import os
from dataclasses import dataclass, field

# Ensure .env is always loaded, regardless of which entry point runs first
# (e.g. `python -m src.ingestion` would otherwise miss it and fall back to cloud).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # ---- LLM (generation) ----
    llm_api_key: str = _get("LLM_API_KEY")
    llm_base_url: str = _get("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = _get("LLM_MODEL", "gpt-4o-mini")

    # ---- Embedding ----
    # cloud -> OpenAI-compatible embeddings API ; local -> offline model via fastembed
    embed_provider: str = _get("EMBED_PROVIDER", "cloud")
    embed_api_key: str = field(default_factory=lambda: _get("EMBED_API_KEY") or _get("LLM_API_KEY"))
    embed_base_url: str = _get("EMBED_BASE_URL", "https://api.openai.com/v1")
    embed_model: str = _get("EMBED_MODEL", "text-embedding-3-small")
    embed_dim: int = int(_get("EMBED_DIM", "1536"))

    # ---- Reranker (optional, cloud cross-encoder or local offline) ----
    # none | jina | cohere | local
    rerank_provider: str = _get("RERANK_PROVIDER", "none")
    rerank_api_key: str = _get("RERANK_API_KEY", "")
    rerank_model: str = _get("RERANK_MODEL", "jina-reranker-v2-base-multilingual")
    # local backend: embedding (reuse cached encoder) | cross-encoder (bge-reranker-v2-m3)
    local_reranker: str = _get("LOCAL_RERANKER", "embedding")

    # ---- Retrieval ----
    top_k_vector: int = int(_get("TOP_K_VECTOR", "20"))
    top_k_bm25: int = int(_get("TOP_K_BM25", "20"))
    top_k_final: int = int(_get("TOP_K_FINAL", "6"))
    rrf_k: int = int(_get("RRF_K", "60"))

    # ---- Agentic loop ----
    max_iterations: int = int(_get("MAX_ITERATIONS", "3"))
    # continuous-confidence threshold to stop self-reflection early
    reflect_confidence_threshold: float = float(_get("REFLECT_CONFIDENCE_THRESHOLD", "0.8"))
    # LLM may emit any number of sub-queries; cap for safety (was hard-coded <=2)
    max_sub_queries: int = int(_get("MAX_SUB_QUERIES", "4"))

    # ---- Paths ----
    index_dir: str = _get("INDEX_DIR", "data/index")
    docs_dir: str = _get("DOCS_DIR", "data/docs")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def has_llm_credentials() -> bool:
    s = get_settings()
    return bool(s.llm_api_key and s.llm_base_url)
