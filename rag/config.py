"""Central configuration, read from environment variables (and an optional .env file).

Every tunable used in the design document (k, chunk size, models, thresholds)
lives here so that evaluation ablations can override them without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    # LLM (Groq exposes an OpenAI-compatible API)
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024")))
    max_answer_words: int = field(default_factory=lambda: int(os.getenv("MAX_ANSWER_WORDS", "180")))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "6")))

    # Embeddings + re-ranking (FastEmbed / ONNX)
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    use_reranker: bool = field(default_factory=lambda: _bool("USE_RERANKER", True))
    model_cache_dir: Path = field(default_factory=lambda: _path("MODEL_CACHE_DIR", ".models"))

    # Chunking + retrieval
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    fetch_k: int = field(default_factory=lambda: int(os.getenv("FETCH_K", "10")))
    # Out-of-corpus guardrail: refuse without calling the LLM if the best
    # cosine similarity between the question and any chunk is below this.
    min_relevance: float = field(default_factory=lambda: float(os.getenv("MIN_RELEVANCE", "0.30")))

    # Prompt variant (used by the prompt ablation): "cited" or "plain"
    prompt_variant: str = field(default_factory=lambda: os.getenv("PROMPT_VARIANT", "cited"))

    # Paths
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", "data/policies"))
    chroma_dir: Path = field(default_factory=lambda: _path("CHROMA_DIR", "chroma_db"))
    collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "nilebyte_policies"))
    cache_path: Path = field(default_factory=lambda: _path("LLM_CACHE_PATH", "cache/llm_cache.sqlite"))
    use_cache: bool = field(default_factory=lambda: _bool("USE_LLM_CACHE", True))

    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))

    @property
    def llm_configured(self) -> bool:
        return bool(self.groq_api_key)

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)


def get_settings() -> Settings:
    return Settings()
