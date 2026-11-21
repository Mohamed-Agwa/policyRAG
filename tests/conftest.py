import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import HashEmbeddings, make_fake_llm  # noqa: E402
from rag.cache import SQLiteCache  # noqa: E402
from rag.config import get_settings  # noqa: E402
from rag.ingest import build_index  # noqa: E402
from rag.pipeline import PolicyRAG, open_vectorstore  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rag")
    return get_settings().with_overrides(
        chroma_dir=tmp / "chroma",
        cache_path=tmp / "cache.sqlite",
        use_reranker=False,
        groq_api_key="",
        min_relevance=0.30,
    )


@pytest.fixture(scope="session")
def embeddings():
    return HashEmbeddings()


@pytest.fixture(scope="session")
def built_index(settings, embeddings):
    return build_index(settings, embeddings=embeddings, quiet=True)


@pytest.fixture()
def make_pipeline(settings, embeddings, built_index):
    def _make(fixed_answer=None, **overrides):
        s = settings.with_overrides(**overrides) if overrides else settings
        cache = SQLiteCache(s.cache_path) if s.use_cache else None
        return PolicyRAG(s, open_vectorstore(s, embeddings), make_fake_llm(fixed_answer), cache=cache)
    return _make


@pytest.fixture()
def client(make_pipeline):
    from app import create_app

    app = create_app(pipeline=make_pipeline(use_cache=False))
    app.testing = True
    return app.test_client()
