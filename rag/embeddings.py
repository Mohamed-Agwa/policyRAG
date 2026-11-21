"""Embedding model wrapper.

Uses FastEmbed (ONNX Runtime) to run sentence-transformers/all-MiniLM-L6-v2,
the same embedding model as the course's Praxa app, without installing
PyTorch. This keeps memory low enough for Render's free 512 MB instance.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by fastembed.TextEmbedding."""

    def __init__(self, model_name: str, cache_dir: Path | str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self._model = None
        self._lock = Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding  # imported lazily: heavy

                    self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([text]))).tolist()


def get_embeddings(settings) -> Embeddings:
    return FastEmbedEmbeddings(settings.embed_model, cache_dir=settings.model_cache_dir)
