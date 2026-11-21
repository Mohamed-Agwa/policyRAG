"""Cross-encoder re-ranking.

Vector similarity is only a proxy for relevance: the bi-encoder embeds the
question and each chunk separately. A cross-encoder reads the question and the
chunk together and scores how well the chunk answers the question. We retrieve
a wider candidate set (FETCH_K) and keep the best TOP_K after re-ranking.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Sequence


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path | str | None = None):
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._model = None
        self._lock = Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._model = TextCrossEncoder(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, list(texts))]


def get_reranker(settings) -> CrossEncoderReranker | None:
    if not settings.use_reranker:
        return None
    return CrossEncoderReranker(settings.rerank_model, cache_dir=settings.model_cache_dir)
