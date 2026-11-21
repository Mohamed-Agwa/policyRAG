"""The RAG pipeline: retrieve -> (re-rank) -> guard -> prompt | LLM -> guard.

The generation step is a LangChain Expression Language (LCEL) chain,
`prompt | llm | StrOutputParser()`, as taught in the course. Retrieval,
re-ranking and the guardrails run around it in plain Python so that each step
is testable and can be switched off for ablations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .cache import SQLiteCache, make_key
from .config import Settings, get_settings
from .guardrails import postprocess, validate_question
from .prompts import REFUSAL, build_prompt, format_context

SNIPPET_CHARS = 280
log = logging.getLogger("policy_rag")


class IndexNotReady(RuntimeError):
    pass


class LLMNotConfigured(RuntimeError):
    pass


@dataclass
class RetrievedChunk:
    doc: Document
    vector_score: float
    rerank_score: float | None = None

    @property
    def doc_id(self) -> str:
        return self.doc.metadata["doc_id"]


def open_vectorstore(settings: Settings, embeddings=None):
    from langchain_chroma import Chroma

    from .embeddings import get_embeddings

    if not Path(settings.chroma_dir).exists():
        raise IndexNotReady(f"No index at {settings.chroma_dir}. Run: python -m rag.ingest")
    return Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings or get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
        create_collection_if_not_exists=False,
    )


_INDEX_SIZE: dict[str, int] = {}


def index_size(settings: Settings) -> int | None:
    """Number of indexed chunks, without loading any model (used by /health).

    Cached per index directory: Render calls /health every few seconds, and the
    index only changes on redeploy.
    """
    key = str(settings.chroma_dir)
    if key in _INDEX_SIZE:
        return _INDEX_SIZE[key]
    if not Path(settings.chroma_dir).exists():
        return None
    try:
        import chromadb

        client = chromadb.PersistentClient(path=key)
        _INDEX_SIZE[key] = client.get_collection(settings.collection).count()
        return _INDEX_SIZE[key]
    except Exception:
        return None


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= SNIPPET_CHARS:
        return text
    cut = text[:SNIPPET_CHARS].rsplit(" ", 1)[0]
    return cut + " …"


class PolicyRAG:
    def __init__(self, settings: Settings, vectorstore, llm, reranker=None, cache: SQLiteCache | None = None):
        self.settings = settings
        self.vectorstore = vectorstore
        self.llm = llm
        self.reranker = reranker
        self.cache = cache
        self.prompt = build_prompt(settings.prompt_variant, settings.max_answer_words)
        self.chain = self.prompt | llm | StrOutputParser()  # LCEL

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PolicyRAG":
        from .llm import make_chat_llm
        from .rerank import get_reranker

        settings = settings or get_settings()
        t0 = time.perf_counter()
        cache = SQLiteCache(settings.cache_path) if settings.use_cache else None
        store = open_vectorstore(settings)
        log.info("Index opened in %.1fs (%s)", time.perf_counter() - t0, settings.chroma_dir)
        t1 = time.perf_counter()
        store.embeddings.embed_query("load model")  # load the ONNX model now, not mid-request
        log.info("Embedding model %s loaded in %.1fs", settings.embed_model, time.perf_counter() - t1)
        reranker = get_reranker(settings)
        if reranker:
            t2 = time.perf_counter()
            reranker.score("load model", ["load model"])
            log.info("Re-ranker %s loaded in %.1fs", settings.rerank_model, time.perf_counter() - t2)
        log.info("Pipeline ready: llm=%s reranker=%s", settings.llm_model, bool(reranker))
        return cls(settings, store, make_chat_llm(settings), reranker=reranker, cache=cache)

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, question: str) -> tuple[list[RetrievedChunk], float]:
        s = self.settings
        k = s.fetch_k if self.reranker else s.top_k
        hits = self.vectorstore.similarity_search_with_relevance_scores(question, k=k)
        chunks = [RetrievedChunk(doc, float(score)) for doc, score in hits]
        best = max((c.vector_score for c in chunks), default=0.0)
        if self.reranker and chunks:
            scores = self.reranker.score(question, [c.doc.page_content for c in chunks])
            for c, sc in zip(chunks, scores):
                c.rerank_score = sc
            chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks[: s.top_k], best

    # ------------------------------------------------------------------ generation
    def _generate(self, question: str, context: str, use_cache: bool) -> tuple[str, bool]:
        key = make_key(self.settings.llm_model, self.settings.prompt_variant,
                       self.settings.max_answer_words, question, context)
        if use_cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit, True
        if not self.settings.llm_configured and not getattr(self.llm, "is_fake", False):
            raise LLMNotConfigured("GROQ_API_KEY is not set. Add it to .env or the host's environment.")
        text = self.chain.invoke({"question": question, "context": context})
        if self.cache is not None:
            self.cache.set(key, text)
        return text, False

    def answer(self, question: str, use_cache: bool = True) -> dict:
        t0 = time.perf_counter()
        question = validate_question(question)
        chunks, best = self.retrieve(question)
        t_retrieval = (time.perf_counter() - t0) * 1000
        retrieved = [{
            "doc_id": c.doc_id,
            "section": c.doc.metadata["section"],
            "vector_score": round(c.vector_score, 4),
            "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 4),
        } for c in chunks]

        base = {"model": self.settings.llm_model, "retrieved": retrieved, "best_score": round(best, 4)}

        # Guardrail 1: out-of-corpus questions are refused before any LLM call.
        if not chunks or best < self.settings.min_relevance:
            log.info("Refused (best score %.3f < %.2f) after %.0f ms retrieval",
                     best, self.settings.min_relevance, t_retrieval)
            return {**base, "answer": REFUSAL, "refused": True, "refusal_reason": "low_relevance",
                    "timings_ms": {"retrieval": round(t_retrieval, 1), "llm": 0.0},
                    "citations": [], "citation_inferred": False, "cached": False, "llm_called": False,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

        context = format_context(chunks)
        t1 = time.perf_counter()
        raw, cached = self._generate(question, context, use_cache)
        t_llm = (time.perf_counter() - t1) * 1000
        log.info("Answered: retrieval %.0f ms, llm %.0f ms%s", t_retrieval, t_llm, " (cached)" if cached else "")

        # Guardrails 2-4: model refusal, citation validation, length cap.
        guarded = postprocess(raw, [c.doc_id for c in chunks], self.settings.max_answer_words)
        citations = []
        for doc_id in guarded.cited_ids:
            best_chunk = next(c for c in chunks if c.doc_id == doc_id)
            m = best_chunk.doc.metadata
            citations.append({
                "doc_id": doc_id,
                "title": m["title"],
                "section": m["section"],
                "snippet": _snippet(m["text"]),
                "url": f"/docs/{doc_id}#{m['anchor']}",
            })
        return {**base, "answer": guarded.answer, "refused": guarded.refused,
                "refusal_reason": "model" if guarded.refused else None,
                "citations": citations, "citation_inferred": guarded.citation_inferred,
                "truncated": guarded.truncated, "cached": cached, "llm_called": not cached,
                "context": context,
                "timings_ms": {"retrieval": round(t_retrieval, 1), "llm": round(t_llm, 1)},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
