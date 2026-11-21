"""Flask web app for the Nile Byte policy assistant.

Endpoints
---------
GET  /                 chat interface
POST /chat             {"question": "..."} -> answer + citations + snippets (JSON)
GET  /health           service status (JSON)
GET  /docs/<doc_id>    rendered source policy (citation links point here)
GET  /docs/<doc_id>/raw  original file (md / html / pdf)

Importing this module is cheap: models and the index load lazily on the first
/chat request, so `python -c "import app"` works in CI without an API key.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from threading import Lock

import markdown as md
from flask import Flask, abort, jsonify, render_template, request, send_file

from rag.config import get_settings
from rag.guardrails import InvalidQuestion
from rag.ingest import load_documents_cached
from rag.pipeline import IndexNotReady, LLMNotConfigured, PolicyRAG, index_size

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("policy_rag")
APP_VERSION = "1.0.0"


def llm_error(exc: Exception, settings) -> tuple[int, str]:
    """Map LLM/API exceptions to an HTTP status and a message the user can act on."""
    name = type(exc).__name__
    code = getattr(exc, "status_code", None)
    if code == 429 or "RateLimit" in name:
        return 429, "The AI service's free rate limit was reached. Please wait about a minute and try again."
    if code == 404 or "NotFound" in name:
        return 502, (f"The language model '{settings.llm_model}' is not available on the provider. "
                     "Set LLM_MODEL to a current model.")
    if code in (401, 403) or "Authentication" in name or "PermissionDenied" in name:
        return 502, "The AI service rejected the API key. Check GROQ_API_KEY."
    if "Timeout" in name:
        return 504, "The AI service took too long to respond. Please try again."
    return 502, "The language model request failed. Please try again shortly."


def create_app(pipeline: PolicyRAG | None = None, settings=None) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    settings = settings or (pipeline.settings if pipeline else get_settings())
    state = {"pipeline": pipeline, "warmup": "not started" if pipeline is None else "n/a"}
    lock = Lock()

    def get_pipeline() -> PolicyRAG:
        if state["pipeline"] is None:
            with lock:
                if state["pipeline"] is None:
                    state["pipeline"] = PolicyRAG.from_settings(settings)
        return state["pipeline"]

    def warm_up() -> None:
        """Load the index and models in the background so the first user
        request is fast (useful on Render after a cold start)."""
        state["warmup"] = "loading"
        log.info("Warm-up: loading index and models")
        try:
            p = get_pipeline()
            p.retrieve("annual leave carryover")
            state["warmup"] = "done"
            log.info("Warm-up complete")
        except Exception as exc:  # pragma: no cover
            state["warmup"] = f"failed: {exc}"
            log.exception("Warm-up failed")

    if pipeline is None and os.getenv("WARMUP_ON_START", "false").lower() == "true":
        threading.Thread(target=warm_up, daemon=True).start()

    @app.get("/")
    def index():
        return render_template("index.html", company="Nile Byte Technologies")

    @app.post("/chat")
    def chat():
        t0 = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        question = payload.get("question")
        use_cache = not bool(payload.get("no_cache", False))
        log.info("/chat received (%d chars)", len(question) if isinstance(question, str) else 0)
        try:
            result = get_pipeline().answer(question, use_cache=use_cache)
        except InvalidQuestion as exc:
            return jsonify(error=str(exc)), 400
        except (IndexNotReady, LLMNotConfigured) as exc:
            log.error("chat unavailable: %s", exc)
            return jsonify(error=str(exc)), 503
        except Exception as exc:  # LLM/API problems, surfaced with a useful message
            status, message = llm_error(exc, settings)
            log.exception("chat failed (%s)", status)
            return jsonify(error=message), status
        result.pop("context", None)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify(result)

    @app.get("/health")
    def health():
        p = state["pipeline"]
        chunks = p.vectorstore._collection.count() if p is not None else index_size(settings)
        ok = bool(chunks) and settings.llm_configured
        return jsonify(
            status="ok" if ok else "degraded",
            version=APP_VERSION,
            pipeline_loaded=p is not None,
            warmup=state["warmup"],
            index_chunks=chunks,
            llm_configured=settings.llm_configured,
            model=settings.llm_model,
            embedding_model=settings.embed_model,
            reranker=settings.rerank_model if settings.use_reranker else None,
            top_k=settings.top_k,
        ), 200

    @app.get("/docs/<doc_id>")
    def show_doc(doc_id: str):
        doc = load_documents_cached(str(settings.data_dir)).get(doc_id)
        if doc is None:
            abort(404)
        html = md.markdown(doc.markdown, extensions=["tables", "toc"])
        return render_template("doc.html", doc=doc, body=html)

    @app.get("/docs/<doc_id>/raw")
    def raw_doc(doc_id: str):
        doc = load_documents_cached(str(settings.data_dir)).get(doc_id)
        if doc is None:
            abort(404)
        return send_file(doc.path)

    @app.get("/api/docs")
    def list_docs():
        docs = load_documents_cached(str(settings.data_dir)).values()
        return jsonify([{"doc_id": d.doc_id, "title": d.title, "format": d.kind,
                         "url": f"/docs/{d.doc_id}"} for d in docs])

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
