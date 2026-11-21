"""Ingestion: parse and clean the policy files, chunk them, embed them and
store the vectors in a persistent Chroma collection.

Run:  python -m rag.ingest            (build or rebuild the index)
      python -m rag.ingest --stats    (print corpus statistics only)

Supported formats: Markdown (.md), plain text (.txt), HTML (.html/.htm), PDF.
Every document gets a stable doc_id (e.g. POL-HR-003) used for citations.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from markdown.extensions.toc import slugify
from pypdf import PdfReader

from .config import Settings, get_settings

DOC_ID_FROM_NAME = re.compile(r"^(POL-[A-Z]+-\d{3})")
SUPPORTED = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
WORDS_PER_PAGE = 300  # a formatted A4 policy page (our PDF renders ~300 words/page)


@dataclass
class SourceDoc:
    doc_id: str
    title: str
    markdown: str  # cleaned text with Markdown headings, used for chunking and /docs view
    path: Path
    kind: str  # md | txt | html | pdf
    meta: dict

    @property
    def word_count(self) -> int:
        return len(self.markdown.split())


# --------------------------------------------------------------------------- parsing

def _doc_id_from_name(path: Path) -> str:
    m = DOC_ID_FROM_NAME.match(path.name)
    return m.group(1) if m else path.stem.upper()


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_markdown(path: Path) -> SourceDoc:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = raw
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        meta = yaml.safe_load(fm) or {}
    doc_id = str(meta.get("doc_id") or _doc_id_from_name(path))
    title = str(meta.get("title") or _first_heading(body) or path.stem)
    kind = "txt" if path.suffix == ".txt" else "md"
    return SourceDoc(doc_id, title, _clean(body), path, kind, meta)


def _first_heading(md: str) -> str | None:
    m = re.search(r"^#\s+(.+)$", md, re.M)
    return m.group(1).strip() if m else None


def parse_html(path: Path) -> SourceDoc:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    meta = {m.get("name"): m.get("content") for m in soup.find_all("meta") if m.get("name")}
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    # Clean: drop boilerplate that is not policy content.
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    root = soup.find("main") or soup.body or soup
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
        if el.name in {"p", "li"} and el.find_parent(["li", "table"]) and el.name != "li":
            continue
        text = re.sub(r"\s+([.,;:)])", r"\1", " ".join(el.get_text(" ", strip=True).split()))
        if not text:
            continue
        if el.name.startswith("h"):
            lines.append("\n" + "#" * int(el.name[1]) + " " + text + "\n")
        elif el.name == "li":
            lines.append("- " + text)
        elif el.name == "tr":
            cells = [" ".join(c.get_text(" ", strip=True).split()) for c in el.find_all(["th", "td"])]
            is_header = bool(el.find("th"))
            if is_header:
                lines.append("")  # Markdown tables need a blank line before them
            lines.append("| " + " | ".join(cells) + " |")
            if is_header:
                lines.append("|" + " --- |" * len(cells))
        else:
            lines.append(text + "\n")
    doc_id = meta.get("doc_id") or _doc_id_from_name(path)
    return SourceDoc(doc_id, title, _clean("\n".join(lines)), path, "html", meta)


PDF_H2 = re.compile(r"^(\d{1,2})\.\s+([A-Z][^.]{2,80})$")         # "2. Medical Insurance"
PDF_H3 = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+([A-Z][^.]{2,80})$")  # "2.1 Coverage"


def parse_pdf(path: Path) -> SourceDoc:
    """Extract text page by page and restore numbered section headings.

    PDF text extraction loses heading styles, so lines such as "2. Medical
    Insurance" or "2.1 Coverage" are turned back into Markdown headings. This
    lets PDF chunks carry a meaningful section name for citations.
    """
    reader = PdfReader(str(path))
    info = reader.metadata or {}
    title = (info.title or path.stem).strip()
    doc_id = (info.subject or "").strip() or _doc_id_from_name(path)
    out = [f"# {title}"]
    for page in reader.pages:
        for line in _clean(page.extract_text() or "").splitlines():
            s = line.strip()
            if not s or s == title:
                continue
            if PDF_H3.match(s):
                out.append(f"\n### {s}\n")
            elif PDF_H2.match(s):
                out.append(f"\n## {s}\n")
            else:
                out.append(s)
    meta = {"pages": len(reader.pages)}
    return SourceDoc(doc_id, title, _clean("\n".join(out)), path, "pdf", meta)


def parse_file(path: Path) -> SourceDoc:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return parse_markdown(path)
    if suffix in {".html", ".htm"}:
        return parse_html(path)
    if suffix == ".pdf":
        return parse_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_documents(data_dir: Path) -> list[SourceDoc]:
    files = sorted(p for p in Path(data_dir).iterdir() if p.suffix.lower() in SUPPORTED)
    docs = [parse_file(p) for p in files]
    ids = [d.doc_id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate doc_id(s) in corpus: {sorted(dupes)}")
    return docs


@lru_cache(maxsize=4)
def load_documents_cached(data_dir: str) -> dict[str, SourceDoc]:
    return {d.doc_id: d for d in load_documents(Path(data_dir))}


# --------------------------------------------------------------------------- chunking

def chunk_document(doc: SourceDoc, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split by Markdown headings first, then into overlapping windows."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True
    )
    window_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks: list[Document] = []
    for section_doc in header_splitter.split_text(doc.markdown):
        h = section_doc.metadata
        heading = h.get("h3") or h.get("h2") or h.get("h1") or doc.title
        section = " > ".join(x for x in (h.get("h2"), h.get("h3")) if x) or h.get("h1") or doc.title
        for piece in window_splitter.split_text(section_doc.page_content):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            idx = len(chunks)
            chunks.append(Document(
                # The embedded text carries the title and section so that a
                # chunk is understandable (and retrievable) on its own.
                page_content=f"{doc.title} > {section}\n{piece}",
                metadata={
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "section": section,
                    "anchor": slugify(heading, "-"),
                    "source": doc.path.name,
                    "kind": doc.kind,
                    "chunk_id": f"{doc.doc_id}::{idx:03d}",
                    "text": piece,
                },
            ))
    return chunks


def chunk_corpus(docs: list[SourceDoc], settings: Settings) -> list[Document]:
    out: list[Document] = []
    for d in docs:
        out.extend(chunk_document(d, settings.chunk_size, settings.chunk_overlap))
    return out


# --------------------------------------------------------------------------- indexing

def build_index(settings: Settings | None = None, embeddings=None, quiet: bool = False) -> int:
    """(Re)build the Chroma collection. Returns the number of chunks indexed."""
    from langchain_chroma import Chroma

    from .embeddings import get_embeddings

    settings = settings or get_settings()
    random.seed(settings.seed)
    docs = load_documents(settings.data_dir)
    chunks = chunk_corpus(docs, settings)
    embeddings = embeddings or get_embeddings(settings)

    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    store = Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    batch = 128
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        store.add_documents(part, ids=[c.metadata["chunk_id"] for c in part])
    if not quiet:
        print(f"Indexed {len(chunks)} chunks from {len(docs)} documents into {settings.chroma_dir}")
    return len(chunks)


def corpus_stats(settings: Settings) -> dict:
    docs = load_documents(settings.data_dir)
    chunks = chunk_corpus(docs, settings)
    words = sum(d.word_count for d in docs)
    return {
        "documents": len(docs),
        "formats": sorted({d.kind for d in docs}),
        "words": words,
        "estimated_pages": round(words / WORDS_PER_PAGE),
        "chunks": len(chunks),
        "per_document": [(d.doc_id, d.kind, d.word_count, d.title) for d in docs],
    }


def warm_models(settings: Settings) -> None:
    """Download the embedding (and re-ranker) models into MODEL_CACHE_DIR."""
    from .embeddings import get_embeddings
    from .rerank import get_reranker

    get_embeddings(settings).embed_query("warm up")
    reranker = get_reranker(settings)
    if reranker:
        reranker.score("warm up", ["warm up"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the policy vector index.")
    parser.add_argument("--stats", action="store_true", help="print corpus statistics and exit")
    args = parser.parse_args(argv)
    settings = get_settings()

    stats = corpus_stats(settings)
    print(f"Corpus: {stats['documents']} documents ({', '.join(stats['formats'])}), "
          f"{stats['words']:,} words (~{stats['estimated_pages']} pages), {stats['chunks']} chunks "
          f"at chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    for doc_id, kind, words, title in stats["per_document"]:
        print(f"  {doc_id:<12} {kind:<4} {words:>6} words  {title}")
    if args.stats:
        return 0
    build_index(settings)
    warm_models(settings)
    print("Models cached in", settings.model_cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
