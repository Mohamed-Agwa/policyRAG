from rag.ingest import WORDS_PER_PAGE, chunk_corpus, load_documents


def test_loads_all_formats_with_ids(settings):
    docs = load_documents(settings.data_dir)
    assert 5 <= len(docs) <= 20
    assert {d.kind for d in docs} >= {"md", "html", "pdf"}
    for d in docs:
        assert d.doc_id.startswith("POL-")
        assert d.path.name.startswith(d.doc_id)  # filename and metadata agree
        assert d.title and d.word_count > 300


def test_corpus_size_within_rubric_range(settings):
    words = sum(d.word_count for d in load_documents(settings.data_dir))
    assert 30 <= words / WORDS_PER_PAGE <= 120


def test_html_is_cleaned(settings):
    doc = {d.doc_id: d for d in load_documents(settings.data_dir)}["POL-HR-004"]
    assert "<" not in doc.markdown and "Internal use only" not in doc.markdown  # tags + footer removed
    assert "## 6. Working on a Public Holiday" in doc.markdown


def test_chunks_have_metadata_and_size_limit(settings):
    chunks = chunk_corpus(load_documents(settings.data_dir), settings)
    assert chunks
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        m = c.metadata
        assert {"doc_id", "title", "section", "anchor", "source", "text"} <= set(m)
        assert len(m["text"]) <= settings.chunk_size
        assert c.page_content.startswith(m["title"])


def test_pdf_sections_recovered(settings):
    chunks = [c for c in chunk_corpus(load_documents(settings.data_dir), settings)
              if c.metadata["doc_id"] == "POL-HR-008"]
    sections = {c.metadata["section"] for c in chunks}
    assert any(s.startswith("7. Learning and Development") for s in sections)


def test_index_built(built_index):
    assert built_index > 100
