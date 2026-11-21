import pytest

from rag.guardrails import (InvalidQuestion, clean_citations, is_refusal, postprocess,
                            truncate_words, validate_question)
from rag.prompts import REFUSAL


def test_invented_citations_are_removed():
    text, ids = clean_citations("Carry over 7 days [POL-HR-003]. Ask HR [POL-XX-999].", {"POL-HR-003"})
    assert ids == ["POL-HR-003"]
    assert "POL-XX-999" not in text and "[POL-HR-003]" in text


def test_multi_id_brackets():
    text, ids = clean_citations("See [POL-HR-003, POL-HR-004 > 2. Holidays].", {"POL-HR-003", "POL-HR-004"})
    assert ids == ["POL-HR-003", "POL-HR-004"]
    assert "[POL-HR-003, POL-HR-004]" in text


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal("I can only answer about our policies.")
    assert not is_refusal("You get 21 days [POL-HR-003].")


def test_postprocess_refusal_has_no_citations():
    g = postprocess("I can only answer about our policies.", ["POL-HR-003"], 180)
    assert g.refused and g.cited_ids == [] and g.answer == REFUSAL


def test_uncited_answer_gets_inferred_citation():
    g = postprocess("You get 21 days.", ["POL-HR-003", "POL-HR-005"], 180)
    assert g.cited_ids == ["POL-HR-003"] and g.citation_inferred


def test_length_cap():
    text, cut = truncate_words("word " * 400, 50)
    assert cut and len(text.split()) <= 51


@pytest.mark.parametrize("bad", ["", "   ", None, 42, "x" * 1001])
def test_invalid_questions(bad):
    with pytest.raises(InvalidQuestion):
        validate_question(bad)
