import json
from pathlib import Path

from eval.metrics import citation_correct, key_fact_match, percentile_ms, token_f1

ROOT = Path(__file__).resolve().parents[1]


def test_token_f1_ignores_citations_and_commas():
    assert token_f1("EGP 15,000 per year [POL-HR-008]", "EGP 15000 per year") == 1.0
    assert token_f1("nothing related", "EGP 600 per month") == 0.0


def test_key_fact_match():
    exact, partial, frac = key_fact_match("Up to 7 days, used by 31 March [POL-HR-003].", ["7", "31 March"])
    assert exact and partial and frac == 1.0
    assert key_fact_match("Up to 7 days.", ["7", "31 March"])[:2] == (False, True)


def test_citation_correct_rules():
    assert citation_correct(True, False, ["POL-HR-003"], ["POL-HR-003"], True)
    assert not citation_correct(True, False, ["POL-HR-004"], ["POL-HR-003"], True)   # wrong doc
    assert not citation_correct(True, False, [], ["POL-HR-003"], None)               # no citation
    assert not citation_correct(True, False, ["POL-HR-003"], ["POL-HR-003"], False)  # judge rejects
    assert citation_correct(False, True, [], [], None)                               # correct refusal
    assert not citation_correct(False, False, ["POL-HR-003"], [], True)              # should refuse


def test_percentiles():
    assert percentile_ms([100, 200, 300, 400], 50) == 250.0
    assert percentile_ms([], 95) is None


def test_question_set_is_valid():
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if line]
    assert 15 <= len(qs) <= 30
    assert len({q["id"] for q in qs}) == len(qs)
    corpus_ids = {p.name.split("-", 3)[0] + "-" + p.name.split("-")[1] + "-" + p.name.split("-")[2]
                  for p in (ROOT / "data" / "policies").iterdir()}
    for q in qs:
        assert q["answerable"] == bool(q["expected_doc_ids"])
        assert set(q["expected_doc_ids"]) <= corpus_ids
