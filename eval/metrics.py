"""Pure metric helpers (no network), unit-tested in tests/test_eval_metrics.py."""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

_CITATION = re.compile(r"\[[^\]]*POL-[A-Z]+-\d{3}[^\]]*\]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    text = _CITATION.sub(" ", text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation) - {","})
    text = text.replace(",", "")  # "15,000" == "15000"
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalize(prediction).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def key_fact_match(prediction: str, must_include: list[str]) -> tuple[bool, bool, float]:
    """(exact, partial, fraction) of required key facts present in the answer."""
    if not must_include:
        return False, False, 0.0
    pred = normalize(prediction)
    hits = [normalize(f) in pred for f in must_include]
    frac = sum(hits) / len(hits)
    return all(hits), any(hits), frac


def percentile_ms(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def citation_correct(answerable: bool, refused: bool, cited: list[str], expected: list[str],
                     judge_supported: bool | None) -> bool:
    """Citation accuracy for one answer.

    Answerable: not refused, at least one citation, every cited doc is an
    expected source, and (when judged) the cited passages support the answer.
    Unanswerable: correct only if the assistant refused and cited nothing.
    """
    if not answerable:
        return refused and not cited
    if refused or not cited:
        return False
    if not set(cited) <= set(expected):
        return False
    return judge_supported is not False
