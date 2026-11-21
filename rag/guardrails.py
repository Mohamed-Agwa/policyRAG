"""Guardrails applied around the LLM.

1. Input validation (empty / overly long questions).
2. Out-of-corpus refusal (done in the pipeline before the LLM is called, and
   again here if the model itself refuses).
3. Citation validation: only document IDs that were actually retrieved are
   kept; invented IDs are removed from the answer text.
4. Output length limit (max_tokens at the API + a word cap here).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompts import REFUSAL

DOC_ID_RE = re.compile(r"POL-[A-Z]+-\d{3}")
BRACKET_RE = re.compile(r"\[([^\[\]]*POL-[A-Z]+-\d{3}[^\[\]]*)\]")
MAX_QUESTION_CHARS = 1000

_REFUSAL_PATTERNS = [
    "i can only answer about our policies",
    "couldn't find an answer to this in the nile byte policy",
    "could not find an answer to this in the nile byte policy",
]


class InvalidQuestion(ValueError):
    pass


def validate_question(question) -> str:
    if not isinstance(question, str) or not question.strip():
        raise InvalidQuestion("Please send a non-empty 'question' string.")
    question = " ".join(question.split())
    if len(question) > MAX_QUESTION_CHARS:
        raise InvalidQuestion(f"Questions are limited to {MAX_QUESTION_CHARS} characters.")
    return question


def is_refusal(text: str) -> bool:
    t = text.strip().lower().replace("’", "'")
    return any(p in t for p in _REFUSAL_PATTERNS)


def clean_citations(text: str, allowed_ids: set[str]) -> tuple[str, list[str]]:
    """Keep only allowed doc IDs inside [..] citations; return text + ordered IDs."""
    found: list[str] = []

    def repl(match: re.Match) -> str:
        ids = [i for i in DOC_ID_RE.findall(match.group(1)) if i in allowed_ids]
        for i in ids:
            if i not in found:
                found.append(i)
        return f"[{', '.join(dict.fromkeys(ids))}]" if ids else ""

    cleaned = BRACKET_RE.sub(repl, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), found


def truncate_words(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("]"), cut.rfind("!"), cut.rfind("?"))
    if end > len(cut) * 0.5:
        cut = cut[: end + 1]
    return cut.rstrip() + " …", True


@dataclass
class GuardedAnswer:
    answer: str
    refused: bool
    cited_ids: list[str] = field(default_factory=list)
    citation_inferred: bool = False
    truncated: bool = False


def postprocess(raw: str, retrieved_ids: list[str], max_words: int) -> GuardedAnswer:
    """Apply refusal detection, citation validation and the length cap."""
    raw = (raw or "").strip()
    if not raw or is_refusal(raw):
        return GuardedAnswer(answer=REFUSAL, refused=True)

    text, cited = clean_citations(raw, set(retrieved_ids))
    text, truncated = truncate_words(text, max_words)

    inferred = False
    if not cited and retrieved_ids:
        # The model answered without a valid citation. Never return an
        # uncited answer: attribute it to the top-ranked retrieved document
        # and flag the citation as inferred so the UI and evaluation know.
        cited = [retrieved_ids[0]]
        inferred = True
    return GuardedAnswer(answer=text, refused=False, cited_ids=cited,
                         citation_inferred=inferred, truncated=truncated)
