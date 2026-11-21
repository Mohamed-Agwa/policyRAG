"""Test doubles: deterministic embeddings and a fake LLM.

CI must not download models or call the Groq API (no key, no quota), so the
tests swap in a hashing bag-of-words embedder and a rule-based fake LLM. The
real pipeline code (retrieval, guardrails, citations, Flask routes) still runs.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

STOPWORDS = set("""a an and are as at be by can do does for from how i if in is it its me my of on or
our should that the their this to us was we what when where which who why will with you your""".split())
DIM = 512


class HashEmbeddings(Embeddings):
    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        v[0] = 1e-3  # avoid zero vectors for out-of-vocabulary questions
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in STOPWORDS or len(tok) < 3:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[1 + h % (DIM - 1)] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def make_fake_llm(fixed_answer: str | None = None):
    """A Runnable that mimics a chat model: answers citing the first context label."""

    def respond(prompt_value) -> AIMessage:
        if fixed_answer is not None:
            return AIMessage(content=fixed_answer)
        human = prompt_value.to_messages()[-1].content
        ids = re.findall(r"\[(POL-[A-Z]+-\d{3}) \|", human)
        if not ids:
            return AIMessage(content="I can only answer about our policies.")
        return AIMessage(content=f"The policy covers this topic in detail [{ids[0]}].")

    llm = RunnableLambda(respond)
    llm.is_fake = True
    return llm
