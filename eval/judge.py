"""LLM-as-judge for groundedness and citation support.

A different model family judges the answers (qwen/qwen3.8-27b judges
openai/gpt-oss-120b answers) to reduce self-preference bias. Temperature
is 0 and the rubric is fixed. Judgments are cached, so re-running costs nothing.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from rag.cache import SQLiteCache, make_key
from rag.llm import make_chat_llm

JUDGE_SYSTEM = """You are a strict evaluator of a retrieval-augmented question-answering system for company policies.
You receive the CONTEXT passages that were retrieved (each labelled with a document ID), the QUESTION, and the ANSWER.

Decide two things:
1. grounded: true only if every factual statement in the ANSWER is fully supported by the CONTEXT. Any detail that is absent from or contradicted by the CONTEXT makes it false. Ignore citation brackets when judging this.
2. citations_supported: true only if each document ID cited in the ANSWER (in square brackets) labels a passage in the CONTEXT that actually supports the statement it is attached to. If the ANSWER cites nothing, set false.

Reply with JSON only, no other text:
{"grounded": true|false, "citations_supported": true|false, "unsupported_claims": ["..."], "reason": "one short sentence"}"""

JUDGE_USER = """CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}"""


def _parse(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # drop any visible reasoning
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("grounded"), bool) or not isinstance(data.get("citations_supported"), bool):
        return None
    return data


class Judge:
    def __init__(self, settings, cache: SQLiteCache | None = None):
        self.settings = settings
        self.cache = cache
        self.llm = make_chat_llm(settings, model=settings.judge_model, max_tokens=1500)

    def judge(self, question: str, answer: str, context: str) -> dict:
        key = make_key("judge-v1", self.settings.judge_model, question, answer, context)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit:
                return {**json.loads(hit), "cached": True}
        messages = [SystemMessage(JUDGE_SYSTEM),
                    HumanMessage(JUDGE_USER.format(context=context, question=question, answer=answer))]
        data = None
        for _ in range(2):  # one retry if the JSON is malformed
            data = _parse(self.llm.invoke(messages).content)
            if data:
                break
        if data is None:
            return {"grounded": None, "citations_supported": None, "reason": "judge output unparseable",
                    "unsupported_claims": [], "cached": False}
        if self.cache is not None:
            self.cache.set(key, json.dumps(data))
        return {**data, "cached": False}
