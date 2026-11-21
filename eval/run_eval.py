"""Run the evaluation set through the RAG pipeline and score it.

Usage
-----
python -m eval.run_eval                         # baseline, writes eval/results/baseline/
python -m eval.run_eval --tag k2 --set TOP_K=2  # an ablation run
python -m eval.run_eval --no-judge              # skip the LLM judge (auto metrics only)

Metrics (see design-and-evaluation.md)
--------------------------------------
groundedness       % of answers fully supported by the retrieved context (LLM judge;
                   correct refusals count as grounded)
citation_accuracy  % of answers whose citations point to the right passages
                   (auto check against expected sources + judge check)
refusal accuracy, key-fact exact/partial match, token F1, retrieval hit rate,
latency p50/p95 over fresh (non-cached) LLM calls.

Answers and judgments are cached in the LLM cache, so re-running is free.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import fields
from pathlib import Path

from rag.cache import SQLiteCache
from rag.config import ROOT, Settings, get_settings
from rag.ingest import build_index
from rag.pipeline import PolicyRAG

from .judge import Judge
from .metrics import citation_correct, key_fact_match, percentile_ms, token_f1

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
INDEX_FIELDS = {"chunk_size", "chunk_overlap", "embed_model"}


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def apply_overrides(settings: Settings, pairs: list[str], tag: str) -> Settings:
    """--set TOP_K=2 -> settings.top_k = 2 (types follow the Settings field)."""
    types = {f.name: f.type for f in fields(Settings)}
    changes = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        name = key.strip().lower()
        if name not in types:
            raise SystemExit(f"Unknown setting: {key}")
        current = getattr(settings, name)
        if isinstance(current, bool):
            changes[name] = value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int):
            changes[name] = int(value)
        elif isinstance(current, float):
            changes[name] = float(value)
        elif isinstance(current, Path):
            changes[name] = Path(value)
        else:
            changes[name] = value
    if INDEX_FIELDS & changes.keys():
        # A different chunking/embedding needs its own index.
        changes["chroma_dir"] = ROOT / f"chroma_db_{tag}"
    return settings.with_overrides(**changes)


def evaluate(settings: Settings, questions: list[dict], tag: str, use_judge: bool = True,
             sleep: float = 0.0, verbose: bool = True) -> dict:
    if not Path(settings.chroma_dir).exists():
        print(f"Building index for '{tag}' at {settings.chroma_dir} ...")
        build_index(settings)
    pipeline = PolicyRAG.from_settings(settings)
    judge = Judge(settings, SQLiteCache(settings.cache_path)) if use_judge else None

    rows = []
    for i, q in enumerate(questions, 1):
        res = pipeline.answer(q["question"])
        cited = [c["doc_id"] for c in res["citations"]]
        retrieved = [r["doc_id"] for r in res["retrieved"]]

        if res["refused"]:
            verdict = {"grounded": True, "citations_supported": not cited, "reason": "refusal", "cached": True}
        elif judge:
            verdict = judge.judge(q["question"], res["answer"], res.get("context", ""))
            if not verdict.get("cached") and sleep:
                time.sleep(sleep)
        else:
            verdict = {"grounded": None, "citations_supported": None, "reason": "not judged", "cached": True}

        exact, partial, frac = key_fact_match(res["answer"], q["must_include"]) if q["answerable"] else (False, False, 0.0)
        row = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answerable": q["answerable"],
            "answer": res["answer"],
            "refused": res["refused"],
            "refusal_reason": res.get("refusal_reason"),
            "cited": ";".join(cited),
            "citation_inferred": res.get("citation_inferred", False),
            "expected": ";".join(q["expected_doc_ids"]),
            "retrieved": ";".join(retrieved),
            "hit": bool(set(retrieved) & set(q["expected_doc_ids"])) if q["answerable"] else None,
            "best_score": res.get("best_score"),
            "grounded": verdict.get("grounded"),
            "judge_citations_supported": verdict.get("citations_supported"),
            "judge_reason": verdict.get("reason"),
            "citation_correct": citation_correct(q["answerable"], res["refused"], cited,
                                                 q["expected_doc_ids"], verdict.get("citations_supported")),
            "refusal_correct": res["refused"] == (not q["answerable"]),
            "key_facts_exact": exact,
            "key_facts_partial": partial,
            "key_facts_fraction": round(frac, 3),
            "token_f1": round(token_f1(res["answer"], q["gold_answer"]), 3) if q["answerable"] else None,
            "cached": res.get("cached", False),
            "llm_called": res.get("llm_called", False),
            "latency_ms": res["latency_ms"],
        }
        rows.append(row)
        if verbose:
            flag = "REFUSED" if row["refused"] else ("ok " if row["citation_correct"] else "CIT?")
            print(f"[{i:02d}/{len(questions)}] {q['id']} {flag:<7} grounded={row['grounded']} "
                  f"cited={row['cited'] or '-'} {row['latency_ms']:.0f} ms{' (cached)' if row['cached'] else ''}")
        if res.get("llm_called") and sleep:
            time.sleep(sleep)

    summary = summarize(rows, settings, tag)
    write_outputs(rows, summary, tag, settings)
    return summary


def _pct(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(100 * sum(bool(v) for v in values) / len(values), 1) if values else None


def summarize(rows: list[dict], settings: Settings, tag: str) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    fresh = [r["latency_ms"] for r in rows if r["llm_called"]]
    return {
        "tag": tag,
        "n_questions": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "groundedness_pct": _pct(r["grounded"] for r in rows),
        "citation_accuracy_pct": _pct(r["citation_correct"] for r in rows),
        "refusal_accuracy_unanswerable_pct": _pct(r["refused"] for r in unans),
        "false_refusal_pct": _pct(r["refused"] for r in ans),
        "key_facts_exact_pct": _pct(r["key_facts_exact"] for r in ans),
        "key_facts_partial_pct": _pct(r["key_facts_partial"] for r in ans),
        "mean_token_f1": round(sum(r["token_f1"] for r in ans) / len(ans), 3) if ans else None,
        "retrieval_hit_rate_pct": _pct(r["hit"] for r in ans),
        "inferred_citation_pct": _pct(r["citation_inferred"] for r in ans if not r["refused"]),
        "judged": sum(r["grounded"] is not None for r in rows),
        "latency_fresh_calls": len(fresh),
        "latency_p50_ms": percentile_ms(fresh, 50),
        "latency_p95_ms": percentile_ms(fresh, 95),
        "settings": {
            "llm_model": settings.llm_model, "judge_model": settings.judge_model,
            "embed_model": settings.embed_model, "use_reranker": settings.use_reranker,
            "top_k": settings.top_k, "fetch_k": settings.fetch_k, "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap, "min_relevance": settings.min_relevance,
            "prompt_variant": settings.prompt_variant,
        },
    }


def write_outputs(rows: list[dict], summary: dict, tag: str, settings: Settings) -> None:
    out = RESULTS_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Sample 10 answered questions for a human check of the judge (seeded).
    answered = [r for r in rows if not r["refused"]]
    sample = random.Random(settings.seed).sample(answered, min(10, len(answered)))
    with open(out / "human_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "cited", "judge_grounded",
                         "judge_citations_supported", "human_grounded", "human_citations_supported"])
        for r in sample:
            writer.writerow([r["id"], r["question"], r["answer"], r["cited"], r["grounded"],
                             r["judge_citations_supported"], "", ""])

    lines = [f"# Evaluation results: {tag}", "",
             "| Metric | Value |", "| --- | --- |"]
    labels = {
        "groundedness_pct": "Groundedness (%)",
        "citation_accuracy_pct": "Citation accuracy (%)",
        "refusal_accuracy_unanswerable_pct": "Refusal accuracy on unanswerable (%)",
        "false_refusal_pct": "False refusals on answerable (%)",
        "key_facts_exact_pct": "Key facts, exact match (%)",
        "key_facts_partial_pct": "Key facts, partial match (%)",
        "mean_token_f1": "Mean token F1 vs gold",
        "retrieval_hit_rate_pct": f"Retrieval hit rate @k={summary['settings']['top_k']} (%)",
        "latency_p50_ms": "Latency p50 (ms)",
        "latency_p95_ms": "Latency p95 (ms)",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {summary[key]} |")
    lines += ["", f"{summary['n_questions']} questions ({summary['n_answerable']} answerable, "
                  f"{summary['n_unanswerable']} unanswerable); {summary['judged']} judged; "
                  f"latency over {summary['latency_fresh_calls']} fresh LLM calls."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="seconds to pause after each fresh LLM call (Groq tokens/minute limit)")
    args = parser.parse_args(argv)

    settings = apply_overrides(get_settings(), args.overrides, args.tag)
    if not settings.llm_configured:
        raise SystemExit("GROQ_API_KEY is not set. Add it to .env first.")
    random.seed(settings.seed)
    questions = load_questions(args.questions)[: args.limit]
    summary = evaluate(settings, questions, args.tag, use_judge=not args.no_judge, sleep=args.sleep)
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))
    print(f"Wrote {RESULTS_DIR / args.tag}/(results.csv, summary.json, summary.md, human_check.csv)")
    return summary


if __name__ == "__main__":
    main()
