"""Run the ablation grid and write eval/results/ablations.md.

Usage:  python -m eval.ablations            (all variants)
        python -m eval.ablations k2 no_rerank   (selected variants)

Each variant is a full evaluation run (29 answers + up to 24 judge calls).
Answers and judgments are cached, so variants that produce the same prompt as
an earlier run (and any re-run) cost nothing. Expect roughly 10 minutes per
fresh variant on Groq's free plan because of the tokens-per-minute limit.
"""

from __future__ import annotations

import json
import sys

from .run_eval import RESULTS_DIR, main as run_eval

VARIANTS: dict[str, tuple[str, list[str]]] = {
    "baseline": ("k=4, chunk 800/100, re-ranker on, cited prompt", []),
    "k2": ("top-k = 2", ["TOP_K=2"]),
    "k6": ("top-k = 6", ["TOP_K=6"]),
    "chunk500": ("chunk size 500, overlap 50", ["CHUNK_SIZE=500", "CHUNK_OVERLAP=50"]),
    "chunk1200": ("chunk size 1200, overlap 150", ["CHUNK_SIZE=1200", "CHUNK_OVERLAP=150"]),
    "no_rerank": ("re-ranker off (vector order only)", ["USE_RERANKER=false"]),
    "plain_prompt": ("prompt without grounding/citation rules", ["PROMPT_VARIANT=plain"]),
}

COLUMNS = [
    ("groundedness_pct", "Grounded %"),
    ("citation_accuracy_pct", "Citation acc. %"),
    ("refusal_accuracy_unanswerable_pct", "Refusal acc. %"),
    ("key_facts_exact_pct", "Key facts exact %"),
    ("mean_token_f1", "Token F1"),
    ("retrieval_hit_rate_pct", "Hit rate %"),
    ("latency_p50_ms", "p50 ms"),
]


def write_table() -> None:
    lines = ["# Ablation results", "",
             "| Variant | Change | " + " | ".join(c[1] for c in COLUMNS) + " |",
             "| --- | --- | " + " | ".join("---" for _ in COLUMNS) + " |"]
    for tag, (desc, _) in VARIANTS.items():
        path = RESULTS_DIR / tag / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"| {tag} | {desc} | " + " | ".join(str(s.get(k)) for k, _ in COLUMNS) + " |")
    lines += ["", "Latency p50 is computed over fresh LLM calls only; cached re-runs show `None`."]
    (RESULTS_DIR / "ablations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    selected = argv if argv else list(VARIANTS)
    for tag in selected:
        if tag not in VARIANTS:
            raise SystemExit(f"Unknown variant {tag}; choose from {list(VARIANTS)}")
        print(f"\n=== {tag}: {VARIANTS[tag][0]} ===")
        args = ["--tag", tag]
        for pair in VARIANTS[tag][1]:
            args += ["--set", pair]
        run_eval(args)
    write_table()


if __name__ == "__main__":
    main(sys.argv[1:])
