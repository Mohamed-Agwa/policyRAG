"""Judge vs human agreement.

1. Open eval/results/baseline/human_check.csv.
2. For each row, read the answer and the cited policy (follow /docs/<id>), and fill
   human_grounded and human_citations_supported with true or false.
3. Run: python -m eval.agreement   (or --tag <name>)
"""

from __future__ import annotations

import argparse
import csv
import json

from .run_eval import RESULTS_DIR


def as_bool(v: str):
    v = (v or "").strip().lower()
    return {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}.get(v)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args(argv)
    path = RESULTS_DIR / args.tag / "human_check.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))

    def agree(judge_col, human_col):
        pairs = [(as_bool(r[judge_col]), as_bool(r[human_col])) for r in rows]
        pairs = [(j, h) for j, h in pairs if j is not None and h is not None]
        return (round(100 * sum(j == h for j, h in pairs) / len(pairs), 1) if pairs else None), len(pairs)

    g, ng = agree("judge_grounded", "human_grounded")
    c, nc = agree("judge_citations_supported", "human_citations_supported")
    result = {"groundedness_agreement_pct": g, "n_grounded": ng,
              "citation_agreement_pct": c, "n_citations": nc}
    (RESULTS_DIR / args.tag / "agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not ng:
        print("Fill in the human_* columns in", path, "first.")
    return result


if __name__ == "__main__":
    main()
