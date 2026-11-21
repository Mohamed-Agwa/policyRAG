"""Measure end-to-end latency of POST /chat (request -> answer), cache disabled.

Usage
-----
python -m eval.latency                                   # in-process Flask test client
python -m eval.latency --url http://127.0.0.1:5000       # a running local server
python -m eval.latency --url https://<your-app>.onrender.com --n 20

Each request sends {"no_cache": true}, so every query makes a real LLM call
(except refusals, which are answered without the LLM by design). One warm-up
request is sent first and excluded, so model loading and cold starts don't
distort the numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

from .metrics import percentile_ms
from .run_eval import EVAL_DIR, RESULTS_DIR, load_questions


def post(url: str | None, client, question: str) -> tuple[float, dict]:
    body = {"question": question, "no_cache": True}
    t0 = time.perf_counter()
    if client is not None:
        resp = client.post("/chat", json=body)
        data = resp.get_json()
    else:
        req = urllib.request.Request(url.rstrip("/") + "/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000, data


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="base URL of a running app; default: in-process")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=3.0, help="pause between requests (rate limits)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    client = None
    if args.url is None:
        from app import create_app

        client = create_app().test_client()

    questions = load_questions(EVAL_DIR / "questions.jsonl")
    random.Random(args.seed).shuffle(questions)
    questions = questions[: args.n]

    post(args.url, client, "How many annual leave days do I get?")  # warm-up, not measured
    samples = []
    for i, q in enumerate(questions, 1):
        ms, data = post(args.url, client, q["question"])
        samples.append({"id": q["id"], "ms": round(ms, 1), "refused": data.get("refused"),
                        "llm_called": data.get("llm_called")})
        print(f"[{i:02d}/{len(questions)}] {q['id']} {ms:7.0f} ms {'(refused)' if data.get('refused') else ''}")
        time.sleep(args.sleep)

    all_ms = [s["ms"] for s in samples]
    llm_ms = [s["ms"] for s in samples if s["llm_called"]]
    result = {
        "target": args.url or "in-process",
        "n": len(samples),
        "p50_ms": percentile_ms(all_ms, 50),
        "p95_ms": percentile_ms(all_ms, 95),
        "mean_ms": round(sum(all_ms) / len(all_ms), 1),
        "max_ms": max(all_ms),
        "n_llm_calls": len(llm_ms),
        "p50_llm_ms": percentile_ms(llm_ms, 50),
        "p95_llm_ms": percentile_ms(llm_ms, 95),
        "samples": samples,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = "latency_local.json" if args.url is None or "127.0.0.1" in args.url or "localhost" in args.url \
        else "latency_deployed.json"
    Path(RESULTS_DIR / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    print(f"Wrote {RESULTS_DIR / name}")
    return result


if __name__ == "__main__":
    main()
