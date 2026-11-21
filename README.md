# Nile Byte Policy Assistant

![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)

A Retrieval-Augmented Generation (RAG) app that answers employee questions about the
company policies of **Nile Byte Technologies** (a fictional 300-person software company in
Cairo), with citations, source snippets and links to the exact policy section.

- **Corpus:** 12 policies (10 Markdown, 1 HTML, 1 PDF), about 15,300 words (around 50 pages)
- **Stack:** Flask, LangChain (LCEL), Chroma, FastEmbed (`all-MiniLM-L6-v2` embeddings and an
  `ms-marco-MiniLM-L-6-v2` cross-encoder re-ranker on ONNX Runtime), Groq free API
  (`openai/gpt-oss-120b`)
- **Guardrails:** out-of-corpus refusal, citation validation, length limits, prompt-injection
  resistance
- **Evaluation:** 29 questions; groundedness, citation accuracy, latency p50/p95, key-fact
  match, token F1, hit rate, judge-vs-human agreement and 7 ablations
- **CI/CD:** GitHub Actions on every push/PR, with optional deploy to Render after tests pass

See [design-and-evaluation.md](design-and-evaluation.md) for design decisions and results.

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python **3.11** | Tested with 3.11. Check with `python3.11 --version` |
| Git | To clone the repo |
| A free **Groq API key** | Create one at <https://console.groq.com/keys> |
| Internet access | The first run downloads the embedding (~90 MB) and re-ranker (~80 MB) models from Hugging Face |

## 2. Set up (once)

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install the pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure your API key
cp .env.example .env                 # Windows: copy .env.example .env
# then edit .env and set GROQ_API_KEY=gsk_...
```

**Windows (PowerShell):** `python3.11` usually doesn't exist on Windows. Use the Python launcher instead:

```powershell
py -3.11 -m venv .venv          # or: python -m venv .venv  (if `python --version` shows 3.11)
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   (once), then retry
pip install -r requirements.txt
copy .env.example .env
```

## 3. Build the index

```bash
python -m rag.ingest
```

This parses and cleans the 12 policy files in `data/policies/`, splits them into heading-aware
chunks (800 characters, 100 overlap), embeds them, stores them in `chroma_db/`, and downloads
the models into `.models/`. Expected output:

```
Corpus: 12 documents (html, md, pdf), 15,280 words (~51 pages), 216 chunks at chunk_size=800, overlap=100
  POL-FIN-001  md     1300 words  Travel and Expense Reimbursement Policy
  ...
Indexed 216 chunks from 12 documents into .../chroma_db
```

Re-run this command whenever you change a policy file or `CHUNK_SIZE` / `CHUNK_OVERLAP`.

## 4. Run the app

```bash
python app.py                        # development server on http://127.0.0.1:5000
# or, the production server used on Render:
gunicorn app:app --workers 1 --threads 4 --bind 127.0.0.1:5000
```

Open <http://127.0.0.1:5000> and ask, for example, *"How many leave days can I carry over?"*

### API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Chat interface |
| POST | `/chat` | Body `{"question": "..."}`; returns the answer, citations and snippets |
| GET | `/health` | JSON status: index size, model, whether the API key is configured |
| GET | `/docs/<doc_id>` | Rendered policy; citation links jump to the cited section |
| GET | `/docs/<doc_id>/raw` | Original file (md, html or pdf) |
| GET | `/api/docs` | List of policies |

```bash
curl -s -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the hotel limit for a trip to London?"}'
```

Example response (shortened):

```json
{
  "answer": "The maximum hotel rate in London is USD 250 per night, excluding taxes [POL-FIN-001].",
  "refused": false,
  "citations": [{
    "doc_id": "POL-FIN-001",
    "title": "Travel and Expense Reimbursement Policy",
    "section": "5. Business Travel > 5.3 Hotels",
    "snippet": "| Location | Maximum rate per night | ... | International, high-cost cities (London, New York, ...) | USD 250 | ...",
    "url": "/docs/POL-FIN-001#53-hotels"
  }],
  "retrieved": [{"doc_id": "POL-FIN-001", "section": "...", "vector_score": 0.61, "rerank_score": 7.9}],
  "cached": false,
  "latency_ms": 1350.2,
  "model": "openai/gpt-oss-120b"
}
```

`curl http://127.0.0.1:5000/health` returns, for example:

```json
{"status": "ok", "index_chunks": 216, "llm_configured": true, "model": "openai/gpt-oss-120b", ...}
```

## 5. Run the tests

```bash
pytest -q
```

The 31 tests use a deterministic fake embedder and a fake LLM, so they need no API key and no
model downloads. They cover ingestion of all three formats, chunk metadata, the guardrails,
every endpoint, citation validation, caching and the evaluation metrics. The same tests run in
GitHub Actions on every push and pull request (`.github/workflows/ci.yml`).

## 6. Run the evaluation

Requires `GROQ_API_KEY` and a built index. Responses and judgments are cached in
`cache/llm_cache.sqlite`, so re-running costs no API calls.

```bash
# 1) Main evaluation: 29 questions, LLM judge (about 10 minutes on Groq's free plan)
python -m eval.run_eval
#    -> eval/results/baseline/{summary.md, summary.json, results.csv, human_check.csv}

# 2) Latency, end to end over HTTP, with caching disabled (start the app first in another terminal)
python -m eval.latency --url http://127.0.0.1:5000 --n 20
#    -> eval/results/latency_local.json
#    After deploying: python -m eval.latency --url https://<your-app>.onrender.com --n 20

# 3) Ablations: k, chunk size, re-ranker, prompt (about 1 hour the first time)
python -m eval.ablations
#    -> eval/results/ablations.md

# 4) Judge vs human agreement: fill in the human_* columns of
#    eval/results/baseline/human_check.csv (true/false), then:
python -m eval.agreement
```

Paste the tables from `eval/results/` into `design-and-evaluation.md` (section 6).

If Groq returns rate-limit errors (HTTP 429), the client retries with backoff automatically. You
can also raise the pause between calls with `python -m eval.run_eval --sleep 5`.

## 7. Deploy to Render (free)

1. Push the repo to GitHub.
2. In Render: **New > Blueprint**, select the repo. Render reads `render.yaml`.
3. When asked, enter the `GROQ_API_KEY` value. The other settings come from `render.yaml`.
4. Wait for the build. It installs dependencies, builds the index and downloads the models.
   The first build takes about 5 to 10 minutes.
5. Open `https://<your-app>.onrender.com/health` and check that you see `"status": "ok"`.
6. Put the URL in `deployed.md`.

**Optional: deploy only after CI passes.** In Render, go to Settings, copy the **Deploy Hook**
URL and set **Auto-Deploy** to Off. In GitHub, go to Settings > Secrets and variables > Actions
and add `RENDER_DEPLOY_HOOK_URL`. The `deploy` job in `ci.yml` then triggers a deploy after the
tests pass on `main`.

**Free-tier notes.**
- The service sleeps after 15 idle minutes, and waking takes about a minute. Open the URL a
  minute before demoing.
- If the instance runs out of memory, set `USE_RERANKER=false` in the Render environment.

## 8. Configuration

All settings are environment variables. See `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | (none) | Groq API key (required for answers) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Answer model |
| `JUDGE_MODEL` | `qwen/qwen3.8-27b` | Evaluation judge model (a different model family) |
| `TOP_K` / `FETCH_K` | 4 / 10 | Chunks given to the LLM / candidates before re-ranking |
| `USE_RERANKER` | true | Cross-encoder re-ranking |
| `MIN_RELEVANCE` | 0.30 | Refuse without calling the LLM if the best cosine similarity is below this |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 100 | Chunking (characters) |
| `MAX_TOKENS` / `MAX_ANSWER_WORDS` | 1024 / 180 | API token cap (includes the model's hidden reasoning) / visible answer word cap |
| `USE_LLM_CACHE` | true | Cache LLM responses in SQLite |

## 9. Project structure

```
app.py                     Flask app (/, /chat, /health, /docs)
rag/
  config.py                settings from environment variables
  ingest.py                parse md/html/pdf -> clean -> chunk -> embed -> Chroma
  embeddings.py            FastEmbed (ONNX) embeddings for LangChain
  rerank.py                cross-encoder re-ranker
  prompts.py               system prompt, refusal text, context formatting
  guardrails.py            input validation, refusal detection, citation validation, length cap
  pipeline.py              retrieve -> re-rank -> guard -> prompt | LLM (LCEL) -> guard
  llm.py, cache.py         Groq client (OpenAI-compatible) and SQLite response cache
templates/, static/        chat UI and the source-document viewer
data/policies/             the corpus (12 files)
corpus_src/                Markdown source of the PDF policy (scripts/build_pdf.py)
eval/                      questions, evaluation, judge, latency, ablations, agreement
tests/                     pytest suite (fake LLM + fake embeddings)
.github/workflows/ci.yml   CI (and optional Render deploy)
render.yaml                Render blueprint
```

## 10. Troubleshooting

| Problem | Fix |
| --- | --- |
| `No index at .../chroma_db` | Run `python -m rag.ingest` |
| `GROQ_API_KEY is not set` | Add the key to `.env` (local) or the Render environment |
| Model download fails | Check your internet connection or proxy; the models come from huggingface.co |
| Groq `404 model ... does not exist` or `model_decommissioned` | The model was retired. Check <https://console.groq.com/docs/deprecations> and set `LLM_MODEL` / `JUDGE_MODEL` in `.env` to current models |
| HTTP 429 from Groq | Wait a minute; the client retries automatically. Use `--sleep` for evaluations |
| Render out of memory | Set `USE_RERANKER=false` |
