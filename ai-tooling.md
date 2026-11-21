# AI Tooling

> **Draft. Edit it into your own words before submitting.** Keep what is true for you, and add
> your own experiences in the places marked *[add]*.

## Tools used

| Tool | Used for |
| --- | --- |
| **Claude** (Anthropic), in the Claude app | Planning the architecture from the rubric and course material; drafting the synthetic policy corpus; writing the first version of the code, tests, CI workflow and evaluation harness; drafting documentation |
| *[add: e.g. VS Code + GitHub Copilot, Cursor, ChatGPT, or none]* | *[add]* |

## How I used them

1. **Planning.** I worked through the course lessons and the rubric, then used Claude to map
   each rubric requirement to course concepts and to propose an architecture. I made the key
   decisions myself:
   - Groq instead of OpenRouter, because of the free-tier limits;
   - deploying to Render;
   - the fictional company and policy topics;
   - including re-ranking and ablations.
2. **Corpus.** Claude drafted 12 policies from the topic list I approved. I reviewed them for
   consistency, for example making sure cross-references between policies point to the right
   IDs and that the numbers agree. *[add any edits you made]*
3. **Code.** Claude generated the initial implementation: ingestion, the RAG pipeline,
   guardrails, the Flask app, tests and the evaluation scripts. I reviewed it, ran it locally,
   and fixed or changed the following. *[add: e.g. bugs you hit, settings you tuned such as
   MIN_RELEVANCE, UI changes]*
4. **Evaluation.** I ran the evaluation and ablations myself, hand-checked 10 answers for the
   judge-agreement metric, and interpreted the results. *[add]*
5. **Deployment and CI.** I set up the GitHub repository, Actions secrets and the Render service
   myself. *[add any problems and fixes]*

## What worked well

- Turning the rubric into a concrete checklist and architecture quickly.
- Generating a large, internally consistent synthetic corpus, with gold evaluation questions
  checked against it automatically.
- Boilerplate-heavy parts: the Flask routes, the test fixtures with a fake LLM and fake
  embeddings, and the GitHub Actions workflow.
- *[add]*

## What didn't work well / limitations

- The assistant could not run anything that needed the internet from its own sandbox. Hugging
  Face model downloads were blocked, so the code was tested there only with fake embeddings and
  a fake LLM. Real end-to-end runs, the evaluation and deployment had to be done by me.
- Some generated details needed checking against current documentation: model names, rate
  limits and library APIs change quickly.
- *[add: anything that was wrong or needed rework]*

## Lessons

- Treat AI output like a pull request from a colleague: read it, run it and test it before
  accepting it.
- Pin versions and keep tests offline (fake LLM), so that AI-generated code is verifiable
  without spending API quota.
- *[add]*
