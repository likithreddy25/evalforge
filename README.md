# EvalForge

**EvalForge** is an open-source LLM and RAG pipeline evaluation framework. It measures retrieval quality and generation quality of AI agents using automated metrics and an LLM-as-judge scoring system.

```
evalforge run --dataset my-dataset --endpoint http://localhost:9000/query
```

```
╔══════════════════╦═════════╗
║ Metric           ║ Score   ║
╠══════════════════╬═════════╣
║ Precision@K      ║ 82.3%   ║
║ Recall@K         ║ 91.0%   ║
║ Faithfulness     ║ 78.5%   ║
║ Relevance        ║ 85.2%   ║
║ Coherence        ║ 88.0%   ║
║ Overall Score    ║ 85.0%   ║
╚══════════════════╩═════════╝
```

---

## Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────────┐
│  TypeScript CLI │ ────────────► │  Python FastAPI       │
│  (Commander.js) │               │  Evaluation Engine   │
└─────────────────┘               └──────────┬───────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                   ┌────────────┐   ┌──────────────┐   ┌──────────────┐
                   │  ChromaDB  │   │  RAG Endpoint│   │  OpenAI API  │
                   │ (datasets) │   │  (your app)  │   │  (LLM judge) │
                   └────────────┘   └──────────────┘   └──────────────┘
```

**Components:**
- **TypeScript CLI** — `evalforge run`, `evalforge report`, `evalforge dataset` commands
- **Python FastAPI engine** — retrieval + generation evaluators, run storage
- **ChromaDB** — persistent vector store for Q&A eval datasets
- **LLM-as-judge** — GPT-4o-mini scores faithfulness, relevance, and coherence

---

## Metrics

| Metric | Definition |
|---|---|
| **Precision@K** | Fraction of top-K retrieved chunks relevant to the expected answer |
| **Recall@K** | Whether the expected answer is covered in the top-K retrieved chunks |
| **Faithfulness** | Does the answer stay grounded in the retrieved context? (LLM judge, 1–5) |
| **Relevance** | Does the answer address the question? (LLM judge, 1–5) |
| **Coherence** | Is the answer well-structured and readable? (LLM judge, 1–5) |
| **Overall Score** | Mean of all 5 metrics, normalized to 0.0–1.0 |

---

## Quickstart

### Prerequisites
- Docker + Docker Compose
- Node.js 20+
- Python 3.11+
- OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/likithreddy25/evalforge.git
cd evalforge
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Start the engine

```bash
docker compose up --build
```

### 3. Install the CLI

```bash
cd cli
npm install
npm run build
npm link   # makes 'evalforge' available globally
```

### 4. Add a dataset

```bash
evalforge dataset add --file datasets/sample_qa.json --dataset-id my-dataset
```

### 5. Run an evaluation

```bash
evalforge run \
  --dataset my-dataset \
  --endpoint http://your-rag-app/query \
  --model gpt-4o-mini \
  --top-k 3
```

### 6. Fetch a report

```bash
evalforge report --run-id <run-id> --output report.json
```

---

## RAG Endpoint Contract

EvalForge sends a `POST` request to your RAG endpoint:

```json
{ "query": "What is RAG?", "top_k": 3 }
```

Expected response (any of these formats work):

```json
{ "answer": "RAG is...", "chunks": ["chunk1", "chunk2"] }
{ "response": "RAG is..." }
{ "results": ["chunk1", "chunk2"] }
```

---

## Support-eval example

`examples/support_eval/` is an end-to-end example on a customer-support RAG scenario for a fictional home-sharing marketplace ("Nestly"). It covers LLM-assisted synthetic data generation, an EvalForge run with retrieval metrics and the LLM judge, and a failure-mode analysis.

| File | What it does |
|---|---|
| `generate_dataset.py` | gpt-4o-mini writes a short policy (`policy_docs.md`), then 25 seeded Q&A items for each of 6 categories (refunds, cancellations, host no-show, safety concern, account access, payment dispute). About 20% are "hard": ambiguous, out-of-scope (the expected behaviour is to say it can't answer and escalate), or adversarial. A quality filter drops empty and duplicate items and logs the counts. The output is `dataset.json`, in the `datasets/sample_qa.json` format plus `category`, `difficulty` and `hard_type` fields. |
| `rag_app.py` | A tiny FastAPI RAG app over `policy_docs.md`. It uses the engine's ChromaDB setup and implements the endpoint contract (`POST /query` returns `answer` and `chunks`). |
| `run_eval.py` | Loads the dataset into the engine, runs the eval and saves the raw report to `results/run_<id>.json`. |
| `analyze_failures.py` | Mean scores by category, difficulty and hard-item type, the 10 lowest-scoring items with the judge's reasoning, and rule-based failure-mode counts. Writes `results/summary.md` and `results/hand_labels.csv` (20 random items) and refreshes the results block below. |
| `judge_agreement.py` | Judge-vs-human agreement after you fill in the `human_*` columns of `hand_labels.csv`. |

To support per-item analysis, the engine's run report now also has an `items` list (per-item scores, RAG answer, retrieved chunks, judge reasoning and a `judge_ok` flag) and a `failed_queries` count. The judge prompt asks for a one-line `REASONING`, and the engine runs at most 5 items at a time. The existing aggregate `metrics` are computed the same way as before.

### Running it

Run these from the repo root, with `OPENAI_API_KEY` exported:

```bash
pip install -r engine/requirements.txt
cd examples/support_eval && python generate_dataset.py && cd ../..
(cd engine && uvicorn main:app --port 8000) &
(cd examples/support_eval && uvicorn rag_app:app --port 9000) &
python examples/support_eval/run_eval.py
python examples/support_eval/analyze_failures.py
pytest examples/support_eval/tests -v
```

### Results

All numbers below are written by `analyze_failures.py` from a real run report in `examples/support_eval/results/`. None are typed in by hand. If this block says "not run yet", no results exist.

<!-- support-eval-results:start -->
_Not run yet._ Run the commands above to fill in this block.
<!-- support-eval-results:end -->

### Limitations

- **Synthetic data.** gpt-4o-mini generated both the questions and the policy. They are cleaner and more uniform than real support tickets, and the "hard" items are hard in the ways the generator thought of.
- **Small n.** There are about 150 items (about 25 per category and about 5 hard items per category). Per-category differences of a few points are within noise, and there are no confidence intervals.
- **LLM-judge bias.** Faithfulness, relevance and coherence are single-sample 1–5 scores from one judge prompt, at temperature 0. Judges are known to favour fluent and longer answers, and coherence is close to saturated. If the judge reply can't be parsed, the score falls back to 0.5. These cases are counted as `judge_fallback`.
- **Same model family.** The generator, the RAG app's answer model and the judge are all gpt-4o-mini. This likely inflates agreement between the expected answers, the generated answers and the judge's notion of "good".
- **Crude retrieval metrics.** Precision and recall@k use token overlap between each chunk and the expected answer (threshold 0.3). Out-of-scope items have recall near 0 by construction, because their expected answer is an escalation message. That's why the table breaks results down by hard-item type and why `wrong_retrieval` excludes those items.
- **Heuristic failure modes.** Failure modes come from regex and threshold rules. They are not human-verified, and one item can match several.
- **Judge agreement is pending.** `results/hand_labels.csv` has 20 random items with empty human columns. The judge's agreement with a human is unknown until those columns are filled in and `judge_agreement.py` is run.

The code for this example was written with the help of Claude (Anthropic) via Claude Code.

---

## Running Tests

```bash
cd engine
pip install -r requirements.txt
pytest tests/ -v
```

CI runs on every push via GitHub Actions — see `.github/workflows/ci.yml`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| CLI | TypeScript, Commander.js, Axios, Chalk |
| API | Python, FastAPI, Pydantic, Uvicorn |
| Vector Store | ChromaDB |
| LLM Judge | OpenAI API (gpt-4o-mini) |
| Containerization | Docker, Docker Compose |
| CI | GitHub Actions |

---

## License

MIT
