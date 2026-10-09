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

_The table above shows the output format only. These are illustrative numbers, not measured results. For real results, see [Support-eval example](#support-eval-example)._

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
| `judge_agreement.py` | Judge-vs-human agreement for one or more hand-label CSVs, with 95% bootstrap CIs. |
| `make_hard_labels.py` | Draws the harder set 2 sample (`results/hand_labels_set2.csv`) for hand labelling. |

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
Run `72d1c984-7286-42fd-a758-65b2c1163246` — 132 items, 0 failed requests, 0 unmatched to dataset, 0 judge fallbacks, 91.81s. Generated by `analyze_failures.py` from the raw run report.

**Overall (engine metrics, 0–1)**

| precision_at_k | recall_at_k | faithfulness | relevance | coherence | overall_score |
|---|---|---|---|---|---|
| 0.674 | 0.924 | 0.750 | 0.739 | 0.852 | 0.788 |

**By category**

| category | n | precision_at_k | recall_at_k | faithfulness | relevance | coherence | judge_mean | overall |
|---|---|---|---|---|---|---|---|---|
| account access | 21 | 0.556 | 0.905 | 0.810 | 0.786 | 0.881 | 0.825 | 0.787 |
| cancellations | 23 | 0.783 | 0.870 | 0.750 | 0.717 | 0.848 | 0.772 | 0.793 |
| host no-show | 24 | 0.597 | 0.917 | 0.677 | 0.677 | 0.823 | 0.726 | 0.738 |
| payment dispute | 19 | 0.719 | 0.947 | 0.658 | 0.658 | 0.816 | 0.711 | 0.760 |
| refunds | 22 | 0.894 | 1.000 | 0.727 | 0.705 | 0.830 | 0.754 | 0.831 |
| safety concern | 23 | 0.507 | 0.913 | 0.870 | 0.880 | 0.913 | 0.888 | 0.817 |

**By difficulty**

| difficulty | n | precision_at_k | recall_at_k | faithfulness | relevance | coherence | judge_mean | overall |
|---|---|---|---|---|---|---|---|---|
| hard | 32 | 0.531 | 0.781 | 0.688 | 0.625 | 0.797 | 0.703 | 0.684 |
| normal | 100 | 0.720 | 0.970 | 0.770 | 0.775 | 0.870 | 0.805 | 0.821 |

**By hard-item type**

| hard_type | n | precision_at_k | recall_at_k | faithfulness | relevance | coherence | judge_mean | overall |
|---|---|---|---|---|---|---|---|---|
| adversarial | 12 | 0.556 | 0.833 | 0.750 | 0.604 | 0.812 | 0.722 | 0.711 |
| ambiguous | 11 | 0.515 | 0.818 | 0.682 | 0.727 | 0.795 | 0.735 | 0.708 |
| none | 101 | 0.723 | 0.970 | 0.772 | 0.777 | 0.871 | 0.807 | 0.823 |
| out_of_scope | 8 | 0.458 | 0.625 | 0.562 | 0.469 | 0.750 | 0.594 | 0.573 |

**Failure modes (rule-based, an item can match several)**

| failure mode | count | % of items |
|---|---|---|
| rag_error (empty answer) | 0 | 0.0% |
| judge_fallback (unparsed, score=0.5) | 0 | 0.0% |
| missed_escalation (out-of-scope answered) | 0 | 0.0% |
| over_refusal (in-policy question refused) | 19 | 14.4% |
| wrong_retrieval (recall@k = 0, in-policy) | 7 | 5.3% |
| hallucinated_policy (faithfulness <= 2/5) | 21 | 15.9% |
| off_topic (relevance <= 2/5) | 31 | 23.5% |

**10 lowest-scoring items (by mean judge score)**

| # | category | difficulty | F/R/C (1–5) | question | judge reasoning |
|---|---|---|---|---|---|
| 1 | refunds | normal (none) | 1/1/3 | What documentation do I need to submit for an extenuating circumstance refund? | The answer does not reference any of the relevant information from the context regarding the documentation needed for an extenuating circumstance refund, makin… |
| 2 | cancellations | normal (none) | 2/1/3 | Can I cancel and get money back if my host has already been penalized? | The answer does not accurately reflect the policies outlined in the context regarding cancellations and refunds, leading to a low score in faithfulness and rel… |
| 3 | safety concern | normal (none) | 2/2/3 | What kind of evidence should I provide for safety concerns? | The answer does not accurately reflect the information provided in the context regarding the types of evidence to report safety concerns, leading to low scores… |
| 4 | payment dispute | normal (none) | 2/2/3 | What if my payment dispute is due to fraud? | The answer does not accurately reflect the provided context regarding payment disputes related to fraud, as it fails to mention the necessary steps for submitt… |
| 5 | host no-show | normal (none) | 2/2/3 | Is there a way to prevent host no-shows? | The answer does not accurately reflect the context provided, as it fails to address the prevention of host no-shows and instead suggests escalation, which is n… |
| 6 | host no-show | normal (none) | 2/2/3 | Is there any specific documentation I need for a host no-show? | The answer does not accurately reflect the context provided, as it fails to address the question about documentation for a host no-show and instead suggests es… |
| 7 | host no-show | normal (none) | 2/2/3 | My friend said they found a host who is more reliable; can I change my booking? | The answer does not accurately reflect the policies regarding changing a booking and fails to provide relevant information about cancellation and rebooking opt… |
| 8 | safety concern | normal (none) | 2/2/3 | Are there times when I shouldn’t report safety concerns? | The answer does not accurately reflect the policy regarding reporting safety concerns, as it suggests escalation to a human support agent instead of providing … |
| 9 | payment dispute | normal (none) | 2/2/3 | Can I dispute a payment if I changed my mind about the booking? | The answer does not accurately reflect the policies regarding payment disputes and fails to provide relevant information about the conditions under which a dis… |
| 10 | payment dispute | normal (none) | 2/2/3 | If I did not initiate the payment dispute myself, can I still be involved? | The answer does not accurately reflect the context provided, as it fails to address the specifics of the payment dispute process and incorrectly suggests that … |

<!-- support-eval-results:end -->

### Judge validation

These sections check how far the LLM judge's 1–5 scores agree with one human labeller (the repo author). There are two hand-labeled sets. **They used different reference contexts, so they are reported separately.**

| Set | File | Items | Labeled against |
|---|---|---|---|
| 1 | `results/hand_labels.csv` | 20 random items | The full policy document (`policy_docs.md`) |
| 2 | `results/hand_labels_set2.csv` | 15 new items, stratified toward hard cases | Only `retrieved_context`, the chunks the judge saw |

**Set 1 results** (output of `judge_agreement.py`, n=20 per axis)

| axis | exact | within ±1 | mean abs diff | kappa (pass ≥4) | bias (judge − human) | judge mean | human mean |
|---|---|---|---|---|---|---|---|
| faithfulness | 0.55 | 0.90 | 0.55 | 0.00 | −0.25 | 4.50 | 4.75 |
| relevance | 0.55 | 0.85 | 0.70 | 0.32 | +0.10 | 4.55 | 4.45 |
| coherence | 0.60 | 0.90 | 0.50 | 0.35 | +0.40 | 4.70 | 4.30 |

On these 20 items, the judge scored coherence higher than the human did and faithfulness slightly lower. Relevance is roughly unbiased.

**Why kappa is low here (the ceiling effect).** Kappa measures agreement on pass (score ≥ 4) versus fail beyond what chance would give. When almost every item passes, chance agreement is already close to 1, so a single disagreement on one of the few "fails" moves kappa a lot. Set 1 is mostly passes. For faithfulness, every human score in `hand_labels.csv` is 4 or 5, so the human never marks a "fail". In that case kappa is 0 no matter how closely the judge tracks the human. Faithfulness kappa 0.00 therefore says little about the judge. Exact, within-1 and bias are more informative for set 1, and with n=20 all of them are noisy.

**Set 2: a harder sample.** `make_hard_labels.py` draws 15 items that are not in set 1 from the same run report, using a fixed seed. It aims for 5 out-of-scope items, 5 ambiguous or adversarial items, and 5 normal items where the judge gave a 3 or below on at least one axis. This puts more likely "fails" into the sample, which is what kappa needs. Set 2 is labeled against `retrieved_context` only, not the full policy. The context then matches what the judge saw, so a disagreement reflects scoring rather than different information. The labelling instructions are in the header of `make_hard_labels.py`.

**Set 2 results: pending.** Set 2 has not been labeled yet. Once it is, `judge_agreement.py results/hand_labels.csv results/hand_labels_set2.csv` generates the results for each set and for both combined. Each metric comes with a 95% bootstrap confidence interval (2000 resamples, fixed seed), plus the share of items both raters scored 4–5. No set 2 numbers will appear here until that has been run on real labels.

### Limitations

- **Synthetic data.** gpt-4o-mini generated both the questions and the policy. They are cleaner and more uniform than real support tickets, and the "hard" items are hard in the ways the generator thought of.
- **Small n.** There are about 150 items (about 25 per category and about 5 hard items per category). Per-category differences of a few points are within noise, and there are no confidence intervals.
- **LLM-judge bias.** Faithfulness, relevance and coherence are single-sample 1–5 scores from one judge prompt, at temperature 0. Judges are known to favour fluent and longer answers, and coherence is close to saturated. If the judge reply can't be parsed, the score falls back to 0.5. These cases are counted as `judge_fallback`.
- **Same model family.** The generator, the RAG app's answer model and the judge are all gpt-4o-mini. This likely inflates agreement between the expected answers, the generated answers and the judge's notion of "good".
- **Crude retrieval metrics.** Precision and recall@k use token overlap between each chunk and the expected answer (threshold 0.3). Out-of-scope items have recall near 0 by construction, because their expected answer is an escalation message. That's why the table breaks results down by hard-item type and why `wrong_retrieval` excludes those items.
- **Heuristic failure modes.** Failure modes come from regex and threshold rules. They are not human-verified, and one item can match several.
- **Judge validation is thin.** Set 1 has 20 items labeled by one person and is dominated by passes, so kappa is unreliable there (see Judge validation). Set 2, the harder sample, is pending. Treat the judge scores in this section as rough signals.

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
