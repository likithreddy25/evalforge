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
