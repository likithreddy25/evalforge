import time
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from models.schemas import (
    DatasetAddRequest,
    EvalRunRequest,
    EvalRunResponse,
    EvalMetrics,
)
from store.chroma import add_qa_pairs, get_all_qa_pairs, list_datasets
from evaluators.retrieval import evaluate_retrieval
from evaluators.generation import evaluate_generation

load_dotenv()

app = FastAPI(
    title="EvalForge Engine",
    description="LLM & RAG pipeline evaluation API",
    version="0.1.0",
)

# In-memory run store (use Redis/Postgres for production)
_run_store: dict[str, EvalRunResponse] = {}


@app.get("/health")
def health():
    return {"status": "ok", "service": "evalforge-engine"}


@app.post("/dataset/add")
def add_dataset(req: DatasetAddRequest):
    pairs = [p.model_dump() for p in req.qa_pairs]
    count = add_qa_pairs(req.dataset_id, pairs)
    return {"message": "Dataset added", "dataset_id": req.dataset_id, "count": count}


@app.get("/dataset/list")
def list_all_datasets():
    return {"datasets": list_datasets()}


@app.post("/eval/run", response_model=EvalRunResponse)
async def run_eval(req: EvalRunRequest):
    # Load dataset
    qa_pairs = get_all_qa_pairs(req.dataset_id)
    if not qa_pairs:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found or empty")

    start = time.time()
    retrieval_scores = []
    generation_scores = []

    # Run evals concurrently in batches of 5
    async def eval_pair(pair: dict):
        r = await evaluate_retrieval(
            endpoint=req.rag_endpoint,
            question=pair["question"],
            expected_answer=pair["expected_answer"],
            top_k=req.top_k,
        )
        g = await evaluate_generation(
            endpoint=req.rag_endpoint,
            question=pair["question"],
            expected_answer=pair["expected_answer"],
            context=pair.get("context", ""),
            model=req.model,
            top_k=req.top_k,
        )
        return r, g

    tasks = [eval_pair(p) for p in qa_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            continue
        r, g = result
        retrieval_scores.append(r)
        generation_scores.append(g)

    n = len(retrieval_scores) or 1
    avg = lambda key, lst: sum(d[key] for d in lst) / len(lst) if lst else 0.0

    precision = avg("precision_at_k", retrieval_scores)
    recall = avg("recall_at_k", retrieval_scores)
    faithfulness = avg("faithfulness", generation_scores)
    relevance = avg("relevance", generation_scores)
    coherence = avg("coherence", generation_scores)
    overall = (precision + recall + faithfulness + relevance + coherence) / 5

    metrics = EvalMetrics(
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        faithfulness=round(faithfulness, 4),
        relevance=round(relevance, 4),
        coherence=round(coherence, 4),
        overall_score=round(overall, 4),
    )

    run_id = str(uuid.uuid4())
    response = EvalRunResponse(
        run_id=run_id,
        status="completed",
        dataset_id=req.dataset_id,
        metrics=metrics,
        total_queries=len(qa_pairs),
        duration_seconds=round(time.time() - start, 2),
    )
    _run_store[run_id] = response
    return response


@app.get("/eval/report/{run_id}", response_model=EvalRunResponse)
def get_report(run_id: str):
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _run_store[run_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
