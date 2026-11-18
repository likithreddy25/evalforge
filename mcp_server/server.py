"""EvalForge MCP server.

Exposes the EvalForge engine (FastAPI) as Model Context Protocol tools so an
AI coding agent can build a dataset, run an eval against a RAG endpoint, and
check a run for regressions without leaving the editor.

Run:  python server.py            (stdio transport)
Env:  EVALFORGE_URL  engine base URL (default http://localhost:8000)
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = os.environ.get("EVALFORGE_URL", "http://localhost:8000")
TIMEOUT = float(os.environ.get("EVALFORGE_TIMEOUT", "300"))
METRICS = ("precision_at_k", "recall_at_k", "faithfulness", "relevance", "coherence", "overall_score")

mcp = FastMCP("evalforge")

# Overridable in tests (httpx.MockTransport).
_transport: httpx.AsyncBaseTransport | None = None


class EngineError(RuntimeError):
    """Engine returned an error or could not be reached."""


async def _call(method: str, path: str, **kwargs: Any) -> dict:
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT, transport=_transport) as client:
            resp = await client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise EngineError(f"Cannot reach EvalForge engine at {BASE_URL}: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise EngineError(f"Engine returned {resp.status_code}: {detail}")
    return resp.json()


def compare_metrics(baseline: dict, candidate: dict, tolerance: float = 0.02) -> dict:
    """Pure function: per-metric delta and which metrics regressed beyond tolerance."""
    deltas, regressions = {}, []
    for key in METRICS:
        delta = round(candidate.get(key, 0.0) - baseline.get(key, 0.0), 4)
        deltas[key] = delta
        if delta < -tolerance:
            regressions.append(key)
    return {"deltas": deltas, "regressions": regressions, "passed": not regressions, "tolerance": tolerance}


def summarize_run(run: dict, worst_n: int = 3) -> dict:
    """Pure function: headline metrics plus the lowest-scoring questions."""
    items = run.get("items", [])
    scored = sorted(
        items,
        key=lambda i: (i["faithfulness"] + i["relevance"] + i["coherence"] + i["precision_at_k"] + i["recall_at_k"]),
    )
    return {
        "run_id": run["run_id"],
        "dataset_id": run["dataset_id"],
        "metrics": run["metrics"],
        "total_queries": run["total_queries"],
        "failed_queries": run.get("failed_queries", 0),
        "judge_failures": sum(1 for i in items if not i.get("judge_ok", True)),
        "worst_questions": [
            {"question": i["question"], "answer": i.get("answer", ""), "reasoning": i.get("judge_reasoning", "")}
            for i in scored[:worst_n]
        ],
    }


@mcp.tool()
async def engine_health() -> dict:
    """Check that the EvalForge engine is reachable."""
    return await _call("GET", "/health")


@mcp.tool()
async def list_datasets() -> dict:
    """List datasets stored in the engine."""
    return await _call("GET", "/dataset/list")


@mcp.tool()
async def add_dataset(dataset_id: str, qa_pairs: list[dict]) -> dict:
    """Add question/answer pairs to a dataset.

    qa_pairs: list of {"question": str, "expected_answer": str, "context": str (optional)}
    """
    if not qa_pairs:
        raise EngineError("qa_pairs must not be empty")
    for p in qa_pairs:
        if not p.get("question") or not p.get("expected_answer"):
            raise EngineError("each pair needs 'question' and 'expected_answer'")
    return await _call("POST", "/dataset/add", json={"dataset_id": dataset_id, "qa_pairs": qa_pairs})


@mcp.tool()
async def run_eval(dataset_id: str, rag_endpoint: str, model: str = "gpt-4o-mini", top_k: int = 3) -> dict:
    """Run an eval of a RAG endpoint over a dataset. Returns a summary with the worst questions."""
    if not 1 <= top_k <= 10:
        raise EngineError("top_k must be between 1 and 10")
    run = await _call(
        "POST", "/eval/run",
        json={"dataset_id": dataset_id, "rag_endpoint": rag_endpoint, "model": model, "top_k": top_k},
    )
    return summarize_run(run)


@mcp.tool()
async def get_report(run_id: str) -> dict:
    """Fetch a finished run's summary by id."""
    return summarize_run(await _call("GET", f"/eval/report/{run_id}"))


@mcp.tool()
async def check_regression(baseline_run_id: str, candidate_run_id: str, tolerance: float = 0.02) -> dict:
    """Compare two runs; fails any metric that dropped by more than `tolerance`."""
    base = await _call("GET", f"/eval/report/{baseline_run_id}")
    cand = await _call("GET", f"/eval/report/{candidate_run_id}")
    return compare_metrics(base["metrics"], cand["metrics"], tolerance)


if __name__ == "__main__":
    mcp.run()
