import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402

METRICS = {"precision_at_k": 0.8, "recall_at_k": 0.7, "faithfulness": 0.9,
           "relevance": 0.85, "coherence": 0.9, "overall_score": 0.83}


def run_payload(run_id="r1", metrics=None, items=None):
    return {"run_id": run_id, "dataset_id": "d", "status": "completed",
            "metrics": metrics or METRICS, "total_queries": 2,
            "duration_seconds": 1.0, "failed_queries": 0, "items": items or []}


def item(q, score, ok=True):
    return {"question": q, "expected_answer": "a", "answer": "x", "precision_at_k": score,
            "recall_at_k": score, "faithfulness": score, "relevance": score,
            "coherence": score, "judge_reasoning": f"r-{q}", "judge_ok": ok}


@pytest.fixture
def engine(monkeypatch):
    calls = []
    routes = {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.content))
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"detail": "not found"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    monkeypatch.setattr(server, "_transport", httpx.MockTransport(handler))
    return routes, calls


def test_compare_metrics_flags_regression():
    cand = {**METRICS, "faithfulness": 0.7}
    out = server.compare_metrics(METRICS, cand, tolerance=0.02)
    assert out["regressions"] == ["faithfulness"]
    assert out["passed"] is False
    assert out["deltas"]["faithfulness"] == -0.2


def test_compare_metrics_within_tolerance_passes():
    cand = {**METRICS, "faithfulness": 0.89}
    assert server.compare_metrics(METRICS, cand, tolerance=0.02)["passed"] is True


def test_compare_metrics_improvement_is_not_a_regression():
    cand = {k: v + 0.05 for k, v in METRICS.items()}
    assert server.compare_metrics(METRICS, cand)["regressions"] == []


def test_summarize_run_orders_worst_first_and_counts_judge_failures():
    run = run_payload(items=[item("good", 0.9), item("bad", 0.1, ok=False), item("mid", 0.5)])
    out = server.summarize_run(run, worst_n=2)
    assert [w["question"] for w in out["worst_questions"]] == ["bad", "mid"]
    assert out["judge_failures"] == 1


@pytest.mark.asyncio
async def test_engine_health(engine):
    routes, _ = engine
    routes[("GET", "/health")] = (200, {"status": "ok"})
    assert (await server.engine_health())["status"] == "ok"


@pytest.mark.asyncio
async def test_add_dataset_posts_payload(engine):
    routes, calls = engine
    routes[("POST", "/dataset/add")] = (200, {"count": 1})
    await server.add_dataset("d", [{"question": "q", "expected_answer": "a"}])
    body = json.loads(calls[-1][2])
    assert body["dataset_id"] == "d" and body["qa_pairs"][0]["question"] == "q"


@pytest.mark.asyncio
async def test_add_dataset_rejects_bad_pairs(engine):
    with pytest.raises(server.EngineError):
        await server.add_dataset("d", [])
    with pytest.raises(server.EngineError):
        await server.add_dataset("d", [{"question": "q"}])


@pytest.mark.asyncio
async def test_run_eval_returns_summary(engine):
    routes, calls = engine
    routes[("POST", "/eval/run")] = (200, run_payload(items=[item("q", 0.4)]))
    out = await server.run_eval("d", "http://rag/ask", top_k=5)
    assert out["worst_questions"][0]["question"] == "q"
    assert json.loads(calls[-1][2])["top_k"] == 5


@pytest.mark.asyncio
async def test_run_eval_validates_top_k(engine):
    with pytest.raises(server.EngineError):
        await server.run_eval("d", "http://rag", top_k=0)


@pytest.mark.asyncio
async def test_engine_error_surfaces_detail(engine):
    routes, _ = engine
    routes[("GET", "/eval/report/missing")] = (404, {"detail": "Run 'missing' not found"})
    with pytest.raises(server.EngineError, match="not found"):
        await server.get_report("missing")


@pytest.mark.asyncio
async def test_check_regression_end_to_end(engine):
    routes, _ = engine
    routes[("GET", "/eval/report/a")] = (200, run_payload("a"))
    routes[("GET", "/eval/report/b")] = (200, run_payload("b", {**METRICS, "recall_at_k": 0.5}))
    out = await server.check_regression("a", "b")
    assert out["passed"] is False and "recall_at_k" in out["regressions"]


@pytest.mark.asyncio
async def test_unreachable_engine_raises_engine_error(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(server, "_transport", httpx.MockTransport(boom))
    with pytest.raises(server.EngineError, match="Cannot reach"):
        await server.engine_health()


@pytest.mark.asyncio
async def test_tools_are_registered_with_mcp():
    names = {t.name for t in await server.mcp.list_tools()}
    assert {"engine_health", "list_datasets", "add_dataset", "run_eval", "get_report", "check_regression"} <= names
