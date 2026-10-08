"""
Load dataset.json into the EvalForge engine, run an eval against the support-eval RAG app,
and save the raw run report to results/run_<run_id>.json.

Uses the same engine HTTP API as the CLI (`evalforge dataset add` + `evalforge run`).

    python run_eval.py --engine http://localhost:8000 --endpoint http://localhost:9000/query
"""
import argparse
import hashlib
import json
from pathlib import Path

import httpx

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://localhost:8000")
    ap.add_argument("--endpoint", default="http://localhost:9000/query")
    ap.add_argument("--dataset", type=Path, default=HERE / "dataset.json")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    raw = args.dataset.read_bytes()
    pairs = json.loads(raw)
    # Content-addressed ID so re-running never appends duplicate rows to an existing collection
    dataset_id = "support-eval-" + hashlib.sha256(raw).hexdigest()[:8]

    with httpx.Client(base_url=args.engine, timeout=3600) as c:
        existing = c.get("/dataset/list").raise_for_status().json()["datasets"]
        if dataset_id not in existing:
            r = c.post("/dataset/add", json={"dataset_id": dataset_id, "qa_pairs": pairs}).raise_for_status()
            print(f"Added dataset {dataset_id}: {r.json()['count']} items")
        else:
            print(f"Dataset {dataset_id} already loaded")

        print("Running eval (this makes ~3 LLM calls per item; expect a few minutes)...")
        report = c.post("/eval/run", json={
            "dataset_id": dataset_id,
            "rag_endpoint": args.endpoint,
            "model": "gpt-4o-mini",
            "top_k": args.top_k,
        }).raise_for_status().json()

    out = HERE / "results" / f"run_{report['run_id']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2))
    print(f"total_queries={report['total_queries']} failed={report.get('failed_queries')} "
          f"duration={report['duration_seconds']}s")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
