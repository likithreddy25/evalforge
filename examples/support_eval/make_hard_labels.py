"""
Draw a harder second hand-label sample (set 2) from an existing run report.

LABELLING INSTRUCTIONS FOR SET 2
    Score each item against the `retrieved_context` column ONLY. That column holds the chunks
    the RAG app retrieved for that question, which is the context the LLM judge saw. Do NOT
    use the full policy document (policy_docs.md). Set 1 (results/hand_labels.csv) was labeled
    against the full policy, so the two sets use different reference contexts and are
    reported separately.

    Faithfulness: is the answer supported by retrieved_context? (1 = invented, 5 = fully supported)
    Relevance:    does the answer address the question?         (1 = off-topic, 5 = direct answer)
    Coherence:    is the answer clear and well organized?       (1 = incoherent, 5 = clear)

Sampling: 15 items that are not in set 1, drawn with a fixed seed and stratified toward hard items:
    out_of_scope            hard_type == "out_of_scope"                                 (5)
    ambiguous_adversarial   hard_type in ("ambiguous", "adversarial")                   (5)
    normal_low_judge        difficulty == "normal" and the judge gave a score of 3 or
                            below on at least one of faithfulness/relevance/coherence    (5)
If a stratum has too few items, it takes all it has, and the shortfall is reported, not filled.

About retrieved_context: the report stores the chunks from the engine's retrieval call. The
judge's context comes from a second, identical query to the RAG app made during the same run.
ChromaDB retrieval is deterministic for the same query and index, so they are the same chunks.
The "[chunk N]" separators are added here for readability only.

    python make_hard_labels.py [--report results/run_<id>.json] [--seed 2]
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import analyze_failures

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SET1 = RESULTS / "hand_labels.csv"
SET2 = RESULTS / "hand_labels_set2.csv"

JUDGE = analyze_failures.JUDGE
STRATA = {"out_of_scope": 5, "ambiguous_adversarial": 5, "normal_low_judge": 5}
LOW = 3  # judge score on the 1-5 scale


def judge_1to5(row: dict, axis: str) -> int:
    return round(row[axis] * 4 + 1)


def stratum_of(row: dict) -> str | None:
    if row["hard_type"] == "out_of_scope":
        return "out_of_scope"
    if row["hard_type"] in ("ambiguous", "adversarial"):
        return "ambiguous_adversarial"
    if row["difficulty"] == "normal" and min(judge_1to5(row, a) for a in JUDGE) <= LOW:
        return "normal_low_judge"
    return None


def sample_set2(rows: list[dict], exclude_questions: set[str], seed: int,
                quotas: dict[str, int] = STRATA) -> tuple[list[dict], dict[str, int]]:
    """Return (sampled rows in stratum order, count drawn per stratum)."""
    pools: dict[str, list[dict]] = {s: [] for s in quotas}
    for r in rows:
        s = stratum_of(r)
        if s in pools and r["question"] not in exclude_questions:
            pools[s].append(r)
    rng = random.Random(seed)
    picked, counts = [], {}
    for s, quota in quotas.items():
        pool = sorted(pools[s], key=lambda r: r["question"])  # input order can't affect the draw
        got = rng.sample(pool, min(quota, len(pool)))
        counts[s] = len(got)
        picked.extend(got)
    return picked, counts


def format_context(chunks: list[str]) -> str:
    return "\n\n".join(f"[chunk {i}]\n{c}" for i, c in enumerate(chunks, 1))


def write_set2(rows: list[dict], path: Path) -> None:
    with SET1.open() as f:
        set1_cols = next(csv.reader(f))
    cols = set1_cols + ["retrieved_context"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {c: "" for c in cols}
            out.update({
                "question": r["question"], "category": r["category"], "difficulty": r["difficulty"],
                "hard_type": r["hard_type"], "expected_answer": r["expected_answer"], "rag_answer": r["answer"],
                "judge_reasoning": r.get("judge_reasoning", ""),
                "retrieved_context": format_context(r.get("retrieved_chunks", [])),
                **{f"judge_{a}_1to5": judge_1to5(r, a) for a in JUDGE},
            })
            w.writerow(out)  # human_* columns stay blank


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", type=Path, default=SET2)
    args = ap.parse_args(argv)

    if args.out.exists():
        sys.exit(f"{args.out} already exists; refusing to overwrite (it may contain your labels).")
    report_path = args.report or sorted(RESULTS.glob("run_*.json"), key=lambda p: p.stat().st_mtime)[-1]
    report = json.loads(report_path.read_text())
    dataset = json.loads((HERE / "dataset.json").read_text())
    rows, _ = analyze_failures.join(report, dataset)
    with SET1.open() as f:
        set1_questions = {r["question"] for r in csv.DictReader(f)}

    picked, counts = sample_set2(rows, set1_questions, args.seed)
    write_set2(picked, args.out)
    print(f"Report: {report_path.name}  seed={args.seed}  excluded {len(set1_questions)} set-1 items")
    for s, quota in STRATA.items():
        short = f"  (wanted {quota}, only {counts[s]} available)" if counts[s] < quota else ""
        print(f"  {s:<22} {counts[s]}{short}")
    print(f"  {'total':<22} {len(picked)}")
    print(f"Wrote {args.out}. Label it against retrieved_context only (see the header of this script).")


if __name__ == "__main__":
    main()
