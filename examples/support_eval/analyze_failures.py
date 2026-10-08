"""
Failure-mode analysis for a support-eval run.

- Joins the engine's per-item results with category/difficulty from dataset.json
- Mean scores by category, difficulty, and hard_type
- 10 lowest-scoring items (by mean judge score) with the judge's reasoning
- Rule-based failure-mode clustering with counts
- Writes a 20-item hand-label CSV (only if it does not exist yet, so your labels are never overwritten)
- Writes results/summary.md and refreshes the results block in the repo README

    python analyze_failures.py                       # newest results/run_*.json
    python analyze_failures.py results/run_<id>.json
"""
import csv
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
README = HERE.parent.parent / "README.md"
README_START = "<!-- support-eval-results:start -->"
README_END = "<!-- support-eval-results:end -->"

METRICS = ["precision_at_k", "recall_at_k", "faithfulness", "relevance", "coherence"]
JUDGE = ["faithfulness", "relevance", "coherence"]

# "I can't answer this from policy" — wrong for in-policy questions
REFUSAL_RE = re.compile(
    r"can ?not (answer|help)|can't (answer|help)|unable to (answer|help|assist)|not covered|"
    r"(don't|do not) have (that|this|any|enough) information",
    re.I,
)
# Refusal or hand-off to a human — required for out-of-scope questions
ESCALATION_RE = re.compile(REFUSAL_RE.pattern + r"|escalat|human (support )?agent|support (agent|team)", re.I)

FAILURE_RULES = {
    "rag_error (empty answer)":
        lambda r: not r["answer"].strip(),
    "judge_fallback (unparsed, score=0.5)":
        lambda r: not r.get("judge_ok", True),
    "missed_escalation (out-of-scope answered)":
        lambda r: r["hard_type"] == "out_of_scope" and r["answer"].strip() and not ESCALATION_RE.search(r["answer"]),
    "over_refusal (in-policy question refused)":
        lambda r: r["difficulty"] == "normal" and bool(REFUSAL_RE.search(r["answer"])),
    "wrong_retrieval (recall@k = 0, in-policy)":
        lambda r: r["hard_type"] != "out_of_scope" and r["recall_at_k"] == 0,
    "hallucinated_policy (faithfulness <= 2/5)":
        lambda r: r["faithfulness"] <= 0.25,
    "off_topic (relevance <= 2/5)":
        lambda r: r["relevance"] <= 0.25,
}


def judge_mean(r: dict) -> float:
    return sum(r[m] for m in JUDGE) / len(JUDGE)


def overall(r: dict) -> float:
    return sum(r[m] for m in METRICS) / len(METRICS)


def join(report: dict, dataset: list[dict]) -> tuple[list[dict], int]:
    """Attach category/difficulty to each report item by question text. Returns (rows, unmatched)."""
    meta = {d["question"]: d for d in dataset}
    rows, unmatched = [], 0
    for it in report["items"]:
        d = meta.get(it["question"])
        if d is None:
            unmatched += 1
            continue
        rows.append({**it, "category": d["category"], "difficulty": d["difficulty"],
                     "hard_type": d.get("hard_type") or "none"})
    return rows, unmatched


def aggregate_by(rows: list[dict], key: str) -> list[dict]:
    """Mean of each metric (plus judge_mean and overall) per value of `key`, sorted by key."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out = []
    for k in sorted(groups):
        g = groups[k]
        agg = {key: k, "n": len(g)}
        for m in METRICS:
            agg[m] = sum(r[m] for r in g) / len(g)
        agg["judge_mean"] = sum(judge_mean(r) for r in g) / len(g)
        agg["overall"] = sum(overall(r) for r in g) / len(g)
        out.append(agg)
    return out


def cluster_failures(rows: list[dict]) -> dict[str, list[dict]]:
    return {name: [r for r in rows if rule(r)] for name, rule in FAILURE_RULES.items()}


def md_table(aggs: list[dict], key: str) -> str:
    cols = [key, "n", "precision_at_k", "recall_at_k", "faithfulness", "relevance", "coherence", "judge_mean", "overall"]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join(
        "| " + " | ".join(str(a[c]) if c in (key, "n") else f"{a[c]:.3f}" for c in cols) + " |\n" for a in aggs
    )
    return head + body


def _clip(s: str, n: int = 160) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def write_hand_label_csv(rows: list[dict], path: Path, n: int = 20, seed: int = 0) -> bool:
    if path.exists():
        return False
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "category", "difficulty", "hard_type", "expected_answer", "rag_answer",
                    "judge_faithfulness_1to5", "judge_relevance_1to5", "judge_coherence_1to5", "judge_reasoning",
                    "human_faithfulness_1to5", "human_relevance_1to5", "human_coherence_1to5", "human_notes"])
        for r in sample:
            w.writerow([r["question"], r["category"], r["difficulty"], r["hard_type"], r["expected_answer"],
                        r["answer"], *(round(r[m] * 4 + 1) for m in JUDGE), r.get("judge_reasoning", ""),
                        "", "", "", ""])
    return True


def build_summary(report: dict, rows: list[dict], unmatched: int) -> str:
    m = report["metrics"]
    clusters = cluster_failures(rows)
    lowest = sorted(rows, key=lambda r: (judge_mean(r), overall(r)))[:10]

    s = [f"Run `{report['run_id']}` — {report['total_queries']} items, "
         f"{report.get('failed_queries', 0)} failed requests, {unmatched} unmatched to dataset, "
         f"{sum(1 for r in rows if not r.get('judge_ok', True))} judge fallbacks, "
         f"{report['duration_seconds']}s. Generated by `analyze_failures.py` from the raw run report.\n\n"]
    s.append("**Overall (engine metrics, 0–1)**\n\n")
    s.append("| " + " | ".join(m) + " |\n|" + "---|" * len(m) + "\n| "
             + " | ".join(f"{v:.3f}" for v in m.values()) + " |\n")
    for key, title in [("category", "By category"), ("difficulty", "By difficulty"), ("hard_type", "By hard-item type")]:
        s.append(f"\n**{title}**\n\n")
        s.append(md_table(aggregate_by(rows, key), key))
    s.append("\n**Failure modes (rule-based, an item can match several)**\n\n")
    s.append("| failure mode | count | % of items |\n|---|---|---|\n")
    for name, hits in clusters.items():
        s.append(f"| {name} | {len(hits)} | {100 * len(hits) / max(len(rows), 1):.1f}% |\n")
    s.append("\n**10 lowest-scoring items (by mean judge score)**\n\n")
    s.append("| # | category | difficulty | F/R/C (1–5) | question | judge reasoning |\n|---|---|---|---|---|---|\n")
    for i, r in enumerate(lowest, 1):
        frc = "/".join(str(round(r[k] * 4 + 1)) for k in JUDGE)
        s.append(f"| {i} | {r['category']} | {r['difficulty']} ({r['hard_type']}) | {frc} | "
                 f"{_clip(r['question']).replace('|', '/')} | {_clip(r.get('judge_reasoning', '')).replace('|', '/')} |\n")
    return "".join(s)


def update_readme(summary: str) -> bool:
    if not README.exists():
        return False
    text = README.read_text()
    if README_START not in text or README_END not in text:
        return False
    pre, rest = text.split(README_START, 1)
    _, post = rest.split(README_END, 1)
    README.write_text(f"{pre}{README_START}\n{summary}\n{README_END}{post}")
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        run_path = Path(argv[0])
    else:
        runs = sorted(RESULTS.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
        if not runs:
            sys.exit("No results/run_*.json found — run run_eval.py first.")
        run_path = runs[-1]
    report = json.loads(run_path.read_text())
    if not report.get("items"):
        sys.exit(f"{run_path} has no per-item results (was it produced by an older engine?).")
    dataset = json.loads((HERE / "dataset.json").read_text())

    rows, unmatched = join(report, dataset)
    summary = build_summary(report, rows, unmatched)
    print(summary)

    (RESULTS / "summary.md").write_text(summary)
    print(f"Saved {RESULTS / 'summary.md'}")
    csv_path = RESULTS / "hand_labels.csv"
    if write_hand_label_csv(rows, csv_path):
        print(f"Wrote {csv_path} — fill in the human_* columns, then run judge_agreement.py")
    else:
        print(f"{csv_path} already exists; left untouched")
    if update_readme(summary):
        print(f"Updated results block in {README}")


if __name__ == "__main__":
    main()
