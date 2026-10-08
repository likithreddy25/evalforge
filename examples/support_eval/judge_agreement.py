"""
Judge-vs-human agreement on results/hand_labels.csv (fill in the human_* columns first).

Per axis: exact agreement, within-1 agreement, mean absolute difference (1–5 scale), and
Cohen's kappa on pass/fail (score >= 4). Rows with a blank human score are skipped.

    python judge_agreement.py [path/to/hand_labels.csv]
"""
import csv
import sys
from pathlib import Path

AXES = ["faithfulness", "relevance", "coherence"]
PASS = 4


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(rows: list[dict]) -> dict:
    out = {}
    for ax in AXES:
        pairs = []
        for r in rows:
            h = (r.get(f"human_{ax}_1to5") or "").strip()
            if h:
                pairs.append((int(r[f"judge_{ax}_1to5"]), int(h)))
        if not pairs:
            out[ax] = {"n": 0}
            continue
        n = len(pairs)
        out[ax] = {
            "n": n,
            "exact": sum(j == h for j, h in pairs) / n,
            "within_1": sum(abs(j - h) <= 1 for j, h in pairs) / n,
            "mean_abs_diff": sum(abs(j - h) for j, h in pairs) / n,
            "kappa_pass_fail": cohen_kappa([j >= PASS for j, _ in pairs], [h >= PASS for _, h in pairs]),
        }
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else Path(__file__).parent / "results" / "hand_labels.csv"
    rows = list(csv.DictReader(path.open()))
    res = agreement(rows)
    if all(v["n"] == 0 for v in res.values()):
        print(f"No human labels in {path} yet — agreement pending.")
        return
    print("| axis | n | exact | within ±1 | mean |diff| | kappa (pass ≥4) |\n|---|---|---|---|---|---|")
    for ax, v in res.items():
        if v["n"] == 0:
            print(f"| {ax} | 0 | – | – | – | – |")
            continue
        k = "n/a" if v["kappa_pass_fail"] is None else f"{v['kappa_pass_fail']:.2f}"
        print(f"| {ax} | {v['n']} | {v['exact']:.2f} | {v['within_1']:.2f} | {v['mean_abs_diff']:.2f} | {k} |")


if __name__ == "__main__":
    main()
