"""
Judge-vs-human agreement on one or more hand-label CSVs (fill in the human_* columns first).

Per axis: exact agreement, within-1 agreement, mean absolute difference (1–5 scale), and
Cohen's kappa on pass/fail (score >= 4). Rows with a blank human score are skipped.

For each file, and for all labeled files combined when there is more than one, it also prints
those metrics plus signed bias (judge minus human) with 95% bootstrap confidence intervals
(2000 resamples of items, fixed seed, percentile method), judge and human means, and the share
of items both raters scored 4 or 5.

    python judge_agreement.py [path/to/hand_labels.csv ...]   # default: results/hand_labels.csv
"""
import csv
import math
import random
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


N_RESAMPLES = 2000
SEED = 0


def _pairs(rows: list[dict], ax: str) -> list[tuple[int, int]]:
    out = []
    for r in rows:
        h = (r.get(f"human_{ax}_1to5") or "").strip()
        if h:
            out.append((int(r[f"judge_{ax}_1to5"]), int(h)))
    return out


# Each statistic maps a list of (judge, human) pairs to a number, or None when undefined
STATS = {
    "exact": lambda p: sum(j == h for j, h in p) / len(p),
    "within_1": lambda p: sum(abs(j - h) <= 1 for j, h in p) / len(p),
    "mean_abs_diff": lambda p: sum(abs(j - h) for j, h in p) / len(p),
    "kappa_pass_fail": lambda p: cohen_kappa([j >= PASS for j, _ in p], [h >= PASS for _, h in p]),
    "bias": lambda p: sum(j - h for j, h in p) / len(p),  # judge minus human
}


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 1]."""
    k = max(0, min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[k]


def bootstrap_ci(pairs: list[tuple[int, int]], stats: dict = STATS, n_resamples: int = N_RESAMPLES,
                 seed: int = SEED, alpha: float = 0.05) -> dict:
    """Point estimate and percentile bootstrap CI for each statistic.

    All statistics share the same resampled index sets. Resamples where a statistic is undefined
    (kappa when every item lands in one class) are dropped for that statistic and counted.
    Returns {stat: {"point", "lo", "hi", "undefined"}}.
    """
    rng = random.Random(seed)
    n = len(pairs)
    draws = {name: [] for name in stats}
    undefined = {name: 0 for name in stats}
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        for name, fn in stats.items():
            v = fn(sample)
            if v is None:
                undefined[name] += 1
            else:
                draws[name].append(v)
    out = {}
    for name, fn in stats.items():
        vals = sorted(draws[name])
        out[name] = {
            "point": fn(pairs),
            "lo": _percentile(vals, alpha / 2) if vals else None,
            "hi": _percentile(vals, 1 - alpha / 2) if vals else None,
            "undefined": undefined[name],
        }
    return out


def describe(rows: list[dict]) -> dict:
    """Per axis: n, bootstrap CIs, judge/human means, and the share both raters scored >= PASS."""
    out = {}
    for ax in AXES:
        p = _pairs(rows, ax)
        if not p:
            out[ax] = {"n": 0}
            continue
        out[ax] = {
            "n": len(p),
            "ci": bootstrap_ci(p),
            "judge_mean": sum(j for j, _ in p) / len(p),
            "human_mean": sum(h for _, h in p) / len(p),
            "both_pass": sum(j >= PASS and h >= PASS for j, h in p) / len(p),
        }
    return out


def _fmt(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def _ci_cell(c: dict, signed: bool = False) -> str:
    return f"{_fmt(c['point'], signed)} [{_fmt(c['lo'], signed)}, {_fmt(c['hi'], signed)}]"


def print_basic_table(res: dict) -> None:
    print("| axis | n | exact | within ±1 | mean |diff| | kappa (pass ≥4) |\n|---|---|---|---|---|---|")
    for ax, v in res.items():
        if v["n"] == 0:
            print(f"| {ax} | 0 | – | – | – | – |")
            continue
        k = "n/a" if v["kappa_pass_fail"] is None else f"{v['kappa_pass_fail']:.2f}"
        print(f"| {ax} | {v['n']} | {v['exact']:.2f} | {v['within_1']:.2f} | {v['mean_abs_diff']:.2f} | {k} |")


def print_ci_table(desc: dict) -> None:
    print(f"\nWith 95% bootstrap CIs ({N_RESAMPLES} resamples, seed {SEED}):\n")
    print("| axis | n | exact | within ±1 | mean |diff| | kappa (pass ≥4) | bias (judge − human) |"
          "\n|---|---|---|---|---|---|---|")
    for ax, v in desc.items():
        if v["n"] == 0:
            print(f"| {ax} | 0 | – | – | – | – | – |")
            continue
        c = v["ci"]
        print(f"| {ax} | {v['n']} | {_ci_cell(c['exact'])} | {_ci_cell(c['within_1'])} | "
              f"{_ci_cell(c['mean_abs_diff'])} | {_ci_cell(c['kappa_pass_fail'])} | {_ci_cell(c['bias'], True)} |")
    print("\n| axis | judge mean | human mean | both scored 4–5 | kappa undefined in resamples |\n|---|---|---|---|---|")
    for ax, v in desc.items():
        if v["n"]:
            print(f"| {ax} | {v['judge_mean']:.2f} | {v['human_mean']:.2f} | {v['both_pass']:.0%} | "
                  f"{v['ci']['kappa_pass_fail']['undefined']}/{N_RESAMPLES} |")


NOTES = """
Notes:
- Kappa is unreliable when almost every item passes (score >= 4). With so few failures, one
  disagreement on a rare "fail" moves kappa a lot, and kappa can be 0 or negative even when raw
  agreement is high (the prevalence or ceiling effect). Read it next to "both scored 4–5" and
  exact/within-1 agreement, not on its own.
- n is small, so the confidence intervals are wide. The percentile bootstrap also understates
  uncertainty at small n. Treat these as rough indications only."""


def _has_labels(rows: list[dict]) -> bool:
    return any(_pairs(rows, ax) for ax in AXES)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paths = [Path(a) for a in argv] or [Path(__file__).parent / "results" / "hand_labels.csv"]
    labeled = []
    for path in paths:
        rows = list(csv.DictReader(path.open()))
        if len(paths) > 1:
            print(f"\n## {path.name}\n")
        if not _has_labels(rows):
            print(f"No human labels in {path} yet — agreement pending.")
            continue
        labeled.append(rows)
        print_basic_table(agreement(rows))
        print_ci_table(describe(rows))

    if len(paths) > 1:
        print("\n## Combined (all labeled files)\n")
        if len(labeled) < 2:
            print("Pending: fewer than two files have human labels.")
        else:
            combined = [r for rows in labeled for r in rows]
            print("Caution: if the files were labeled against different reference contexts, the combined "
                  "figures mix two labelling protocols.\n")
            print_basic_table(agreement(combined))
            print_ci_table(describe(combined))
    if labeled:
        print(NOTES)


if __name__ == "__main__":
    main()
