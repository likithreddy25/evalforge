"""
Tests for the set-2 stratified sampler and the bootstrap agreement statistics. No network.
"""
import csv

import pytest

import judge_agreement as ja
import make_hard_labels as mhl


# ---- stratified sampler ------------------------------------------------------

def _row(q, difficulty="normal", hard_type="none", f=1.0, r=1.0, c=1.0):
    return {"question": q, "category": "refunds", "difficulty": difficulty, "hard_type": hard_type,
            "expected_answer": "e", "answer": "a", "retrieved_chunks": ["c1", "c2"], "judge_reasoning": "",
            "faithfulness": f, "relevance": r, "coherence": c}


@pytest.fixture
def rows():
    out = []
    out += [_row(f"oos{i}", "hard", "out_of_scope") for i in range(8)]
    out += [_row(f"amb{i}", "hard", "ambiguous") for i in range(4)]
    out += [_row(f"adv{i}", "hard", "adversarial") for i in range(4)]
    out += [_row(f"low{i}", f=0.5) for i in range(7)]            # judge faithfulness 3/5 -> low
    out += [_row(f"high{i}", f=0.75, r=0.75, c=0.75) for i in range(10)]  # all 4/5 -> not sampled
    out += [_row("hard-none", "hard", "none", f=0.0)]             # hard but no type -> not sampled
    return out


def test_stratum_assignment(rows):
    by_q = {r["question"]: mhl.stratum_of(r) for r in rows}
    assert by_q["oos0"] == "out_of_scope"
    assert by_q["amb0"] == by_q["adv0"] == "ambiguous_adversarial"
    assert by_q["low0"] == "normal_low_judge"
    assert by_q["high0"] is None and by_q["hard-none"] is None


def test_sampler_counts_per_stratum_and_no_overlap_with_set1(rows):
    set1 = {"oos0", "oos1", "oos2", "amb0", "low0"}
    picked, counts = mhl.sample_set2(rows, set1, seed=2)
    assert counts == {"out_of_scope": 5, "ambiguous_adversarial": 5, "normal_low_judge": 5}
    qs = [r["question"] for r in picked]
    assert len(qs) == len(set(qs)) == 15
    assert not set(qs) & set1
    assert all(mhl.stratum_of(r) is not None for r in picked)


def test_sampler_reports_shortfall_without_filling(rows):
    set1 = {f"oos{i}" for i in range(6)}  # leaves only 2 out-of-scope items
    picked, counts = mhl.sample_set2(rows, set1, seed=2)
    assert counts["out_of_scope"] == 2 and len(picked) == 12


def test_sampler_is_reproducible_and_order_independent(rows):
    a, _ = mhl.sample_set2(rows, set(), seed=7)
    b, _ = mhl.sample_set2(list(reversed(rows)), set(), seed=7)
    c, _ = mhl.sample_set2(rows, set(), seed=8)
    assert [r["question"] for r in a] == [r["question"] for r in b]
    assert [r["question"] for r in a] != [r["question"] for r in c]


def test_write_set2_columns_and_blank_human_labels(rows, tmp_path, monkeypatch):
    set1 = tmp_path / "hand_labels.csv"
    cols = ["question", "category", "difficulty", "hard_type", "expected_answer", "rag_answer",
            "judge_faithfulness_1to5", "judge_relevance_1to5", "judge_coherence_1to5", "judge_reasoning",
            "human_faithfulness_1to5", "human_relevance_1to5", "human_coherence_1to5", "human_notes"]
    set1.write_text(",".join(cols) + "\n")
    monkeypatch.setattr(mhl, "SET1", set1)
    out = tmp_path / "set2.csv"
    mhl.write_set2([rows[0], rows[16]], out)
    got = list(csv.DictReader(out.open()))
    assert list(got[0].keys()) == cols + ["retrieved_context"]
    assert got[0]["retrieved_context"] == "[chunk 1]\nc1\n\n[chunk 2]\nc2"
    assert got[1]["judge_faithfulness_1to5"] == "3"
    assert all(g[c] == "" for g in got for c in cols if c.startswith("human_"))


def test_main_refuses_to_overwrite_existing_set2(tmp_path):
    out = tmp_path / "hand_labels_set2.csv"
    out.write_text("my labels")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        mhl.main(["--out", str(out)])
    assert out.read_text() == "my labels"


# ---- bootstrap ----------------------------------------------------------------

def test_bootstrap_perfect_agreement():
    pairs = [(5, 5), (4, 4), (2, 2), (5, 5), (3, 3)]
    ci = ja.bootstrap_ci(pairs)
    for name, lo_hi in {"exact": 1.0, "within_1": 1.0, "mean_abs_diff": 0.0, "bias": 0.0}.items():
        assert ci[name]["point"] == ci[name]["lo"] == ci[name]["hi"] == lo_hi
    assert ci["kappa_pass_fail"]["point"] == 1.0
    assert ci["kappa_pass_fail"]["lo"] == ci["kappa_pass_fail"]["hi"] == 1.0  # whenever defined
    assert ci["kappa_pass_fail"]["undefined"] > 0  # resamples with only passes (or only fails)


def test_bootstrap_hand_computed_case():
    # judge 5,4,3,5 vs human 5,5,3,3
    pairs = [(5, 5), (4, 5), (3, 3), (5, 3)]
    ci = ja.bootstrap_ci(pairs)
    assert ci["exact"]["point"] == 0.5             # 2 of 4 equal
    assert ci["within_1"]["point"] == 0.75         # diffs 0,1,0,2
    assert ci["mean_abs_diff"]["point"] == 0.75    # (0+1+0+2)/4
    assert ci["bias"]["point"] == 0.25             # (0-1+0+2)/4
    # pass: judge T,T,F,T  human T,T,F,F -> po=.75, pe=.75*.5+.25*.5=.5, kappa=(.75-.5)/.5
    assert ci["kappa_pass_fail"]["point"] == pytest.approx(0.5)
    for c in ci.values():
        assert c["lo"] <= c["point"] <= c["hi"]
    assert ci["bias"]["lo"] >= -1 and ci["bias"]["hi"] <= 2  # bounded by the per-item diffs


def test_bootstrap_two_item_distribution_is_exact():
    # diffs 0 and 2: a resampled mean is 0, 1 or 2 with probability 1/4, 1/2, 1/4,
    # so the 2.5th and 97.5th percentiles are 0 and 2
    ci = ja.bootstrap_ci([(3, 3), (5, 3)])
    assert (ci["bias"]["lo"], ci["bias"]["point"], ci["bias"]["hi"]) == (0.0, 1.0, 2.0)


def test_bootstrap_is_seeded():
    pairs = [(5, 4), (4, 4), (3, 5), (5, 5), (2, 4), (4, 3)]
    assert ja.bootstrap_ci(pairs) == ja.bootstrap_ci(pairs)
    assert ja.bootstrap_ci(pairs, seed=1) != ja.bootstrap_ci(pairs, seed=2)


# ---- multi-file CLI ----------------------------------------------------------

def _write_labels(path, rows):
    cols = ["question"] + [f"{who}_{ax}_1to5" for who in ("judge", "human") for ax in ja.AXES]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, (j, h) in enumerate(rows):
            w.writerow({"question": f"q{i}", **{f"judge_{ax}_1to5": j for ax in ja.AXES},
                        **{f"human_{ax}_1to5": h for ax in ja.AXES}})


def test_cli_multiple_files_per_set_and_combined(tmp_path, capsys):
    a, b = tmp_path / "set1.csv", tmp_path / "set2.csv"
    _write_labels(a, [(5, 5), (4, 5), (3, 3)])
    _write_labels(b, [(5, 3), (2, 2), (4, ""), (5, 5)])  # blank human row is skipped
    ja.main([str(a), str(b)])
    out = capsys.readouterr().out
    assert "## set1.csv" in out and "## set2.csv" in out and "## Combined" in out
    assert "| faithfulness | 3 | 0.67 |" in out          # set 1
    assert "| faithfulness | 3 | 0.67 | 0.67 |" in out   # set 2: (5,3),(2,2),(5,5)
    assert "| faithfulness | 6 | 0.67 |" in out          # combined, blank row skipped
    assert "prevalence or ceiling effect" in out and "n is small" in out


def test_cli_unlabeled_second_file_is_pending(tmp_path, capsys):
    a, b = tmp_path / "set1.csv", tmp_path / "set2.csv"
    _write_labels(a, [(5, 5), (4, 5)])
    _write_labels(b, [(5, ""), (4, "")])
    ja.main([str(a), str(b)])
    out = capsys.readouterr().out
    assert "No human labels" in out and "agreement pending" in out
    assert "Pending: fewer than two files have human labels." in out


def test_cli_single_file_keeps_original_table(tmp_path, capsys):
    a = tmp_path / "labels.csv"
    _write_labels(a, [(5, 5), (4, 5), (3, 3)])
    ja.main([str(a)])
    out = capsys.readouterr().out
    assert out.startswith("| axis | n | exact | within ±1 | mean |diff| | kappa (pass ≥4) |")
    assert "## " not in out
