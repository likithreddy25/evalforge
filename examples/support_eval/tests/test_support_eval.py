"""
Tests for the support-eval example. No network: the OpenAI client is mocked.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from generate_dataset import generate_category, quality_filter
from analyze_failures import aggregate_by, cluster_failures, join, write_hand_label_csv
from judge_agreement import agreement, cohen_kappa


def _item(q, a="Some answer.", cat="refunds", diff="normal"):
    return {"question": q, "expected_answer": a, "context": "", "category": cat, "difficulty": diff, "hard_type": None}


# ---- quality filter --------------------------------------------------------

def test_quality_filter_drops_empty_and_duplicates():
    items = [
        _item("How do I get a refund?"),
        _item("how do I get a refund"),        # duplicate after normalization
        _item("  HOW do I  get a refund?!  "),  # duplicate after normalization
        _item("Can I cancel?", a=""),           # empty answer
        _item("Can I cancel?", a="   "),        # whitespace-only answer
        _item("", a="orphan answer"),           # empty question
        _item("What if my host never shows up?"),
    ]
    kept, stats = quality_filter(items)
    assert [k["question"] for k in kept] == ["How do I get a refund?", "What if my host never shows up?"]
    assert stats == {"input": 7, "dropped_empty": 3, "dropped_duplicate": 2, "kept": 2}


def test_quality_filter_empty_answer_does_not_block_later_valid_duplicate():
    kept, stats = quality_filter([_item("Q1", a=""), _item("Q1", a="real answer")])
    assert len(kept) == 1 and kept[0]["expected_answer"] == "real answer"
    assert stats["dropped_empty"] == 1 and stats["dropped_duplicate"] == 0


def test_generate_category_with_mocked_api():
    payload = {"items": [
        {"question": "Refund for early checkout?", "expected_answer": "No refund for unused nights.",
         "difficulty": "normal", "hard_type": "adversarial"},
        {"question": "Can you refund my taxes?", "expected_answer": "I can't answer that; escalating.",
         "difficulty": "hard", "hard_type": "out_of_scope"},
        {"question": "x", "expected_answer": "y", "difficulty": "weird"},
    ]}
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    items = generate_category(client, "POLICY", "refunds", seed=7)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["seed"] == 7 and kwargs["model"] == "gpt-4o-mini"
    assert all(it["category"] == "refunds" and it["context"] == "" for it in items)
    assert [it["difficulty"] for it in items] == ["normal", "hard", "normal"]
    assert [it["hard_type"] for it in items] == [None, "out_of_scope", None]  # hard_type only kept on hard items


# ---- aggregation -----------------------------------------------------------

def _row(q, cat, diff, hard_type="none", p=1.0, r=1.0, f=1.0, rel=1.0, c=1.0, answer="ok"):
    return {"question": q, "category": cat, "difficulty": diff, "hard_type": hard_type, "answer": answer,
            "precision_at_k": p, "recall_at_k": r, "faithfulness": f, "relevance": rel, "coherence": c,
            "judge_ok": True, "expected_answer": "e", "judge_reasoning": ""}


def test_aggregate_by_category_and_difficulty():
    rows = [
        _row("a", "refunds", "normal", f=1.0, rel=1.0, c=1.0),
        _row("b", "refunds", "hard", f=0.0, rel=0.5, c=1.0, p=0.0, r=0.0),
        _row("c", "safety concern", "hard", f=0.5, rel=0.5, c=0.5),
    ]
    by_cat = {a["category"]: a for a in aggregate_by(rows, "category")}
    assert by_cat["refunds"]["n"] == 2
    assert by_cat["refunds"]["faithfulness"] == pytest.approx(0.5)
    assert by_cat["refunds"]["judge_mean"] == pytest.approx((1.0 + 0.5) / 2)
    assert by_cat["refunds"]["overall"] == pytest.approx((1.0 + (0 + 0 + 0 + 0.5 + 1) / 5) / 2)
    assert by_cat["safety concern"]["relevance"] == pytest.approx(0.5)

    by_diff = {a["difficulty"]: a for a in aggregate_by(rows, "difficulty")}
    assert by_diff["hard"]["n"] == 2 and by_diff["normal"]["n"] == 1
    assert by_diff["hard"]["precision_at_k"] == pytest.approx(0.5)
    assert [a["difficulty"] for a in aggregate_by(rows, "difficulty")] == ["hard", "normal"]


def test_join_attaches_metadata_and_counts_unmatched():
    report = {"items": [_row("q1", "", ""), _row("q-unknown", "", "")]}
    dataset = [{"question": "q1", "category": "refunds", "difficulty": "hard", "hard_type": "ambiguous"}]
    rows, unmatched = join(report, dataset)
    assert unmatched == 1
    assert rows[0]["category"] == "refunds" and rows[0]["hard_type"] == "ambiguous"


def test_cluster_failures_rules():
    rows = [
        _row("oos-answered", "refunds", "hard", hard_type="out_of_scope", r=0.0, answer="Sure, 50% back."),
        _row("oos-escalated", "refunds", "hard", hard_type="out_of_scope", r=0.0,
             answer="I can't answer that from policy; I'll escalate to a human agent."),
        _row("over-refusal", "refunds", "normal", answer="I'm unable to help with that."),
        _row("hallucinated", "refunds", "normal", f=0.25),
        _row("empty", "refunds", "normal", answer=""),
    ]
    c = {k.split(" ")[0]: [r["question"] for r in v] for k, v in cluster_failures(rows).items()}
    assert c["missed_escalation"] == ["oos-answered"]
    assert c["over_refusal"] == ["over-refusal"]
    assert c["hallucinated_policy"] == ["hallucinated"]
    assert c["rag_error"] == ["empty"]
    assert c["wrong_retrieval"] == []  # out-of-scope items are excluded from wrong_retrieval


def test_hand_label_csv_is_never_overwritten(tmp_path):
    rows = [_row(f"q{i}", "refunds", "normal") for i in range(30)]
    path = tmp_path / "hand_labels.csv"
    assert write_hand_label_csv(rows, path) is True
    path.write_text("my labels")
    assert write_hand_label_csv(rows, path) is False
    assert path.read_text() == "my labels"


# ---- judge agreement -------------------------------------------------------

def test_agreement_skips_blank_human_labels():
    rows = [
        {"judge_faithfulness_1to5": "5", "human_faithfulness_1to5": "5",
         "judge_relevance_1to5": "4", "human_relevance_1to5": "",
         "judge_coherence_1to5": "3", "human_coherence_1to5": "5"},
        {"judge_faithfulness_1to5": "2", "human_faithfulness_1to5": "3",
         "judge_relevance_1to5": "4", "human_relevance_1to5": "",
         "judge_coherence_1to5": "3", "human_coherence_1to5": ""},
    ]
    res = agreement(rows)
    assert res["faithfulness"]["n"] == 2
    assert res["faithfulness"]["exact"] == 0.5
    assert res["faithfulness"]["within_1"] == 1.0
    assert res["relevance"]["n"] == 0
    assert res["coherence"]["mean_abs_diff"] == 2.0


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert cohen_kappa([True, True], [True, True]) is None  # chance agreement = 1, undefined
