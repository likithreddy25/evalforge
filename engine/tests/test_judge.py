"""
Unit tests for LLM judge score parsing — no API calls needed.
"""
import pytest
from evaluators.judge import _parse_scores, _normalize


def test_parse_scores_valid():
    text = "FAITHFULNESS: 4\nRELEVANCE: 5\nCOHERENCE: 3"
    scores = _parse_scores(text)
    assert scores["faithfulness"] == pytest.approx(0.75, abs=0.01)
    assert scores["relevance"] == pytest.approx(1.0, abs=0.01)
    assert scores["coherence"] == pytest.approx(0.5, abs=0.01)


def test_parse_scores_fallback_on_bad_input():
    scores = _parse_scores("invalid response")
    assert scores["faithfulness"] == 0.5
    assert scores["relevance"] == 0.5
    assert scores["coherence"] == 0.5


def test_normalize_min():
    assert _normalize("1") == 0.0


def test_normalize_max():
    assert _normalize("5") == 1.0


def test_normalize_mid():
    assert _normalize("3") == pytest.approx(0.5, abs=0.01)


def test_normalize_invalid():
    assert _normalize("abc") == 0.5
