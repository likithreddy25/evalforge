"""
Unit tests for retrieval evaluators.
These run without any external API — pure logic tests.
"""
import pytest
from evaluators.retrieval import precision_at_k, recall_at_k, _is_relevant


def test_precision_at_k_perfect():
    retrieved = ["The answer is RAG combines retrieval and generation"]
    expected = "RAG combines retrieval and generation"
    assert precision_at_k(retrieved, expected, k=3) == 1.0


def test_precision_at_k_empty():
    assert precision_at_k([], "anything", k=3) == 0.0


def test_recall_at_k_found():
    retrieved = ["RAG is a retrieval augmented generation technique"]
    expected = "retrieval augmented generation"
    assert recall_at_k(retrieved, expected, k=3) == 1.0


def test_recall_at_k_not_found():
    retrieved = ["completely unrelated chunk about cooking"]
    expected = "retrieval augmented generation"
    assert recall_at_k(retrieved, expected, k=3) == 0.0


def test_is_relevant_high_overlap():
    assert _is_relevant(
        "RAG combines retrieval and generation for grounded answers",
        "RAG retrieval generation",
    ) is True


def test_is_relevant_low_overlap():
    assert _is_relevant(
        "pasta recipe with tomatoes",
        "machine learning model evaluation",
    ) is False


def test_precision_at_k_partial():
    retrieved = [
        "RAG retrieval generation grounded",
        "completely unrelated content",
        "more unrelated content",
    ]
    expected = "RAG retrieval generation"
    score = precision_at_k(retrieved, expected, k=3)
    assert 0.0 < score <= 1.0
