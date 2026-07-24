"""
Retrieval evaluators: precision@k and recall@k.
Queries the RAG endpoint and compares retrieved context against expected answers.
"""
import httpx
from typing import Optional


async def query_rag_endpoint(endpoint: str, question: str, top_k: int) -> list[str]:
    """
    Call the RAG endpoint and return a list of retrieved chunks.
    Expected response format: {"chunks": ["chunk1", "chunk2", ...]}
    Falls back to {"answer": "..."} for simple QA endpoints.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(endpoint, json={"query": question, "top_k": top_k})
        resp.raise_for_status()
        data = resp.json()
        if "chunks" in data:
            return data["chunks"]
        elif "answer" in data:
            return [data["answer"]]
        elif "results" in data:
            return data["results"]
        return []


def precision_at_k(retrieved: list[str], expected: str, k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant to the expected answer."""
    if not retrieved:
        return 0.0
    relevant = sum(1 for chunk in retrieved[:k] if _is_relevant(chunk, expected))
    return relevant / min(k, len(retrieved))


def recall_at_k(retrieved: list[str], expected: str, k: int) -> float:
    """Whether the expected answer is covered in top-k retrieved chunks."""
    if not retrieved:
        return 0.0
    for chunk in retrieved[:k]:
        if _is_relevant(chunk, expected):
            return 1.0
    return 0.0


def _is_relevant(chunk: str, expected: str, threshold: float = 0.3) -> bool:
    """Token overlap similarity between chunk and expected answer."""
    chunk_tokens = set(chunk.lower().split())
    expected_tokens = set(expected.lower().split())
    if not expected_tokens:
        return False
    overlap = chunk_tokens & expected_tokens
    return len(overlap) / len(expected_tokens) >= threshold


async def evaluate_retrieval(
    endpoint: str,
    question: str,
    expected_answer: str,
    top_k: int,
) -> dict:
    try:
        retrieved = await query_rag_endpoint(endpoint, question, top_k)
    except Exception:
        retrieved = []

    return {
        "precision_at_k": precision_at_k(retrieved, expected_answer, top_k),
        "recall_at_k": recall_at_k(retrieved, expected_answer, top_k),
        "retrieved_chunks": retrieved,
    }
