"""
Generation evaluator — orchestrates retrieval + LLM judge for each Q&A pair.
"""
import httpx
from .judge import judge_response


async def get_rag_answer(endpoint: str, question: str, top_k: int) -> tuple[str, list[str]]:
    """Query RAG endpoint, return (answer, chunks)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(endpoint, json={"query": question, "top_k": top_k})
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", data.get("response", ""))
            chunks = data.get("chunks", data.get("results", []))
            return answer, chunks
        except Exception:
            return "", []


async def evaluate_generation(
    endpoint: str,
    question: str,
    expected_answer: str,
    context: str,
    model: str,
    top_k: int,
) -> dict:
    answer, chunks = await get_rag_answer(endpoint, question, top_k)
    combined_context = context + "\n".join(chunks) if chunks else context
    scores = await judge_response(
        question=question,
        context=combined_context,
        answer=answer,
        expected=expected_answer,
        model=model,
    )
    return {**scores, "answer": answer, "chunks": chunks}
