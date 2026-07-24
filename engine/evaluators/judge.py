"""
LLM-as-judge evaluator.
Uses GPT-4o-mini to score generation quality on three axes:
  - Faithfulness: does the answer stay grounded in the retrieved context?
  - Relevance: does the answer address the question?
  - Coherence: is the answer well-structured and readable?
Each axis is scored 1-5, normalized to 0.0-1.0.
"""
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT = """You are an expert evaluator of LLM-generated answers. Score the following answer on three axes.

Question: {question}
Context: {context}
Generated Answer: {answer}
Expected Answer: {expected}

Score each axis from 1 to 5:
- Faithfulness (1-5): Does the answer stay grounded in the provided context? 1=completely hallucinated, 5=fully grounded
- Relevance (1-5): Does the answer address the question? 1=completely off-topic, 5=directly answers the question
- Coherence (1-5): Is the answer well-structured and readable? 1=incoherent, 5=clear and well-organized

Respond in this exact format:
FAITHFULNESS: <score>
RELEVANCE: <score>
COHERENCE: <score>"""


async def judge_response(
    question: str,
    context: str,
    answer: str,
    expected: str,
    model: str = "gpt-4o-mini",
) -> dict:
    try:
        prompt = JUDGE_PROMPT.format(
            question=question,
            context=context or "No context provided",
            answer=answer,
            expected=expected,
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        text = response.choices[0].message.content or ""
        return _parse_scores(text)
    except Exception:
        # Fallback scores if API unavailable
        return {"faithfulness": 0.5, "relevance": 0.5, "coherence": 0.5}


def _parse_scores(text: str) -> dict:
    scores = {"faithfulness": 0.5, "relevance": 0.5, "coherence": 0.5}
    for line in text.strip().splitlines():
        line = line.strip().upper()
        if line.startswith("FAITHFULNESS:"):
            scores["faithfulness"] = _normalize(line.split(":")[-1].strip())
        elif line.startswith("RELEVANCE:"):
            scores["relevance"] = _normalize(line.split(":")[-1].strip())
        elif line.startswith("COHERENCE:"):
            scores["coherence"] = _normalize(line.split(":")[-1].strip())
    return scores


def _normalize(val: str) -> float:
    try:
        return (float(val) - 1) / 4  # 1-5 → 0.0-1.0
    except ValueError:
        return 0.5
