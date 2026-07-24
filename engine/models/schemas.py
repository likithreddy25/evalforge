from pydantic import BaseModel, Field
from typing import Optional
import uuid


class QAPair(BaseModel):
    question: str
    expected_answer: str
    context: Optional[str] = None


class DatasetAddRequest(BaseModel):
    dataset_id: str
    qa_pairs: list[QAPair]


class EvalRunRequest(BaseModel):
    dataset_id: str
    rag_endpoint: str
    model: str = "gpt-4o-mini"
    top_k: int = Field(default=3, ge=1, le=10)


class EvalMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    faithfulness: float
    relevance: float
    coherence: float
    overall_score: float


class EvalRunResponse(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "completed"
    dataset_id: str
    metrics: EvalMetrics
    total_queries: int
    duration_seconds: float
