import axios, { AxiosInstance } from "axios";

const BASE_URL = process.env.EVALFORGE_API_URL || "http://localhost:8000";

export const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,
  headers: { "Content-Type": "application/json" },
});

export interface EvalRunRequest {
  dataset_id: string;
  rag_endpoint: string;
  model: string;
  top_k: number;
}

export interface EvalRunResponse {
  run_id: string;
  status: string;
  metrics: {
    precision_at_k: number;
    recall_at_k: number;
    faithfulness: number;
    relevance: number;
    coherence: number;
    overall_score: number;
  };
  total_queries: number;
  duration_seconds: number;
}

export interface DatasetAddRequest {
  dataset_id: string;
  qa_pairs: Array<{ question: string; expected_answer: string; context?: string }>;
}

export async function startEvalRun(req: EvalRunRequest): Promise<EvalRunResponse> {
  const { data } = await api.post<EvalRunResponse>("/eval/run", req);
  return data;
}

export async function getReport(runId: string): Promise<EvalRunResponse> {
  const { data } = await api.get<EvalRunResponse>(`/eval/report/${runId}`);
  return data;
}

export async function addDataset(req: DatasetAddRequest): Promise<{ message: string; count: number }> {
  const { data } = await api.post("/dataset/add", req);
  return data;
}

export async function listDatasets(): Promise<{ datasets: string[] }> {
  const { data } = await api.get("/dataset/list");
  return data;
}

export async function healthCheck(): Promise<boolean> {
  try {
    await api.get("/health");
    return true;
  } catch {
    return false;
  }
}
