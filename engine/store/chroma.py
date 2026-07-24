import chromadb
from chromadb.config import Settings
from typing import Optional
import os

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

_client: Optional[chromadb.ClientAPI] = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_or_create_collection(dataset_id: str) -> chromadb.Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=f"evalforge_{dataset_id}",
        metadata={"hnsw:space": "cosine"},
    )


def add_qa_pairs(dataset_id: str, qa_pairs: list[dict]) -> int:
    collection = get_or_create_collection(dataset_id)
    ids = [f"{dataset_id}_{i}" for i in range(len(qa_pairs))]
    documents = [p["question"] for p in qa_pairs]
    metadatas = [
        {"expected_answer": p["expected_answer"], "context": p.get("context", "")}
        for p in qa_pairs
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(qa_pairs)


def get_all_qa_pairs(dataset_id: str) -> list[dict]:
    collection = get_or_create_collection(dataset_id)
    results = collection.get(include=["documents", "metadatas"])
    pairs = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        pairs.append({
            "question": doc,
            "expected_answer": meta["expected_answer"],
            "context": meta.get("context", ""),
        })
    return pairs


def list_datasets() -> list[str]:
    client = get_client()
    collections = client.list_collections()
    return [c.name.replace("evalforge_", "") for c in collections if c.name.startswith("evalforge_")]
