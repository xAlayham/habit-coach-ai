import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_FILE = PROJECT_ROOT / "knowledge" / "habit_research.json"
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = "habit_research"

CHUNK_WORDS = 70
CHUNK_OVERLAP_WORDS = 20
DEFAULT_TOP_K = 3
OVERFETCH = 4

_collection = None


class EmptyKnowledgeBase(RuntimeError):
    pass


def chunk_passage(text, chunk_words=CHUNK_WORDS, overlap_words=CHUNK_OVERLAP_WORDS):
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = chunk_words - overlap_words
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def load_records(path=KNOWLEDGE_FILE):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_chunks(records):
    chunks = []
    for record in records:
        for index, text in enumerate(chunk_passage(record["text"])):
            chunks.append({
                "id": f"{record['id']}::{index}",
                "document": text,
                "metadata": {
                    "doc_id": record["id"],
                    "topic": record["topic"],
                    "source": record["source"],
                    "url": record["url"],
                    "kind": record["kind"],
                    "chunk_index": index,
                },
            })
    return chunks


def get_collection():
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )
    return _collection


def search_research(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        raise EmptyKnowledgeBase(
            "The research knowledge base is empty. Run 'python ingest.py' to build it."
        )

    top_k = max(1, min(int(top_k), 5))
    result = collection.query(
        query_texts=[query],
        n_results=min(top_k * OVERFETCH, collection.count()),
    )

    hits = []
    seen_sources = set()
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if metadata["doc_id"] in seen_sources:
            continue
        seen_sources.add(metadata["doc_id"])
        hits.append({
            "excerpt": document,
            "topic": metadata["topic"],
            "source": metadata["source"],
            "url": metadata["url"],
            "kind": metadata["kind"],
            "similarity": round(1.0 - distance, 3),
        })
        if len(hits) == top_k:
            break
    return hits
