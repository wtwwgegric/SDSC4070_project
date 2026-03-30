from typing import List, Optional, Any
import os

import chromadb
from career_copilot.config import get_client

# Embedding model to use (cheap + fast)
_EMBED_MODEL = "text-embedding-3-small"


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings using the OpenAI embeddings API."""
    response = get_client().embeddings.create(model=_EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _get_client(persist_directory: Optional[str] = None) -> chromadb.ClientAPI:
    """Return a ChromaDB client (persistent or ephemeral)."""
    if persist_directory:
        return chromadb.PersistentClient(path=persist_directory)
    return chromadb.EphemeralClient()


def create_collection(
    name: str,
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    persist_directory: Optional[str] = None,
) -> Any:
    """Create or replace a Chroma collection from text documents and return it."""
    if not texts:
        raise ValueError("texts must not be empty")

    client = _get_client(persist_directory)

    # Delete existing collection with the same name if present
    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    auto_ids = ids or [f"{name}_{i}" for i in range(len(texts))]
    embeddings = _embed_texts(texts)

    collection.add(
        documents=texts,
        metadatas=metadatas or [{} for _ in texts],
        ids=auto_ids,
        embeddings=embeddings,
    )
    return collection


def query_collection(name: str, query: str, k: int = 4, persist_directory: Optional[str] = None) -> dict:
    """Query a Chroma collection; returns documents, metadatas, and distances."""
    client = _get_client(persist_directory)
    collection = client.get_collection(name)
    q_emb = _embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return results
