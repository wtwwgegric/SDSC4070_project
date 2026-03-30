from typing import List, Optional, Any
import os

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api import API
    from langchain.embeddings.openai import OpenAIEmbeddings
except Exception as e:
    raise ImportError("Missing optional dependencies for RAG. Install chromadb and langchain.") from e


def _get_client(persist_directory: Optional[str] = None) -> "API":
    """Create a chromadb client. If persist_directory is provided, data will be persisted there."""
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
    return chromadb.Client(settings=settings)


def create_collection(
    name: str,
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    persist_directory: Optional[str] = None,
) -> Any:
    """Create or replace a Chroma collection from text documents.

    - `texts` is a list of document strings.
    - `metadatas` is an optional list of dicts.
    - `ids` optional list of ids.
    - `persist_directory` if set will persist the DB to disk.
    Returns the created collection object.
    """
    client = _get_client(persist_directory)
    try:
        collection = client.get_collection(name)
        # remove existing if present to replace
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name)

    embedder = OpenAIEmbeddings()
    embeddings = embedder.embed_documents(texts)

    collection.add(
        documents=texts,
        metadatas=metadatas or [{} for _ in texts],
        ids=ids,
        embeddings=embeddings,
    )
    if persist_directory:
        client.persist()
    return collection


def query_collection(name: str, query: str, k: int = 4, persist_directory: Optional[str] = None) -> dict:
    """Query a Chroma collection and return documents, metadatas and distances."""
    client = _get_client(persist_directory)
    collection = client.get_collection(name)
    embedder = OpenAIEmbeddings()
    q_emb = embedder.embed_query(query)
    results = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])  # type: ignore
    return results
