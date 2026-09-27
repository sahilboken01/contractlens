from collections.abc import Callable

from app.services.embeddings import embed_texts
from app.services.vector_store import StoredChunk, VectorStore


def retrieve_chunks(
    document_id: str,
    query: str,
    top_k: int,
    store: VectorStore,
    embed: Callable[[list[str]], list[list[float]]] = embed_texts,
) -> list[StoredChunk]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty.")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")
    query_vectors = embed([cleaned_query])
    if len(query_vectors) != 1:
        raise RuntimeError("Query embedding model did not return exactly one vector.")
    return store.search(document_id, query_vectors[0], top_k)
