from collections.abc import Iterable, Mapping
import math
from typing import Any

from app.services.vector_store import StoredChunk


def citation_is_valid(
    citation: Mapping[str, Any],
    expected_document_id: str,
    retrieved_chunks: Iterable[StoredChunk],
    chunk_lookup: Mapping[str, StoredChunk],
) -> bool:
    """Check citation identity, retrieval membership, document ownership, and locators."""
    chunk_id = citation.get("chunk_id")
    retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    chunk = chunk_lookup.get(str(chunk_id))
    if chunk is None or str(chunk_id) not in retrieved_by_id:
        return False
    if chunk.document_id != expected_document_id:
        return False
    if retrieved_by_id[str(chunk_id)].document_id != expected_document_id:
        return False
    if citation.get("source_locator") != chunk.source_locator:
        return False
    if citation.get("page_number") != chunk.page_number:
        return False
    similarity = citation.get("similarity")
    if not isinstance(similarity, (int, float)) or not math.isclose(
        float(similarity), retrieved_by_id[str(chunk_id)].similarity, rel_tol=1e-6, abs_tol=1e-6
    ):
        return False
    return True
