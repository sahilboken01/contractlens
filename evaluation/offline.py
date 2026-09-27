import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

from app.services.llm import LLMProvider
from app.services.vector_store import StoredChunk

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "this", "to", "what", "when",
    "who", "which", "with", "will", "would", "agreement", "contract", "document",
}
VECTOR_DIMENSIONS = 512


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOP_WORDS]


def deterministic_embedding(texts: list[str]) -> list[list[float]]:
    """Stable signed feature hashing for offline tests; not the production BGE model."""
    vectors = []
    for text in texts:
        vector = [0.0] * VECTOR_DIMENSIONS
        for token, count in Counter(_tokens(text)).items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % VECTOR_DIMENSIONS
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * count
        norm = math.sqrt(sum(value * value for value in vector))
        vectors.append([value / norm for value in vector] if norm else vector)
    return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class SyntheticVectorStore:
    """In-memory vector-store adapter that enforces the same document filter."""

    def __init__(self, contracts: list[dict[str, Any]]):
        self.chunks_by_document: dict[str, list[StoredChunk]] = {}
        self.chunks_by_id: dict[str, StoredChunk] = {}
        self.search_history: list[dict[str, Any]] = []
        for contract in contracts:
            document_id = contract["document_id"]
            self.chunks_by_document[document_id] = []
            for row in contract["chunks"]:
                chunk = StoredChunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    source_locator=row["source_locator"],
                    page_number=row.get("page_number"),
                    similarity=0.0,
                    metadata={},
                    document_id=document_id,
                )
                self.chunks_by_document[document_id].append(chunk)
                self.chunks_by_id[chunk.chunk_id] = chunk

    def search(self, document_id: str, query_embedding: list[float], top_k: int) -> list[StoredChunk]:
        candidates = [
            StoredChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_locator=chunk.source_locator,
                page_number=chunk.page_number,
                similarity=cosine_similarity(query_embedding, deterministic_embedding([chunk.text])[0]),
                metadata=chunk.metadata,
                document_id=chunk.document_id,
            )
            for chunk in self.chunks_by_document.get(document_id, [])
        ]
        candidates.sort(key=lambda chunk: chunk.similarity, reverse=True)
        results = candidates[:top_k]
        self.search_history.append({"document_id": document_id, "results": results})
        return results


def _context_from_prompt(user_prompt: str, marker: str) -> list[dict[str, Any]]:
    encoded = user_prompt.split(marker, 1)[1]
    return json.loads(encoded)


class EchoEvidenceProvider(LLMProvider):
    """Mock RAG provider that quotes the best retrieved excerpt and its source label."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        context = _context_from_prompt(user_prompt, "Contract context: ")
        selected = context[0]
        return json.dumps({"answer": selected["text"], "citations": [selected["source"]]})


class ExpectedAnalysisProvider(LLMProvider):
    """Mock structured provider seeded only from explicitly defined gold evaluation facts."""

    def __init__(self, expected_analysis: dict[str, list[dict[str, str]]], adversarial: bool = False):
        self.expected_analysis = expected_analysis
        self.adversarial = adversarial
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        context = _context_from_prompt(user_prompt, "Contract excerpts (JSON): ")
        labels_by_chunk = {source["chunk_id"]: source["source_label"] for source in context}
        output: dict[str, Any] = {}
        for field, facts in self.expected_analysis.items():
            mapped = []
            for fact in facts:
                source_label = labels_by_chunk.get(fact["chunk_id"])
                mapped.append(
                    {
                        "value": fact["value"],
                        "basis": "explicit",
                        "source_labels": [source_label] if source_label else [],
                    }
                )
            if field in {"contract_type", "effective_date", "expiration_date", "governing_law"}:
                output[field] = mapped[0] if mapped else None
            else:
                output[field] = mapped
        if self.adversarial:
            output["governing_law"] = {
                "value": "California law applies.",
                "basis": "explicit",
                "source_labels": ["C999-not-retrieved"],
            }
        return json.dumps(output)


class MalformedAnalysisProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "{malformed JSON"
