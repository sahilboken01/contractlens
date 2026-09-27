import re
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class DocumentChunk:
    document_id: str
    chunk_id: str
    text: str
    source_locator: str
    page_number: int | None
    metadata: dict[str, object]
    embedding: list[float] = field(default_factory=list)


def chunk_segments(
    document_id: str,
    segments: list[tuple[str, str]],
    chunk_size_chars: int = 1200,
    overlap_chars: int = 180,
) -> list[DocumentChunk]:
    """Chunk each source segment near sentence boundaries, with character overlap."""
    if chunk_size_chars < 200:
        raise ValueError("chunk_size_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars * 2 >= chunk_size_chars:
        raise ValueError("overlap_chars must be non-negative and less than half chunk_size_chars")

    chunks: list[DocumentChunk] = []
    for locator, raw_text in segments:
        normalized = re.sub(r"[ \t]+", " ", raw_text).strip()
        if not normalized:
            continue
        page_match = re.fullmatch(r"Page\s+(\d+)", locator, flags=re.IGNORECASE)
        start = 0
        while start < len(normalized):
            proposed_end = min(start + chunk_size_chars, len(normalized))
            end = proposed_end
            if proposed_end < len(normalized):
                # Prefer the last sentence or paragraph boundary near the target.
                min_boundary = start + int(chunk_size_chars * 0.60)
                sentence_ends = [
                    match.end()
                    for match in re.finditer(r"(?:[.!?][\"')\]]*\s+|\n+)", normalized[start:proposed_end])
                    if start + match.end() >= min_boundary
                ]
                if sentence_ends:
                    end = start + sentence_ends[-1]
                else:
                    whitespace = normalized.rfind(" ", min_boundary, proposed_end)
                    newline = normalized.rfind("\n", min_boundary, proposed_end)
                    boundary = max(whitespace, newline)
                    if boundary > start:
                        end = boundary

            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_id=str(uuid4()),
                        text=chunk_text,
                        source_locator=locator,
                        page_number=int(page_match.group(1)) if page_match else None,
                        metadata={"source_locator": locator, "chunk_index": len(chunks)},
                    )
                )
            if end >= len(normalized):
                break

            next_start = max(start + 1, end - overlap_chars)
            # Keep the next chunk's start on a word boundary; this slightly increases
            # overlap instead of cutting a word in half.
            word_boundary = normalized.rfind(" ", start + 1, next_start + 1)
            if word_boundary > start:
                next_start = word_boundary + 1
            start = next_start

    return chunks
