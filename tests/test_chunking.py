from app.services.chunking import chunk_segments


def test_chunks_keep_page_source_and_configured_overlap():
    sentences = [f"Clause {index} establishes a payment obligation and a clear deadline for the parties." for index in range(1, 11)]
    source_text = " ".join(sentences)

    chunks = chunk_segments("doc-123", [("Page 7", source_text)], chunk_size_chars=220, overlap_chars=70)

    assert len(chunks) > 1
    assert all(chunk.document_id == "doc-123" for chunk in chunks)
    assert all(chunk.page_number == 7 and chunk.source_locator == "Page 7" for chunk in chunks)
    assert all(0 < len(chunk.text) <= 220 for chunk in chunks)
    assert any(first.text[-40:] in second.text for first, second in zip(chunks, chunks[1:]))


def test_short_document_produces_one_chunk_without_page_number():
    chunks = chunk_segments("doc-456", [("Document", "Short agreement text.")], 1200, 180)

    assert len(chunks) == 1
    assert chunks[0].page_number is None
    assert chunks[0].text == "Short agreement text."
