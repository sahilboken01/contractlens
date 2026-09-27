from app.services.retrieval import retrieve_chunks
from app.services.vector_store import StoredChunk


class FakeStore:
    def __init__(self):
        self.args = None

    def search(self, document_id, query_embedding, top_k):
        self.args = (document_id, query_embedding, top_k)
        candidates = [
            StoredChunk("c1", "Payment net 30", "Page 2", 2, 0.91, {}),
            StoredChunk("c2", "Termination for cause", "Page 8", 8, 0.72, {}),
        ]
        return sorted(candidates, key=lambda item: item.similarity, reverse=True)[:top_k]


def test_query_is_embedded_and_search_is_document_scoped():
    store = FakeStore()
    embedded_inputs = []

    def fake_embed(texts):
        embedded_inputs.extend(texts)
        return [[0.2, 0.8, 0.1]]

    results = retrieve_chunks("doc-id", "  payment deadline?  ", 1, store, fake_embed)

    assert embedded_inputs == ["payment deadline?"]
    assert store.args == ("doc-id", [0.2, 0.8, 0.1], 1)
    assert [item.chunk_id for item in results] == ["c1"]
    assert results[0].page_number == 2


def test_empty_query_is_rejected_before_embedding():
    store = FakeStore()
    try:
        retrieve_chunks("doc-id", "  ", 3, store, lambda _: [])
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("Expected empty query validation")
