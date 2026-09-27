import json
from uuid import uuid4

import pytest

from app.services.contract_analysis import ContractAnalysisError, analyze_contract
from app.services.rag import NOT_FOUND_ANSWER, answer_question
from app.services.vector_store import StoredChunk
from evaluation.citations import citation_is_valid
from evaluation.dataset import load_dataset
from evaluation.metrics import mean_reciprocal_rank, pass_rate, recall_at_k
from evaluation.offline import MalformedAnalysisProvider, SyntheticVectorStore, deterministic_embedding
from evaluation.runner import MIN_SIMILARITY, run_evaluation


def test_retrieval_evaluation_metrics_are_calculated_from_dataset_cases():
    report = run_evaluation()
    retrieval_cases = [case for case in report["case_results"] if case["suite"] == "retrieval"]
    expected_total = sum(len(case["expected_chunk_ids"]) for case in retrieval_cases)
    found_total = sum(
        len(set(case["expected_chunk_ids"]) & set(case["retrieved_chunk_ids"])) for case in retrieval_cases
    )

    assert report["total_cases"] == len(report["case_results"])
    assert report["retrieval"]["cases"] == len(retrieval_cases)
    assert report["retrieval"]["recall_at_k"] == found_total / expected_total
    assert all(case["document_isolated"] for case in retrieval_cases)


def test_deterministic_rag_answers_are_supported_and_missing_questions_abstain():
    report = run_evaluation()

    assert report["rag"]["answer_and_citation_pass_rate"] == 1.0
    assert report["rag"]["unsupported_answer_count"] == 0
    assert report["hallucination_resistance"]["unsupported_answers"] == 0


def test_citation_validator_rejects_unknown_foreign_unretrieved_and_bad_locator():
    local = StoredChunk("local-1", "text", "Section A", 1, 0.8, {}, "doc-a")
    foreign = StoredChunk("foreign-1", "other text", "Section B", 2, 0.7, {}, "doc-b")
    valid = {"chunk_id": "local-1", "source_locator": "Section A", "page_number": 1, "similarity": 0.8}
    lookup = {"local-1": local, "foreign-1": foreign}

    assert citation_is_valid(valid, "doc-a", [local], lookup)
    assert not citation_is_valid({**valid, "chunk_id": "unknown"}, "doc-a", [local], lookup)
    assert not citation_is_valid(
        {**valid, "chunk_id": "foreign-1", "source_locator": "Section B", "page_number": 2, "similarity": 0.7},
        "doc-a",
        [foreign],
        lookup,
    )
    assert not citation_is_valid(valid, "doc-a", [], lookup)
    assert not citation_is_valid({**valid, "source_locator": "Wrong section"}, "doc-a", [local], lookup)


def test_rag_suppresses_chunks_returned_for_another_document():
    requested_id = "doc-a"
    foreign = StoredChunk("foreign-1", "governing law is Ontario", "Section B", 2, 0.95, {}, "doc-b")

    class ForeignStore:
        def search(self, document_id, query_embedding, top_k):
            return [foreign]

    class MustNotCallProvider:
        def generate(self, *_):
            pytest.fail("foreign context must not reach the LLM")

    result = answer_question(
        requested_id,
        "What governing law applies?",
        3,
        MIN_SIMILARITY,
        ForeignStore(),
        MustNotCallProvider(),
        deterministic_embedding,
    )

    assert result.answer == NOT_FOUND_ANSWER
    assert result.sources == []


def test_analysis_suppresses_foreign_chunks_and_citations():
    dataset = load_dataset()
    foreign_contract = dataset["contracts"][1]
    foreign_store = SyntheticVectorStore([foreign_contract])
    requested_id = dataset["contracts"][0]["document_id"]

    class ForeignReturningStore:
        def search(self, document_id, query_embedding, top_k):
            return [
                StoredChunk("foreign", "Governing law: Ontario", "Section X", 8, 0.99, {}, foreign_contract["document_id"])
            ]

    class MustNotCallProvider:
        def generate(self, *_):
            pytest.fail("foreign context must not reach analysis provider")

    result = analyze_contract(
        requested_id,
        ForeignReturningStore(),
        MustNotCallProvider(),
        top_k=2,
        minimum_similarity=MIN_SIMILARITY,
        embed=deterministic_embedding,
    )

    assert result.status == "insufficient_context"
    assert result.governing_law is None
    assert foreign_store.chunks_by_id  # Confirms the fixture contains a different document.


def test_malformed_analysis_output_is_rejected():
    dataset = load_dataset()
    contract = dataset["contracts"][0]
    store = SyntheticVectorStore([contract])

    with pytest.raises(ContractAnalysisError):
        analyze_contract(
            contract["document_id"],
            store,
            MalformedAnalysisProvider(),
            top_k=3,
            minimum_similarity=MIN_SIMILARITY,
            embed=deterministic_embedding,
        )


def test_analysis_extraction_missing_information_and_hallucination_metrics():
    report = run_evaluation()

    assert report["analysis"]["extraction_pass_rate"] == 1.0
    assert report["analysis"]["missing_information_pass_rate"] == 1.0
    assert report["hallucination_resistance"]["invalid_analysis_citation_suppressed"] is True
    assert report["citations"]["validity_rate"] == 1.0


def test_metric_helpers_cover_recall_rank_and_empty_sets():
    retrieved = [["x", "a"], ["b", "c"]]
    relevant = [["a", "b"], ["missing"]]

    assert recall_at_k(retrieved, relevant) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.25)
    assert pass_rate([True, False, True]) == pytest.approx(2 / 3)
    assert recall_at_k([[]], [[]]) is None
    assert mean_reciprocal_rank([], []) is None
    assert pass_rate([]) is None
