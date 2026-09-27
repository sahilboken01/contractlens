from typing import Any

from app.schemas.analysis import ContractAnalysisResponse
from app.services.contract_analysis import ContractAnalysisError, analyze_contract
from app.services.rag import NOT_FOUND_ANSWER, answer_question
from app.services.retrieval import retrieve_chunks
from evaluation.citations import citation_is_valid
from evaluation.dataset import load_dataset
from evaluation.metrics import mean_reciprocal_rank, pass_rate, recall_at_k
from evaluation.offline import (
    EchoEvidenceProvider,
    ExpectedAnalysisProvider,
    MalformedAnalysisProvider,
    SyntheticVectorStore,
    deterministic_embedding,
)

# Offline lexical feature vectors have a different cosine distribution from BGE.
MIN_SIMILARITY = 0.08


def _record(outcomes: list[dict[str, Any]], suite: str, case_id: str, passed: bool, **details: Any) -> None:
    outcomes.append({"suite": suite, "case_id": case_id, "passed": bool(passed), **details})


def _analysis_findings(response: ContractAnalysisResponse, field_name: str) -> list[Any]:
    value = getattr(response, field_name)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def run_evaluation() -> dict[str, Any]:
    dataset = load_dataset()
    contracts = dataset["contracts"]
    contract_by_id = {contract["document_id"]: contract for contract in contracts}
    store = SyntheticVectorStore(contracts)
    chunk_lookup = store.chunks_by_id
    outcomes: list[dict[str, Any]] = []

    retrieval_ids: list[list[str]] = []
    expected_ids: list[list[str]] = []
    ranks: list[int | None] = []
    isolation_outcomes: list[bool] = []
    for case in dataset["retrieval_cases"]:
        results = retrieve_chunks(
            case["document_id"], case["question"], case["top_k"], store, deterministic_embedding
        )
        ids = [result.chunk_id for result in results]
        expected = case["expected_chunk_ids"]
        rank = next((index for index, chunk_id in enumerate(ids, start=1) if chunk_id in expected), None)
        isolated = all(result.document_id == case["document_id"] for result in results)
        hit = set(expected).issubset(ids)
        passed = hit and isolated
        retrieval_ids.append(ids)
        expected_ids.append(expected)
        ranks.append(rank)
        isolation_outcomes.append(isolated)
        _record(
            outcomes,
            "retrieval",
            case["case_id"],
            passed,
            rank=rank,
            expected_chunk_ids=expected,
            retrieved_chunk_ids=ids,
            document_isolated=isolated,
        )

    citation_count = 0
    valid_citation_count = 0
    rag_outcomes: list[bool] = []
    unsupported_answer_count = 0
    for case in dataset["rag_cases"]:
        history_start = len(store.search_history)
        result = answer_question(
            case["document_id"],
            case["question"],
            top_k=4,
            minimum_similarity=MIN_SIMILARITY,
            store=store,
            provider=EchoEvidenceProvider(),
            embed=deterministic_embedding,
        )
        actual_retrievals = [entry for entry in store.search_history[history_start:]]
        retrieved = actual_retrievals[-1]["results"] if actual_retrievals else []
        retrieved = [
            chunk
            for chunk in retrieved
            if chunk.similarity >= MIN_SIMILARITY and chunk.document_id == case["document_id"]
        ]
        retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved}

        if case.get("not_found"):
            answer_ok = result.answer == NOT_FOUND_ANSWER and not result.sources
            if not answer_ok:
                unsupported_answer_count += 1
            citation_ok = not result.sources
        else:
            expected_chunk_id = case["expected_chunk_id"]
            expected_snippet = case["expected_answer_contains"].lower()
            answer_ok = expected_snippet in result.answer.lower()
            citation_ok = any(
                source.chunk_id == expected_chunk_id
                and citation_is_valid(
                    {
                        "chunk_id": source.chunk_id,
                        "source_locator": source.source_locator,
                        "page_number": source.page_number,
                        "similarity": source.similarity,
                    },
                    case["document_id"],
                    retrieved,
                    chunk_lookup,
                )
                and expected_snippet in source.text.lower()
                for source in result.sources
            )

        for source in result.sources:
            citation_count += 1
            citation_valid = citation_is_valid(
                {
                    "chunk_id": source.chunk_id,
                    "source_locator": source.source_locator,
                    "page_number": source.page_number,
                    "similarity": source.similarity,
                },
                case["document_id"],
                retrieved,
                chunk_lookup,
            )
            valid_citation_count += int(citation_valid)
        passed = answer_ok and citation_ok
        rag_outcomes.append(passed)
        _record(
            outcomes,
            "rag",
            case["case_id"],
            passed,
            answer_supported=answer_ok,
            citations_valid_and_supporting=citation_ok,
            cited_chunk_ids=[source.chunk_id for source in result.sources],
        )

    analysis_fact_outcomes: list[bool] = []
    missing_field_outcomes: list[bool] = []
    analysis_citation_count = 0
    valid_analysis_citation_count = 0
    for contract in contracts:
        document_id = contract["document_id"]
        provider = ExpectedAnalysisProvider(contract["expected_analysis"])
        history_start = len(store.search_history)
        analysis = analyze_contract(
            document_id=document_id,
            store=store,
            provider=provider,
            top_k=3,
            minimum_similarity=MIN_SIMILARITY,
            embed=deterministic_embedding,
        )
        retrieval_union: dict[str, Any] = {}
        for entry in store.search_history[history_start:]:
            for chunk in entry["results"]:
                if chunk.document_id == document_id and chunk.similarity >= MIN_SIMILARITY:
                    previous = retrieval_union.get(chunk.chunk_id)
                    if previous is None or chunk.similarity > previous.similarity:
                        retrieval_union[chunk.chunk_id] = chunk
        retrieved_chunks = list(retrieval_union.values())
        for field_name in ContractAnalysisResponse.model_fields:
            if field_name in {"document_id", "status"}:
                continue
            for finding in _analysis_findings(analysis, field_name):
                for citation in finding.sources:
                    analysis_citation_count += 1
                    valid_analysis_citation_count += int(
                        citation_is_valid(
                            citation.model_dump(), document_id, retrieved_chunks, chunk_lookup
                        )
                    )
        expected_field_names = set(contract["expected_analysis"])
        for field_name, expected_findings in contract["expected_analysis"].items():
            for index, expected_finding in enumerate(expected_findings):
                matches = [
                    finding
                    for finding in _analysis_findings(analysis, field_name)
                    if finding.value == expected_finding["value"]
                ]
                cited = any(
                    expected_finding["chunk_id"] == citation.chunk_id
                    and citation_is_valid(
                        citation.model_dump(), document_id, retrieved_chunks, chunk_lookup
                    )
                    for finding in matches
                    for citation in finding.sources
                )
                passed = bool(matches) and cited and all(
                    finding.basis in {"explicit", "inference"} and finding.sources for finding in matches
                )
                analysis_fact_outcomes.append(passed)
                _record(
                    outcomes,
                    "analysis_extraction",
                    f"{document_id}:{field_name}:{index}",
                    passed,
                    expected_value=expected_finding["value"],
                    citation_chunk_id=expected_finding["chunk_id"],
                )

        for field_name in contract["expected_missing_fields"]:
            missing = not _analysis_findings(analysis, field_name)
            missing_field_outcomes.append(missing)
            _record(outcomes, "missing_information", f"{document_id}:{field_name}", missing)

    # Negative control: an adversarial mock fabricates governing law with a label that was
    # not in retrieval. Production analysis must drop this unsupported finding.
    cedar = contract_by_id["synthetic-cedar-002"]
    adversarial_provider = ExpectedAnalysisProvider(cedar["expected_analysis"], adversarial=True)
    adversarial_result = analyze_contract(
        document_id=cedar["document_id"],
        store=store,
        provider=adversarial_provider,
        top_k=3,
        minimum_similarity=MIN_SIMILARITY,
        embed=deterministic_embedding,
    )
    hallucination_suppressed = adversarial_provider.calls > 0 and adversarial_result.governing_law is None
    _record(outcomes, "hallucination_resistance", "invalid-analysis-citation", hallucination_suppressed)

    malformed_raised = False
    try:
        analyze_contract(
            document_id=cedar["document_id"],
            store=store,
            provider=MalformedAnalysisProvider(),
            top_k=3,
            minimum_similarity=MIN_SIMILARITY,
            embed=deterministic_embedding,
        )
    except ContractAnalysisError:
        malformed_raised = True
    _record(outcomes, "malformed_analysis", "malformed-json-rejected", malformed_raised)

    total_cases = len(outcomes)
    passed_cases = sum(1 for outcome in outcomes if outcome["passed"])
    total_citations = citation_count + analysis_citation_count
    valid_citations = valid_citation_count + valid_analysis_citation_count
    return {
        "dataset": "synthetic_contracts.json",
        "retrieval_method": "existing retrieve_chunks service with deterministic signed lexical feature vectors and in-memory document-filtered store; this is a repeatable proxy, not the production BGE/pgvector backend",
        "model_evaluation_note": "RAG and analysis providers are deterministic mocks seeded from retrieved evidence or explicit gold labels; metrics validate retrieval, schemas, grounding, mapping, and abstention behavior, not external LLM factual accuracy",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "retrieval": {
            "cases": len(retrieval_ids),
            "recall_at_k": recall_at_k(retrieval_ids, expected_ids),
            "mean_reciprocal_rank": mean_reciprocal_rank(retrieval_ids, expected_ids),
            "ranks": ranks,
            "document_isolation_pass_rate": pass_rate(isolation_outcomes),
        },
        "rag": {
            "cases": len(rag_outcomes),
            "answer_and_citation_pass_rate": pass_rate(rag_outcomes),
            "unsupported_answer_count": unsupported_answer_count,
        },
        "citations": {
            "total_returned": total_citations,
            "valid_returned": valid_citations,
            "validity_rate": valid_citations / total_citations if total_citations else None,
            "rag_validity_rate": valid_citation_count / citation_count if citation_count else None,
            "analysis_validity_rate": valid_analysis_citation_count / analysis_citation_count if analysis_citation_count else None,
        },
        "analysis": {
            "expected_findings": len(analysis_fact_outcomes),
            "findings_with_exact_value_and_valid_citation": sum(analysis_fact_outcomes),
            "extraction_pass_rate": pass_rate(analysis_fact_outcomes),
            "missing_fields_checked": len(missing_field_outcomes),
            "missing_fields_remained_empty": sum(missing_field_outcomes),
            "missing_information_pass_rate": pass_rate(missing_field_outcomes),
        },
        "hallucination_resistance": {
            "unsupported_answers": unsupported_answer_count,
            "invalid_analysis_citation_suppressed": hallucination_suppressed,
        },
        "case_results": outcomes,
    }
