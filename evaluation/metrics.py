from collections.abc import Iterable, Sequence


def recall_at_k(retrieved_ids: Sequence[Sequence[str]], relevant_ids: Sequence[Sequence[str]]) -> float | None:
    total_relevant = sum(len(relevant) for relevant in relevant_ids)
    if total_relevant == 0:
        return None
    found = sum(len(set(retrieved) & set(relevant)) for retrieved, relevant in zip(retrieved_ids, relevant_ids, strict=True))
    return found / total_relevant


def mean_reciprocal_rank(retrieved_ids: Sequence[Sequence[str]], relevant_ids: Sequence[Sequence[str]]) -> float | None:
    ranks: list[float] = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids, strict=True):
        rank = next((index for index, chunk_id in enumerate(retrieved, start=1) if chunk_id in set(relevant)), None)
        if rank is not None:
            ranks.append(1.0 / rank)
    return sum(ranks) / len(retrieved_ids) if retrieved_ids else None


def pass_rate(outcomes: Iterable[bool]) -> float | None:
    values = list(outcomes)
    return sum(values) / len(values) if values else None
