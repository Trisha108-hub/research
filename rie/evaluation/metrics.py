"""Retrieval quality metrics."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not retrieved or k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(item in relevant for item in top) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = retrieved[:k]
    return len(set(top).intersection(relevant)) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(queries: list[tuple[list[str], set[str]]]) -> float:
    if not queries:
        return 0.0
    return sum(reciprocal_rank(retrieved, relevant) for retrieved, relevant in queries) / len(queries)


def evaluate_retrieval(retrieved: list[str], relevant: set[str], k: int = 5) -> dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        "reciprocal_rank": reciprocal_rank(retrieved, relevant),
    }
