"""Citation accuracy evaluation."""

from __future__ import annotations

import re


def extract_cited_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in re.findall(r"\[([A-Za-z]\w{3,})\]|\(([A-Za-z]\w{3,})\)", text):
        ids.append(next(item for item in match if item))
    ids.extend(re.findall(r"\bmock_\d+\b", text))
    return list(dict.fromkeys(ids))


def citation_hit_rate(cited_ids: list[str], retrieved_ids: set[str]) -> float:
    if not cited_ids:
        return 1.0
    return sum(cited_id in retrieved_ids for cited_id in cited_ids) / len(cited_ids)


def false_citation_rate(cited_ids: list[str], retrieved_ids: set[str]) -> float:
    return 1.0 - citation_hit_rate(cited_ids, retrieved_ids)


def evaluate_citations(
    generated_text: str,
    retrieved_paper_ids: list[str],
) -> dict[str, float | int | list[str]]:
    cited = extract_cited_ids(generated_text)
    retrieved = set(retrieved_paper_ids)
    return {
        "cited_count": len(cited),
        "citation_hit_rate": citation_hit_rate(cited, retrieved),
        "false_citation_rate": false_citation_rate(cited, retrieved),
        "hallucinated_ids": [cited_id for cited_id in cited if cited_id not in retrieved],
    }
