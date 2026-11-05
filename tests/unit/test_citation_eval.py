from rie.evaluation.citation_eval import (
    citation_hit_rate,
    evaluate_citations,
    extract_cited_ids,
    false_citation_rate,
)


def test_extract_cited_ids():
    assert extract_cited_ids("See [W1234567890], [mock_0], and (W9876543210).") == [
        "W1234567890",
        "mock_0",
        "W9876543210",
    ]


def test_citation_rates():
    assert citation_hit_rate(["p1", "x"], {"p1"}) == 0.5
    assert false_citation_rate(["p1", "x"], {"p1"}) == 0.5


def test_evaluate_citations():
    result = evaluate_citations("Supported by [mock_0]; contradicted by [mock_1].", ["mock_0"])
    assert result["cited_count"] == 2
    assert result["hallucinated_ids"] == ["mock_1"]
