from rie.evaluation.metrics import (
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == 2 / 3


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, 3) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mean_reciprocal_rank():
    result = mean_reciprocal_rank([(["a"], {"a"}), (["x", "b"], {"b"})])
    assert result == 0.75


def test_evaluate_retrieval_keys():
    result = evaluate_retrieval(["a", "b"], {"a"}, 2)
    assert set(result) == {"precision@2", "recall@2", "reciprocal_rank"}
