from rie.evaluation.ragas_eval import RagasResult, RagasSample, evaluate_with_ragas


def test_empty_samples():
    result = evaluate_with_ragas([])
    assert result.error


def test_result_dict():
    result = RagasResult(sample_count=2)
    assert result.as_dict()["sample_count"] == 2
