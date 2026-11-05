from rie.evaluation.benchmark import BenchmarkEntry, load_benchmark, save_benchmark, seed_benchmark


def test_seed_and_round_trip(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    entries = seed_benchmark(path)
    assert len(entries) == 5
    assert load_benchmark(path) == entries


def test_save_and_load(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    save_benchmark([BenchmarkEntry(question="Test?", relevant_paper_ids=["p1"])], path)
    assert load_benchmark(path)[0].relevant_paper_ids == ["p1"]


def test_missing_benchmark(tmp_path):
    assert load_benchmark(tmp_path / "missing.jsonl") == []
