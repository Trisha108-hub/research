"""Evaluation utilities for research quality."""

from .benchmark import BenchmarkEntry, load_benchmark, save_benchmark, seed_benchmark
from .citation_eval import evaluate_citations, citation_hit_rate, false_citation_rate, extract_cited_ids
from .metrics import evaluate_retrieval, mean_reciprocal_rank, precision_at_k, recall_at_k, reciprocal_rank
from .ragas_eval import RagasResult, RagasSample, evaluate_with_ragas

__all__ = [
    "BenchmarkEntry", "load_benchmark", "save_benchmark", "seed_benchmark",
    "evaluate_citations", "citation_hit_rate", "false_citation_rate", "extract_cited_ids",
    "evaluate_retrieval", "mean_reciprocal_rank", "precision_at_k", "recall_at_k", "reciprocal_rank",
    "RagasResult", "RagasSample", "evaluate_with_ragas",
]
