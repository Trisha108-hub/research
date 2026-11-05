"""Run retrieval and generation evaluation for RIE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rie.config import get_settings
from rie.evaluation.benchmark import load_benchmark, seed_benchmark
from rie.evaluation.citation_eval import evaluate_citations
from rie.evaluation.metrics import evaluate_retrieval
from rie.evaluation.ragas_eval import RagasSample, evaluate_with_ragas
from rie.graph.workflow import ResearchWorkflow
from rie.retrieval.openalex import OpenAlexClient
from rie.retrieval.semantic_scholar import SemanticScholarClient
from rie.state import ResearchState


def run_single(question: str, results_per_source: int = 10, relevant_ids: set[str] | None = None) -> dict:
    settings = get_settings()
    workflow = ResearchWorkflow(
        openalex=OpenAlexClient(),
        semantic_scholar=SemanticScholarClient(settings.semantic_scholar_api_key),
    )
    state = workflow.run(ResearchState(research_question=question), results_per_source)
    retrieved_ids = [paper.paper_id for paper in state.papers]
    citation_result = evaluate_citations(state.report, retrieved_ids)
    ragas_result = evaluate_with_ragas([
        RagasSample(question, state.report[:2000], [chunk.text for chunk in state.chunks[:10]])
    ])
    result = {
        "question": question,
        "papers_retrieved": len(state.papers),
        "chunks_retrieved": len(state.chunks),
        "evidence_items": len(state.evidence),
        "citation_hit_rate": citation_result["citation_hit_rate"],
        "false_citation_rate": citation_result["false_citation_rate"],
        "cited_count": citation_result["cited_count"],
        "hallucinated_ids": citation_result["hallucinated_ids"],
        "ragas": ragas_result.as_dict(),
    }
    if relevant_ids:
        result["retrieval"] = evaluate_retrieval(retrieved_ids, relevant_ids)
        result["has_ground_truth"] = True
    else:
        result["has_ground_truth"] = False
    return result


def run_benchmark(results_per_source: int = 10) -> list[dict]:
    entries = load_benchmark()
    if not entries:
        print("No benchmark entries found. Run with --seed-benchmark first.")
        return []
    return [
        run_single(entry.question, results_per_source, set(entry.relevant_paper_ids))
        for entry in entries
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="RIE Evaluation Runner")
    parser.add_argument("--question")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--seed-benchmark", action="store_true")
    parser.add_argument("--results-per-source", type=int, default=10)
    parser.add_argument("--output", default="data/evaluation_results.json")
    args = parser.parse_args()

    if args.seed_benchmark:
        seed_benchmark()
        print("Created benchmark in data/benchmarks/benchmark.jsonl")
        return
    if args.benchmark:
        results = run_benchmark(args.results_per_source)
    elif args.question:
        results = [run_single(args.question, args.results_per_source)]
    else:
        parser.print_help()
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
