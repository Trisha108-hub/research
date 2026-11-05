"""Integration test for the full Phase 2 LangGraph workflow in offline mode."""

from rie.graph.langgraph_workflow import run_langgraph
from rie.state import Paper


class _MockOpenAlex:
    def search(self, query: str, limit: int = 10) -> list[Paper]:
        return [
            Paper(
                paper_id=f"mock_{index}",
                title=f"Mock paper {index}: {query[:35]}",
                abstract=(
                    f"This study investigates {query} using transformer-based retrieval-augmented "
                    "generation. It achieves state-of-the-art results on NQ and TriviaQA benchmarks. "
                    "Limitations include dataset bias and no multilingual evaluation."
                ),
                year=2024,
                source="openalex",
            )
            for index in range(4)
        ]


class _MockSemanticScholar:
    def search(self, query: str, limit: int = 10) -> list[Paper]:
        return []


def test_phase2_pipeline_produces_all_outputs():
    result = run_langgraph(
        question="How does RAG affect factual accuracy in large language models?",
        results_per_source=4,
        top_k=3,
        export_graph=False,
        openalex=_MockOpenAlex(),
        semantic_scholar=_MockSemanticScholar(),
    )

    assert result.subqueries, "Planning must produce sub-queries"
    assert result.papers, "Retrieval must return papers"
    assert result.chunks, "Vector store must return RAG chunks"
    assert result.evidence, "Extraction must produce evidence"
    assert result.deep_evidence, "Deep extraction must produce structured evidence"
    assert result.comparison_matrix, "Analysis must build comparison matrix"
    assert result.verified_claims, "Verification must produce claims"
    assert result.report, "Synthesis and reporting must produce text"
