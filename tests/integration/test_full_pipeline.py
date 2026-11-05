from rie.graph.workflow import ResearchWorkflow
from rie.state import Paper, ResearchState


class MockOpenAlex:
    def search(self, query: str, limit: int = 10):
        return [
            Paper(
                paper_id=f"mock_{index}",
                title=f"Mock paper {index}: {query[:30]}",
                abstract=f"This paper studies {query} using machine learning and retrieval methods.",
                year=2024,
                source="openalex",
            )
            for index in range(3)
        ]


class MockSemanticScholar:
    def search(self, query: str, limit: int = 10):
        return []


def test_full_pipeline_offline():
    workflow = ResearchWorkflow(
        openalex=MockOpenAlex(),
        semantic_scholar=MockSemanticScholar(),
    )
    state = ResearchState(research_question="How does RAG improve factual accuracy?")
    result = workflow.run(state, results_per_source=3)

    assert result.subqueries
    assert result.papers
    assert result.chunks
    assert result.evidence
    assert result.report
