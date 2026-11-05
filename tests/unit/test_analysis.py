from rie.agents.analysis import AnalysisAgent
from rie.llm.granite import GraniteClient
from rie.state import DeepEvidence, ResearchState


def test_comparison_matrix_built():
    state = ResearchState(research_question="How does RAG improve factual accuracy?")
    state.deep_evidence = [
        DeepEvidence(
            paper_id="p1",
            title="RAG paper",
            methodology="Retrieval augmented generation",
            datasets=["NQ"],
            metrics=["Accuracy"],
            limitations="Limited to open-domain QA.",
        ),
        DeepEvidence(
            paper_id="p2",
            title="Baseline paper",
            methodology="Fine-tuning",
            datasets=["TriviaQA"],
            metrics=["F1"],
        ),
    ]

    AnalysisAgent(GraniteClient()).run(state)

    assert len(state.comparison_matrix) == 2
    assert state.comparison_matrix[0].paper_id == "p1"
