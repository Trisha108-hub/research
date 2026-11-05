"""Unit tests for the Research Gap Explorer Agent."""

from rie.agents.gap_explorer import GapExplorerAgent
from rie.llm.granite import GraniteClient
from rie.state import DeepEvidence, ResearchState


def test_gap_explorer_returns_empty_when_no_deep_evidence():
    state = ResearchState(research_question="test question")
    GapExplorerAgent(GraniteClient()).run(state)
    assert state.research_gaps == []


def test_gap_explorer_skips_papers_with_empty_limitations():
    state = ResearchState(research_question="test question")
    state.deep_evidence = [
        DeepEvidence(paper_id="p1", title="Paper 1", limitations=""),
    ]
    GapExplorerAgent(GraniteClient()).run(state)
    assert state.research_gaps == []
