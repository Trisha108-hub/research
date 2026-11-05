"""Unit tests for the Evidence Verification Agent."""

from rie.agents.extraction import EvidenceExtractionAgent
from rie.agents.verification import VerificationAgent
from rie.llm.granite import GraniteClient
from rie.retrieval.chunker import TextChunker, chunk_paper
from rie.state import Paper, ResearchState


def _make_state() -> ResearchState:
    paper = Paper(
        paper_id="p1",
        title="RAG paper",
        abstract="Retrieval augmented generation improves factual accuracy in language models.",
        year=2024,
        source="openalex",
    )
    state = ResearchState(
        research_question="How does RAG improve factual accuracy?",
        chunks=chunk_paper(paper, TextChunker()),
    )
    EvidenceExtractionAgent(GraniteClient()).run(state)
    return state


def test_verification_produces_one_result_per_evidence_item():
    state = _make_state()
    VerificationAgent(GraniteClient()).run(state)
    assert len(state.verified_claims) == len(state.evidence)


def test_all_verified_claims_have_bool_supported():
    state = _make_state()
    VerificationAgent(GraniteClient()).run(state)
    assert all(isinstance(value.supported, bool) for value in state.verified_claims)


def test_verification_no_crash_when_evidence_is_empty():
    state = ResearchState(research_question="test")
    VerificationAgent(GraniteClient()).run(state)
    assert state.verified_claims == []
