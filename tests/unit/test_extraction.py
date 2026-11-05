from rie.agents.extraction import EvidenceExtractionAgent
from rie.llm.granite import GraniteClient
from rie.retrieval.chunker import TextChunker, chunk_paper
from rie.state import Paper, ResearchState


def test_extraction_produces_evidence_items():
    paper = Paper(
        paper_id="p1",
        title="RAG paper",
        abstract="Retrieval augmented generation improves factual accuracy.",
        year=2024,
        source="openalex",
    )
    state = ResearchState(
        research_question="How does RAG improve factual accuracy?",
        chunks=chunk_paper(paper, TextChunker()),
    )

    result = EvidenceExtractionAgent(GraniteClient()).run(state)

    assert result.evidence
    assert all(item.claim for item in result.evidence)
    assert all(0 <= item.confidence <= 1 for item in result.evidence)
    assert all(item.stance in {"supports", "contradicts", "neutral"} for item in result.evidence)
