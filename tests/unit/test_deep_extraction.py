from rie.agents.deep_extraction import DeepExtractionAgent
from rie.llm.granite import GraniteClient
from rie.state import Paper


def test_deep_extraction_offline():
    paper = Paper(paper_id="p1", title="RAG paper", abstract="RAG improves factual accuracy.")
    evidence = DeepExtractionAgent(GraniteClient()).extract(paper)

    assert evidence.paper_id == "p1"
    assert evidence.title == paper.title
