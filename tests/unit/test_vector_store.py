import pytest

from rie.retrieval.chunker import TextChunker, chunk_paper
from rie.retrieval.vector_store import FaissVectorStore
from rie.state import Paper


@pytest.fixture
def sample_papers() -> list[Paper]:
    return [
        Paper(
            paper_id="p1",
            title="RAG improves factual accuracy",
            abstract="Retrieval-augmented generation reduces hallucination in language models.",
            year=2024,
            source="openalex",
        ),
        Paper(
            paper_id="p2",
            title="Transformer survey",
            abstract="A survey of attention mechanisms and transformer architectures.",
            year=2023,
            source="semantic_scholar",
        ),
    ]


def test_build_and_search(sample_papers):
    chunks = [chunk for paper in sample_papers for chunk in chunk_paper(paper, TextChunker())]
    store = FaissVectorStore("all-MiniLM-L6-v2")
    store.build(chunks)
    results = store.search("hallucination retrieval", k=2)

    assert results
    assert results[0][1] > 0


def test_empty_build_returns_no_results():
    store = FaissVectorStore("all-MiniLM-L6-v2")
    store.build([])
    assert store.search("anything") == []


def test_save_and_load(tmp_path, sample_papers):
    chunks = [chunk for paper in sample_papers for chunk in chunk_paper(paper, TextChunker())]
    store = FaissVectorStore("all-MiniLM-L6-v2")
    store.build(chunks)
    index_path = tmp_path / "test.index"
    chunks_path = tmp_path / "test.jsonl"
    store.save(index_path, chunks_path)

    loaded = FaissVectorStore("all-MiniLM-L6-v2")
    loaded.load(index_path, chunks_path)
    assert loaded.search("language model", k=1)
