from rie.retrieval.chunker import TextChunker, chunk_paper
from rie.state import Paper


def test_chunker_preserves_text_without_shortening_words():
    chunker = TextChunker(chunk_size=4, overlap=1)
    chunks = chunker.split("one two three four five six seven")

    assert chunks
    assert chunks[0] == "one two three four"
    assert " ".join(chunks).split()[0] == "one"


def test_chunk_paper_uses_abstract_and_metadata():
    paper = Paper(
        paper_id="P1",
        title="Test paper",
        abstract="A short abstract that covers retrieval.",
        year=2025,
        source="openalex",
    )

    chunks = chunk_paper(paper, TextChunker(chunk_size=20))

    assert len(chunks) == 1
    assert chunks[0].paper_id == "P1"
    assert chunks[0].metadata["year"] == 2025
