import re

from rie.state import Chunk, Paper


class TextChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        overlap = min(overlap, max(0, chunk_size // 4))
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        words = re.findall(r"\S+", text.strip())
        if not words:
            return []
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start:start + self.chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_size >= len(words):
                break
        return chunks


def chunk_paper(paper: Paper, chunker: TextChunker | None = None) -> list[Chunk]:
    chunker = chunker or TextChunker()
    text = paper.abstract.strip() or paper.title
    return [
        Chunk(
            chunk_id=f"{paper.paper_id}:{index}",
            paper_id=paper.paper_id,
            title=paper.title,
            text=chunk_text,
            metadata={"year": paper.year, "authors": paper.authors, "source": paper.source},
        )
        for index, chunk_text in enumerate(chunker.split(text))
    ]
