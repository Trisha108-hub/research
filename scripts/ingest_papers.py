"""Ingest OpenAlex and Semantic Scholar results into a FAISS index."""

import argparse

from rie.config import get_settings
from rie.retrieval.chunker import chunk_paper
from rie.retrieval.openalex import OpenAlexClient
from rie.retrieval.semantic_scholar import SemanticScholarClient
from rie.retrieval.vector_store import FaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    settings = get_settings()
    openalex_papers = OpenAlexClient().search(args.query, args.limit)
    scholar_papers = SemanticScholarClient(settings.semantic_scholar_api_key).search(args.query, args.limit)
    unique = {paper.paper_id: paper for paper in openalex_papers + scholar_papers}
    chunks = [chunk for paper in unique.values() for chunk in chunk_paper(paper)]

    store = FaissVectorStore(settings.embedding_model)
    store.build(chunks)
    store.save(settings.faiss_index_path, settings.faiss_text_path)
    print(f"Ingested {len(unique)} papers and {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
