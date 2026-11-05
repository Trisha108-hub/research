from collections.abc import Callable

from rie.agents.extraction import EvidenceExtractionAgent
from rie.agents.planning import PlanningAgent
from rie.agents.report import ReportEngine
from rie.config import get_settings
from rie.llm.granite import GraniteClient
from rie.retrieval.chunker import TextChunker, chunk_paper
from rie.retrieval.crossref import CrossrefClient
from rie.retrieval.openalex import OpenAlexClient
from rie.retrieval.semantic_scholar import SemanticScholarClient
from rie.retrieval.vector_store import FaissVectorStore
from rie.state import ResearchState


class ResearchWorkflow:
    """Orchestrates Phase 1 retrieval and synthesis."""

    def __init__(
        self,
        llm: GraniteClient | None = None,
        openalex: OpenAlexClient | None = None,
        semantic_scholar: SemanticScholarClient | None = None,
        vector_store: FaissVectorStore | None = None,
        top_k: int = 5,
        crossref: CrossrefClient | None = None,
    ):
        settings = get_settings()
        llm = llm or GraniteClient(settings)
        self.planner = PlanningAgent(llm)
        self.extractor = EvidenceExtractionAgent(llm)
        self.reporter = ReportEngine(llm)
        self.openalex = openalex
        self.semantic_scholar = semantic_scholar or SemanticScholarClient(settings.semantic_scholar_api_key)
        self.crossref = crossref or CrossrefClient(settings.crossref_mailto)
        self.vector_store = vector_store or FaissVectorStore(settings.embedding_model)
        self.top_k = top_k

    def run(
        self,
        state: ResearchState,
        results_per_source: int = 10,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> ResearchState:
        def report_progress(percent: int, stage: str) -> None:
            if progress_callback:
                progress_callback(percent, stage)

        report_progress(5, "Planning focused searches")
        self.planner.run(state)
        report_progress(15, "Search plan ready · finding academic papers")
        seen = {paper.paper_id for paper in state.papers}
        query_count = max(1, len(state.subqueries))
        for query_index, query in enumerate(state.subqueries):
            report_progress(
                15 + int(32 * query_index / query_count),
                f"Searching Semantic Scholar · query {query_index + 1} of {query_count}",
            )
            query_has_results = False
            if self.openalex is not None:
                try:
                    papers = self.openalex.search(query, limit=results_per_source)
                except Exception as exc:
                    state.errors.append(f"OpenAlex search failed for '{query}': {exc}")
                    papers = []
                query_has_results = query_has_results or bool(papers)
                for paper in papers:
                    if paper.paper_id and paper.paper_id not in seen:
                        state.papers.append(paper)
                        seen.add(paper.paper_id)

            try:
                papers = self.semantic_scholar.search(query, limit=results_per_source)
            except Exception as exc:
                state.errors.append(f"Semantic Scholar search failed for '{query}': {exc}")
                papers = []
            query_has_results = query_has_results or bool(papers)
            for paper in papers:
                if paper.paper_id and paper.paper_id not in seen:
                    state.papers.append(paper)
                    seen.add(paper.paper_id)

            if not query_has_results:
                report_progress(15 + int(32 * (query_index + 0.5) / query_count), "Trying an additional academic index")
                try:
                    papers = self.crossref.search(query, limit=results_per_source)
                except Exception as exc:
                    state.errors.append(f"Crossref fallback search failed: {exc}")
                    papers = []
                for paper in papers:
                    if paper.paper_id and paper.paper_id not in seen:
                        state.papers.append(paper)
                        seen.add(paper.paper_id)
            report_progress(
                15 + int(32 * (query_index + 1) / query_count),
                f"Finished search {query_index + 1} of {query_count}",
            )

        report_progress(52, f"Organizing {len(state.papers)} papers")
        chunker = TextChunker()
        all_chunks = [
            chunk
            for paper in state.papers
            for chunk in chunk_paper(paper, chunker)
        ]
        state.chunks = []
        if all_chunks:
            report_progress(60, "Ranking passages against your question")
            self.vector_store.build(all_chunks)
            seen_chunk_ids = set()
            for query in state.subqueries:
                for chunk, _score in self.vector_store.search(query, k=self.top_k):
                    if chunk.chunk_id not in seen_chunk_ids:
                        state.chunks.append(chunk)
                        seen_chunk_ids.add(chunk.chunk_id)
        report_progress(68, f"Reviewing {len(state.chunks)} evidence passages")
        self.extractor.run(
            state,
            progress_callback=lambda done, total: report_progress(
                68 + int(22 * done / max(1, total)),
                f"Extracting evidence · passage {done} of {total}",
            ),
        )
        report_progress(93, "Writing the source-linked review")
        self.reporter.render(state)
        report_progress(100, "Your literature review is ready")
        return state
