import logging
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from rie.agents.analysis import AnalysisAgent
from rie.agents.deep_extraction import DeepExtractionAgent
from rie.agents.extraction import EvidenceExtractionAgent
from rie.agents.gap_explorer import GapExplorerAgent
from rie.agents.planning import PlanningAgent
from rie.agents.report import ReportEngine
from rie.agents.synthesis import SynthesisAgent
from rie.agents.verification import VerificationAgent
from rie.config import get_settings
from rie.evidence_graph.builder import build_evidence_graph, export_to_html
from rie.llm.granite import GraniteClient
from rie.retrieval.chunker import TextChunker, chunk_paper
from rie.retrieval.openalex import OpenAlexClient
from rie.retrieval.semantic_scholar import SemanticScholarClient
from rie.retrieval.vector_store import FaissVectorStore
from rie.state import ResearchState

logger = logging.getLogger(__name__)


def _safe(name: str, function: Callable[[ResearchState], ResearchState]):
    def wrapped(state_dict: dict[str, Any]) -> dict[str, Any]:
        state = ResearchState.model_validate(state_dict)
        try:
            return function(state).model_dump()
        except Exception as exc:
            logger.exception("%s failed", name)
            state.errors.append(f"{name} failed: {exc}")
            return state.model_dump()
    return wrapped


def build_langgraph(
    llm: GraniteClient | None = None,
    openalex: OpenAlexClient | None = None,
    semantic_scholar: SemanticScholarClient | None = None,
    results_per_source: int = 10,
    top_k: int = 5,
    export_graph: bool = True,
) -> Any:
    settings = get_settings()
    llm = llm or GraniteClient(settings)
    openalex_client = openalex or OpenAlexClient()
    scholar_client = semantic_scholar or SemanticScholarClient(settings.semantic_scholar_api_key)
    planner = PlanningAgent(llm)
    extractor = EvidenceExtractionAgent(llm)
    deep_extractor = DeepExtractionAgent(llm)
    analyser = AnalysisAgent(llm)
    gap_explorer = GapExplorerAgent(llm)
    synthesiser = SynthesisAgent(llm)
    reporter = ReportEngine(llm)
    vector_store = FaissVectorStore(settings.embedding_model)

    def plan(state: ResearchState):
        return planner.run(state)

    def retrieve(state: ResearchState):
        seen = {paper.paper_id for paper in state.papers}
        for query in state.subqueries:
            for source, client in (("OpenAlex", openalex_client), ("Semantic Scholar", scholar_client)):
                try:
                    papers = client.search(query, results_per_source)
                except Exception as exc:
                    state.errors.append(f"{source} failed for '{query}': {exc}")
                    papers = []
                for paper in papers:
                    if paper.paper_id and paper.paper_id not in seen:
                        state.papers.append(paper)
                        seen.add(paper.paper_id)
        return state

    def chunk_index(state: ResearchState):
        all_chunks = [
            chunk
            for paper in state.papers
            for chunk in chunk_paper(paper, TextChunker())
        ]
        if all_chunks:
            vector_store.build(all_chunks)
            seen_ids = set()
            for query in state.subqueries:
                for chunk, _score in vector_store.search(query, k=top_k):
                    if chunk.chunk_id not in seen_ids:
                        state.chunks.append(chunk)
                        seen_ids.add(chunk.chunk_id)
        return state

    def extract(state: ResearchState):
        return extractor.run(state)

    def deep_extract(state: ResearchState):
        return deep_extractor.run(state)

    def analyse(state: ResearchState):
        return analyser.run(state)

    def verify(state: ResearchState):
        return VerificationAgent(llm, vector_store).run(state)

    def explore_gaps(state: ResearchState):
        return gap_explorer.run(state)

    def synthesise(state: ResearchState):
        synthesiser.run(state)
        reporter.render_full_report(state)
        return state

    def graph_export(state: ResearchState):
        if export_graph:
            try:
                graph = build_evidence_graph(state)
                export_to_html(graph)
            except Exception as exc:
                state.errors.append(f"Graph export failed: {exc}")
        return state

    workflow = StateGraph(dict)
    nodes = [
        ("plan", plan),
        ("retrieve", retrieve),
        ("chunk_index", chunk_index),
        ("extract", extract),
        ("deep_extract", deep_extract),
        ("analyse", analyse),
        ("verify", verify),
        ("explore_gaps", explore_gaps),
        ("synthesise", synthesise),
        ("graph_export", graph_export),
    ]
    for name, function in nodes:
        workflow.add_node(name, _safe(name, function))

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "chunk_index")
    workflow.add_edge("chunk_index", "extract")
    workflow.add_edge("extract", "deep_extract")
    workflow.add_edge("deep_extract", "analyse")
    workflow.add_edge("analyse", "verify")
    workflow.add_edge("verify", "explore_gaps")
    workflow.add_edge("explore_gaps", "synthesise")
    workflow.add_edge("synthesise", "graph_export")
    workflow.add_edge("graph_export", END)
    return workflow.compile()


def run_langgraph(question: str, **kwargs) -> ResearchState:
    app = build_langgraph(**kwargs)
    result = app.invoke(ResearchState(research_question=question).model_dump())
    return ResearchState.model_validate(result)
