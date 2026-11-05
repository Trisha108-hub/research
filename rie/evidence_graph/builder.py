from pathlib import Path

import networkx as nx

from rie.state import ResearchState


def build_evidence_graph(state: ResearchState) -> nx.DiGraph:
    graph = nx.DiGraph()
    for paper in state.papers:
        graph.add_node(paper.paper_id, label=paper.title[:50], node_type="paper", title=paper.title)
    for evidence in state.deep_evidence:
        if evidence.methodology:
            node_id = f"method:{evidence.methodology[:40]}"
            graph.add_node(node_id, label=evidence.methodology[:40], node_type="methodology")
            graph.add_edge(evidence.paper_id, node_id, relation="uses_methodology")
        for dataset in evidence.datasets:
            node_id = f"dataset:{dataset[:40]}"
            graph.add_node(node_id, label=dataset[:40], node_type="dataset")
            graph.add_edge(evidence.paper_id, node_id, relation="uses_dataset")
    for contradiction in state.contradictions:
        if graph.has_node(contradiction.paper_a_id) and graph.has_node(contradiction.paper_b_id):
            graph.add_edge(
                contradiction.paper_a_id,
                contradiction.paper_b_id,
                relation="contradicts",
                topic=contradiction.topic,
            )
    for index, gap in enumerate(state.research_gaps):
        gap_id = f"gap:{index}"
        graph.add_node(gap_id, label=gap.title[:50], node_type="gap", description=gap.description)
        for paper_id in gap.supporting_paper_ids:
            if graph.has_node(paper_id):
                graph.add_edge(paper_id, gap_id, relation="supports_gap")
    return graph


def export_to_html(graph: nx.DiGraph, output_path: str | Path = "data/evidence_graph.html") -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        return ""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colours = {"paper": "#3b82d4", "methodology": "#7c5cd8", "dataset": "#15803d", "gap": "#c2410c"}
    network = Network(height="700px", width="100%", directed=True)
    for node_id, attributes in graph.nodes(data=True):
        network.add_node(
            str(node_id),
            label=attributes.get("label", str(node_id)),
            color=colours.get(attributes.get("node_type"), "#57606a"),
            title=attributes.get("title", attributes.get("description", str(node_id))),
        )
    for source, target, attributes in graph.edges(data=True):
        network.add_edge(str(source), str(target), title=attributes.get("relation", ""))
    network.save_graph(str(output_path))
    return str(output_path)
