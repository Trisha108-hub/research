import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rie.llm.granite import GraniteClient
from rie.state import ResearchState


class ReportEngine:
    def __init__(self, llm: GraniteClient | None = None):
        self.llm = llm or GraniteClient()
        template_dir = Path(__file__).resolve().parent / "templates"
        self.environment = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        )

    def render(self, state: ResearchState) -> str:
        papers = {paper.paper_id: paper for paper in state.papers}
        source_ids: dict[str, str] = {}
        for item in state.evidence:
            source_ids.setdefault(item.chunk.paper_id, f"R{len(source_ids) + 1}")

        evidence_lines = []
        for item in state.evidence:
            paper = papers.get(item.chunk.paper_id)
            year = f", {paper.year}" if paper and paper.year else ""
            reference = source_ids[item.chunk.paper_id]
            evidence_lines.append(
                f"- [{reference}] {item.chunk.title}{year}: {item.claim}. "
                f"Evidence summary: {item.summary} (extraction confidence {item.confidence:.2f}; "
                f"stance: {item.stance})"
            )
        evidence_text = "\n".join(evidence_lines) or "No evidence was extracted."
        prompt = (
            "Write a detailed, readable literature review that directly answers the research question. "
            "Use these sections: Executive Summary, Background, Findings and Themes, Comparison of "
            "Evidence, Limitations and Research Gaps, and Conclusion. Explain what the retrieved "
            "studies support, how their findings relate, and where the evidence is mixed or insufficient. "
            "Discuss methods or populations only when they appear in the supplied evidence. Use several "
            "substantive paragraphs; avoid generic filler and repeated conclusions. Ground every factual "
            "statement in the supplied evidence and cite it with the exact source labels such as [R1]. "
            "Never invent a source, result, method, or citation. Clearly distinguish evidence from inference "
            "and say when the retrieved set cannot answer part of the question. Do not create a References "
            "section; the application will add verified references.\n"
            f"Question: {state.research_question}\nEvidence:\n{evidence_text}"
        )
        generated = self.llm.generate(prompt, max_tokens=1800)
        # Keep only model narrative here; references are assembled from retrieved records below.
        generated = re.split(r"(?im)^#{1,3}\s*references\s*$", generated, maxsplit=1)[0].strip()
        references = []
        for paper_id, reference in source_ids.items():
            paper = papers.get(paper_id)
            if paper is None:
                continue
            authors = ", ".join(paper.authors[:3]) or "Unknown authors"
            if len(paper.authors) > 3:
                authors += ", et al."
            year = str(paper.year) if paper.year else "n.d."
            link = paper.url or (f"https://doi.org/{paper.doi.removeprefix('https://doi.org/')}" if paper.doi else "")
            title = f"[{paper.title}]({link})" if link else paper.title
            references.append(f"- [{reference}] {authors} ({year}). {title}.")
        reference_section = "\n".join(references) if references else "No linked source records were available."
        state.report = f"{generated}\n\n## References\n\n{reference_section}".strip()
        return state.report

    def render_markdown_template(self, state: ResearchState) -> str:
        template = self.environment.get_template("report.md.jinja")
        return template.render(state=state)

    def render_full_report(self, state: ResearchState) -> str:
        template = self.environment.get_template("report_full.md.jinja")
        state.report = template.render(state=state)
        return state.report
