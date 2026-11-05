import json
import re

from rie.llm.granite import GraniteClient
from rie.state import DeepEvidence, Paper, ResearchState


PROMPT = """Extract structured information from the paper.
Return JSON with objective, methodology, datasets, metrics, key_results, limitations, and contributions.
Paper title: {title}
Abstract: {abstract}
Respond with valid JSON only."""


class DeepExtractionAgent:
    def __init__(self, llm: GraniteClient | None = None):
        self.llm = llm or GraniteClient()

    def run(self, state: ResearchState) -> ResearchState:
        state.deep_evidence = [
            self.extract(paper)
            for paper in state.papers[:30]
        ]
        return state

    def extract(self, paper: Paper) -> DeepEvidence:
        raw = self.llm.generate(
            PROMPT.format(title=paper.title, abstract=(paper.abstract or paper.title)[:1200]),
            max_tokens=500,
        )
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            data = {}
        return DeepEvidence(
            paper_id=paper.paper_id,
            title=paper.title,
            objective=str(data.get("objective", "") or ""),
            methodology=str(data.get("methodology", "") or ""),
            datasets=data.get("datasets", []) if isinstance(data.get("datasets"), list) else [],
            metrics=data.get("metrics", []) if isinstance(data.get("metrics"), list) else [],
            key_results=str(data.get("key_results", "") or ""),
            limitations=str(data.get("limitations", "") or ""),
            contributions=str(data.get("contributions", "") or ""),
        )
