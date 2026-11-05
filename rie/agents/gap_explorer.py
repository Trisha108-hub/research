from rie.llm.granite import GraniteClient
from rie.state import CandidateGap, ResearchState


class GapExplorerAgent:
    def __init__(self, llm: GraniteClient | None = None):
        self.llm = llm or GraniteClient()

    def run(self, state: ResearchState) -> ResearchState:
        limitations = [item.limitations for item in state.deep_evidence if item.limitations.strip()]
        if not limitations:
            state.research_gaps = []
            return state

        prompt = (
            "Generate 3-5 candidate research gaps from these limitations. "
            "Use TITLE | DESCRIPTION | CONFIDENCE lines.\n"
            + "\n".join(f"[{item.paper_id}] {item.limitations}" for item in state.deep_evidence if item.limitations)
        )
        raw = self.llm.generate(prompt, 800)
        gaps = []
        paper_ids = [item.paper_id for item in state.deep_evidence[:5]]
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 2:
                continue
            try:
                confidence = float(parts[2]) if len(parts) > 2 else 0.5
                gaps.append(CandidateGap(
                    title=parts[0],
                    description=parts[1],
                    supporting_paper_ids=paper_ids,
                    evidence_summary=parts[1],
                    confidence=max(0.0, min(confidence, 1.0)),
                ))
            except ValueError:
                continue
        state.research_gaps = gaps
        return state
