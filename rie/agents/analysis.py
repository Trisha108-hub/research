from rie.llm.granite import GraniteClient
from rie.state import ComparisonEntry, Contradiction, DeepEvidence, ResearchState


CONTRADICTION_PROMPT = """Review these papers and identify factual contradictions.
Use one line per contradiction: PAPER_A_ID | PAPER_B_ID | TOPIC | DESCRIPTION | CONFIDENCE
Respond NONE if none are found.
{summaries}"""


class AnalysisAgent:
    def __init__(self, llm: GraniteClient | None = None):
        self.llm = llm or GraniteClient()

    def run(self, state: ResearchState) -> ResearchState:
        state.comparison_matrix = [
            ComparisonEntry(
                paper_id=item.paper_id,
                title=item.title,
                methodology=item.methodology or "-",
                datasets=", ".join(item.datasets) or "-",
                best_metric=", ".join(item.metrics[:2]) or "-",
                limitations=item.limitations or "-",
            )
            for item in state.deep_evidence
        ]
        state.contradictions = self._find_contradictions(state)
        return state

    def _find_contradictions(self, state: ResearchState) -> list[Contradiction]:
        if len(state.deep_evidence) < 2:
            return []
        summaries = "\n\n".join(
            f"[{item.paper_id}] {item.title}\nResults: {item.key_results}\nLimitations: {item.limitations}"
            for item in state.deep_evidence[:15]
        )
        raw = self.llm.generate(CONTRADICTION_PROMPT.format(summaries=summaries), 600)
        if "NONE" in raw.upper():
            return []

        titles = {item.paper_id: item.title for item in state.deep_evidence}
        found = []
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 4:
                continue
            try:
                confidence = float(parts[4]) if len(parts) > 4 else 0.5
                found.append(Contradiction(
                    paper_a_id=parts[0],
                    paper_a_title=titles.get(parts[0], parts[0]),
                    paper_b_id=parts[1],
                    paper_b_title=titles.get(parts[1], parts[1]),
                    topic=parts[2],
                    description=parts[3],
                    confidence=max(0.0, min(confidence, 1.0)),
                ))
            except ValueError:
                continue
        return found
