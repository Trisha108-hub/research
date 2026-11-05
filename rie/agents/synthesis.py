from rie.llm.granite import GraniteClient
from rie.state import ResearchState


class SynthesisAgent:
    def __init__(self, llm: GraniteClient | None = None):
        self.llm = llm or GraniteClient()

    def run(self, state: ResearchState) -> ResearchState:
        methodologies = "\n".join(
            f"- [{item.paper_id}] {item.methodology}"
            for item in state.deep_evidence if item.methodology
        ) or "No methodologies extracted."
        findings = "\n".join(
            f"- [{item.paper_id}] {item.key_results}"
            for item in state.deep_evidence if item.key_results
        ) or "No key results extracted."
        contradictions = "\n".join(
            f"- {item.topic}: {item.description}" for item in state.contradictions
        ) or "No contradictions detected."
        gaps = "\n".join(f"- {item.title}: {item.description}" for item in state.research_gaps) or "No gaps identified."

        prompt = (
            "Write an evidence-grounded synthesis with Executive Summary, Literature Overview, "
            "Methodological Analysis, Key Findings, Contradictions, Research Limitations, "
            "Candidate Research Gaps, and Conclusion.\n"
            f"Question: {state.research_question}\nMethodologies:\n{methodologies}\n"
            f"Findings:\n{findings}\nContradictions:\n{contradictions}\nGaps:\n{gaps}"
        )
        state.synthesis = self.llm.generate(prompt, 1500)
        state.report = state.synthesis
        return state
