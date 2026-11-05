import re
from collections.abc import Callable

from rie.llm.granite import GraniteClient
from rie.state import Evidence, ResearchState


class EvidenceExtractionAgent:
    def __init__(self, llm: GraniteClient | None = None):
        self.llm = llm or GraniteClient()

    def run(
        self,
        state: ResearchState,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ResearchState:
        evidence = []
        chunks = state.chunks[:20]
        if progress_callback:
            progress_callback(0, len(chunks))
        for index, chunk in enumerate(chunks, start=1):
            prompt = (
                "Extract evidence for the research question. "
                f"Question: {state.research_question}\n"
                f"Source title: {chunk.title}\n"
                f"Source text: {chunk.text}\n"
                "Format: Claim | Summary | Confidence (0-1) | stance (supports/contradicts/neutral)"
            )
            output = self.llm.generate(prompt, max_tokens=300)
            evidence.append(self._parse(output, chunk))
            if progress_callback:
                progress_callback(index, len(chunks))
        state.evidence = evidence
        return state

    def _parse(self, output: str, chunk) -> Evidence:
        confidence_match = re.search(r"Confidence:\s*(0(?:\.\d+)?|1(?:\.0+)?)", output, re.I)
        stance_match = re.search(r"Stance:\s*(supports|contradicts|neutral)", output, re.I)
        claim_match = re.search(r"(?:Claim:\s*|Claim\|)(.+?)(?:\n|$)", output, re.I)
        summary_match = re.search(r"Summary:\s*(.+?)(?:\n|Confidence:|$)", output, re.I | re.S)
        claim = claim_match.group(1).strip() if claim_match else output.splitlines()[0][:180]
        summary = summary_match.group(1).strip() if summary_match else output[:300]
        stance = stance_match.group(1).lower() if stance_match else "neutral"
        return Evidence(
            claim=claim,
            summary=summary.replace("|", "").strip(),
            chunk=chunk,
            confidence=float(confidence_match.group(1)) if confidence_match else 0.5,
            stance=stance,
        )
