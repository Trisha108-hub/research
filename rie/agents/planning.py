from rie.llm.granite import GraniteClient
from rie.state import ResearchState


class PlanningAgent:
    def __init__(self, llm: GraniteClient | None = None):
        self.llm = llm or GraniteClient()

    def run(self, state: ResearchState) -> ResearchState:
        prompt = (
            "Create multiple distinct, concise academic search queries that cover the important "
            "aspects of this research question. Use one short query per line, without numbering "
            "or explanation.\n"
            f"Question: {state.research_question}"
        )
        lines = [line.strip(" -*0123456789.\t") for line in self.llm.generate(prompt, max_tokens=600).splitlines()]
        queries = []
        seen = set()
        for line in lines:
            key = line.casefold()
            if line and key not in seen:
                queries.append(line)
                seen.add(key)
        state.subqueries = queries or [state.research_question]
        return state
