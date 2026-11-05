from rie.agents.planning import PlanningAgent
from rie.llm.granite import GraniteClient
from rie.state import ResearchState


def test_planning_agent_produces_nonempty_queries():
    agent = PlanningAgent(GraniteClient())
    state = ResearchState(research_question="How does RAG improve factual accuracy?")

    result = agent.run(state)

    assert result.subqueries
    assert all(query.strip() for query in result.subqueries)
