"""Unit tests for FastAPI request and response schemas."""

import pytest
from pydantic import ValidationError

from app.api.models import ResearchRequest


def test_research_request_defaults():
    request = ResearchRequest(question="How does RAG affect factual accuracy in LLMs?")
    assert request.results_per_source == 10
    assert request.top_k == 5
    assert request.phase2 is False


def test_research_request_question_too_short():
    with pytest.raises(ValidationError):
        ResearchRequest(question="Too short")


def test_research_request_custom_values():
    request = ResearchRequest(
        question="A sufficiently long research question for testing.",
        top_k=8,
        phase2=True,
    )
    assert request.top_k == 8
    assert request.phase2 is True
