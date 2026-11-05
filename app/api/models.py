"""FastAPI request and response schemas."""

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=500)
    results_per_source: int = Field(default=10, ge=1, le=50)
    top_k: int = Field(default=5, ge=1, le=20)
    phase2: bool = Field(default=False, description="Use the full Phase 2 LangGraph pipeline")


class EvidenceItem(BaseModel):
    claim: str
    summary: str
    source_title: str
    confidence: float
    stance: str


class PaperItem(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    source: str
    citations: int
    url: str | None


class ResearchResponse(BaseModel):
    question: str
    subqueries: list[str]
    report: str
    papers: list[PaperItem]
    evidence: list[EvidenceItem]
    contradiction_count: int
    gap_count: int
    errors: list[str]


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


class EvaluationResponse(BaseModel):
    citation_hit_rate: float
    false_citation_rate: float
    cited_count: int
    hallucinated_ids: list[str]
    ragas: dict
