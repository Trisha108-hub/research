from typing import Any, Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    cited_by_count: int = 0
    source: str = "unknown"


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    query: str
    paper: Paper
    score: float | None = None


class Evidence(BaseModel):
    claim: str
    summary: str
    chunk: Chunk
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    stance: Literal["supports", "contradicts", "neutral"] = "neutral"


class DeepEvidence(BaseModel):
    paper_id: str
    title: str
    objective: str = ""
    methodology: str = ""
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    key_results: str = ""
    limitations: str = ""
    contributions: str = ""


class ComparisonEntry(BaseModel):
    paper_id: str
    title: str
    methodology: str = ""
    datasets: str = ""
    best_metric: str = ""
    limitations: str = ""


class Contradiction(BaseModel):
    paper_a_id: str
    paper_a_title: str
    paper_b_id: str
    paper_b_title: str
    topic: str
    description: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class VerifiedClaim(BaseModel):
    claim: str
    supported: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    flag: str = ""


class CandidateGap(BaseModel):
    title: str
    description: str
    supporting_paper_ids: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ResearchState(BaseModel):
    research_question: str
    subqueries: list[str] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    report: str = ""
    deep_evidence: list[DeepEvidence] = Field(default_factory=list)
    comparison_matrix: list[ComparisonEntry] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    research_gaps: list[CandidateGap] = Field(default_factory=list)
    synthesis: str = ""
    errors: list[str] = Field(default_factory=list)
