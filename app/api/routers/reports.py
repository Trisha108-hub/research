"""Report and evaluation endpoints."""

from fastapi import APIRouter

from app.api.models import EvaluationRequest, EvaluationResponse
from rie.evaluation.citation_eval import evaluate_citations
from rie.evaluation.ragas_eval import RagasSample, evaluate_with_ragas

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/evaluate", response_model=EvaluationResponse, summary="Evaluate a generated answer")
async def evaluate_answer(request: EvaluationRequest) -> EvaluationResponse:
    citation_result = evaluate_citations(request.answer, [])
    ragas_result = evaluate_with_ragas([
        RagasSample(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth,
        )
    ])
    return EvaluationResponse(
        citation_hit_rate=citation_result["citation_hit_rate"],
        false_citation_rate=citation_result["false_citation_rate"],
        cited_count=citation_result["cited_count"],
        hallucinated_ids=citation_result["hallucinated_ids"],
        ragas=ragas_result.as_dict(),
    )
