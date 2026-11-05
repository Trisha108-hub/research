"""Research endpoints."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.models import EvidenceItem, PaperItem, ResearchRequest, ResearchResponse
from rie.config import get_settings
from rie.graph.workflow import ResearchWorkflow
from rie.retrieval.openalex import OpenAlexClient
from rie.retrieval.semantic_scholar import SemanticScholarClient
from rie.state import ResearchState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])
_jobs: dict[str, dict] = {}


def _make_response(state: ResearchState) -> ResearchResponse:
    return ResearchResponse(
        question=state.research_question,
        subqueries=state.subqueries,
        report=state.report,
        papers=[
            PaperItem(
                paper_id=paper.paper_id,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                source=paper.source,
                citations=paper.cited_by_count,
                url=paper.url,
            )
            for paper in state.papers
        ],
        evidence=[
            EvidenceItem(
                claim=evidence.claim,
                summary=evidence.summary,
                source_title=evidence.chunk.title,
                confidence=evidence.confidence,
                stance=evidence.stance,
            )
            for evidence in state.evidence
        ],
        contradiction_count=len(state.contradictions),
        gap_count=len(state.research_gaps),
        errors=state.errors,
    )


def _run_pipeline(
    request: ResearchRequest,
    progress_callback=None,
) -> ResearchResponse:
    settings = get_settings()
    state = ResearchState(research_question=request.question)
    if request.phase2:
        from rie.graph.langgraph_workflow import run_langgraph

        state = run_langgraph(
            question=request.question,
            results_per_source=request.results_per_source,
            top_k=request.top_k,
            export_graph=False,
            openalex=OpenAlexClient(),
            semantic_scholar=SemanticScholarClient(settings.semantic_scholar_api_key),
        )
    else:
        workflow = ResearchWorkflow(
            semantic_scholar=SemanticScholarClient(settings.semantic_scholar_api_key),
            top_k=request.top_k,
        )
        workflow.run(
            state,
            results_per_source=request.results_per_source,
            progress_callback=progress_callback,
        )
    return _make_response(state)


def _run_job(job_id: str, request: ResearchRequest) -> None:
    def update_progress(percent: int, stage: str) -> None:
        job = _jobs.get(job_id)
        if job:
            job.update(progress=percent, stage=stage)

    try:
        result = _run_pipeline(request, progress_callback=update_progress)
        _jobs[job_id].update(
            status="complete",
            progress=100,
            stage="Your literature review is ready",
            result=result.model_dump(),
        )
    except Exception as exc:
        logger.exception("Pipeline failed for question: %s", request.question)
        _jobs[job_id].update(status="failed", error=str(exc), stage="The research run could not be completed")


@router.post("/run", response_model=ResearchResponse, summary="Run the research pipeline")
async def run_research(request: ResearchRequest) -> ResearchResponse:
    try:
        return _run_pipeline(request)
    except Exception as exc:
        logger.exception("Pipeline failed for question: %s", request.question)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/jobs", status_code=202, summary="Start a research run with progress updates")
async def start_research_job(request: ResearchRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = uuid4().hex
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 2,
        "stage": "Preparing your research run",
    }
    background_tasks.add_task(_run_job, job_id, request)
    return {key: value for key, value in _jobs[job_id].items() if key != "result"}


@router.get("/jobs/{job_id}", summary="Read research run progress")
async def get_research_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Research job was not found or has expired.")
    return job


@router.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "ok"}
