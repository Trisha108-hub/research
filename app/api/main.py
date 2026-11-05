"""FastAPI application entry point."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import reports, research

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Research Intelligence Engine API",
    description="Evidence-grounded multi-agent academic literature analysis.",
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(research.router)
app.include_router(reports.router)


@app.get("/", summary="Research Intelligence Engine website", include_in_schema=False)
async def root() -> FileResponse:
    website = Path(__file__).resolve().parents[2] / "index.html"
    return FileResponse(website, media_type="text/html")


@app.get("/api", summary="API information")
async def api_info() -> dict[str, str]:
    return {"name": "Research Intelligence Engine", "version": "0.3.0"}
