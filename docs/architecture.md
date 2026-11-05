# System Architecture

## Overview

The Research Intelligence Engine converts a research question into an evidence-grounded Markdown report.

## Phase 1 Pipeline

ResearchState

- PlanningAgent decomposes the question into subqueries.
- Semantic Scholar retrieves papers in the main workflow, with Crossref as a fallback. OpenAlex remains available to the optional Phase 2 workflow.
- TextChunker creates overlapping text chunks.
- FaissVectorStore retrieves the most relevant chunks.
- EvidenceExtractionAgent extracts claims and evidence.
- ReportEngine generates the final Markdown report.

## ResearchState

| Field | Type | Description |
|---|---|---|
| research_question | `str` | Original user query |
| subqueries | `list[str]` | Decomposed search queries |
| papers | `list[Paper]` | Retrieved papers |
| chunks | `list[Chunk]` | Top-K RAG chunks |
| evidence | `list[Evidence]` | Extracted evidence items |
| report | `str` | Final Markdown report |
| errors | `list[str]` | Non-fatal errors |

## Granite Integration

GraniteClient supports SDK mode, REST fallback, and offline deterministic mode.

## Vector Store

FAISS uses L2-normalized `all-MiniLM-L6-v2` embeddings. Inner product therefore approximates cosine similarity.
