"""JSONL benchmark dataset utilities."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


BENCHMARK_PATH = Path("data/benchmarks/benchmark.jsonl")


class BenchmarkEntry(BaseModel):
    question: str
    relevant_paper_ids: list[str] = Field(default_factory=list)
    expected_summary: str = ""
    known_citation_ids: list[str] = Field(default_factory=list)
    domain: str = "general"
    notes: str = ""


def load_benchmark(path: Path | str = BENCHMARK_PATH) -> list[BenchmarkEntry]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        return []
    entries: list[BenchmarkEntry] = []
    with benchmark_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(BenchmarkEntry.model_validate_json(line))
    return entries


def save_benchmark(entries: list[BenchmarkEntry], path: Path | str = BENCHMARK_PATH) -> None:
    benchmark_path = Path(path)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json() + "\n")


def seed_benchmark(path: Path | str = BENCHMARK_PATH) -> list[BenchmarkEntry]:
    entries = [
        BenchmarkEntry(
            question="How does retrieval-augmented generation affect factual accuracy in LLMs?",
            expected_summary="RAG grounds generated answers in retrieved evidence and reduces unsupported claims.",
            domain="nlp",
        ),
        BenchmarkEntry(
            question="What are the main limitations of transformer-based language models?",
            expected_summary="Important limitations include context length, cost, hallucination, and bias.",
            domain="nlp",
        ),
        BenchmarkEntry(
            question="How do prompt engineering techniques affect LLM performance?",
            expected_summary="Instruction and reasoning prompts can improve task-specific model performance.",
            domain="nlp",
        ),
        BenchmarkEntry(
            question="What evaluation metrics are used for open-domain question answering?",
            expected_summary="Common metrics include exact match, F1, faithfulness, and answer relevancy.",
            domain="nlp",
        ),
        BenchmarkEntry(
            question="How does fine-tuning compare to in-context learning in large language models?",
            expected_summary="Fine-tuning is task-specific and data-intensive; in-context learning is flexible.",
            domain="nlp",
        ),
    ]
    save_benchmark(entries, path)
    return entries
