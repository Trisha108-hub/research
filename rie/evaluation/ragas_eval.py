"""Optional RAGAS evaluation wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RagasSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass
class RagasResult:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    sample_count: int = 0
    error: str = ""

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "sample_count": self.sample_count,
            "error": self.error,
        }


def evaluate_with_ragas(samples: list[RagasSample]) -> RagasResult:
    if not samples:
        return RagasResult(error="No samples provided.")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        return RagasResult(error=f"Import error: {exc}")

    data = {
        "question": [sample.question for sample in samples],
        "answer": [sample.answer for sample in samples],
        "contexts": [sample.contexts for sample in samples],
        "ground_truth": [sample.ground_truth for sample in samples],
    }
    try:
        result = evaluate(Dataset.from_dict(data), metrics=[faithfulness, answer_relevancy, context_precision])
        frame = result.to_pandas()
        return RagasResult(
            faithfulness=float(frame["faithfulness"].mean()),
            answer_relevancy=float(frame["answer_relevancy"].mean()),
            context_precision=float(frame["context_precision"].mean()),
            sample_count=len(samples),
        )
    except Exception as exc:
        logger.exception("RAGAS evaluation failed")
        return RagasResult(error=str(exc))
