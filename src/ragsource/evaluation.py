"""Deterministic, offline evaluation of ranked retrieval results."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank_at_k: float


def evaluate_ranking(
    ranked_ids: list[str], relevant_ids: set[str], k: int
) -> RetrievalMetrics:
    """Evaluate one query using chunk IDs; rank is one-based.

    The input ranking must not repeat chunk IDs. For empty relevance sets,
    recall and reciprocal rank are defined as zero (rather than undefined).
    """
    if k < 1:
        raise ValueError("k must be positive")
    if not all(isinstance(value, str) and value for value in ranked_ids):
        raise ValueError("ranked_ids must contain nonempty strings")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked_ids must be unique")
    if not all(isinstance(value, str) and value for value in relevant_ids):
        raise ValueError("relevant_ids must contain nonempty strings")

    ranked = ranked_ids[:k]
    hits = sum(item in relevant_ids for item in ranked)
    first_hit = next((position for position, item in enumerate(ranked, 1) if item in relevant_ids), None)
    return RetrievalMetrics(
        precision_at_k=hits / k,
        recall_at_k=hits / len(relevant_ids) if relevant_ids else 0.0,
        reciprocal_rank_at_k=1 / first_hit if first_hit else 0.0,
    )


def mean_metrics(items: list[RetrievalMetrics]) -> RetrievalMetrics:
    if not items:
        raise ValueError("evaluation requires at least one query")
    if any(
        not all(isfinite(value) and 0 <= value <= 1 for value in (
            item.precision_at_k, item.recall_at_k, item.reciprocal_rank_at_k
        ))
        for item in items
    ):
        raise ValueError("invalid metric value")
    size = len(items)
    return RetrievalMetrics(
        precision_at_k=sum(item.precision_at_k for item in items) / size,
        recall_at_k=sum(item.recall_at_k for item in items) / size,
        reciprocal_rank_at_k=sum(item.reciprocal_rank_at_k for item in items) / size,
    )
