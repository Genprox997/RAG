"""Retrieval-level evaluation metrics (source-grounded, deterministic).

These complement the LLM-judged metrics (faithfulness / answer relevancy /
context relevance) with reproducible retrieval signals:
  - recall@k   : fraction of relevant SOURCES found within the top-k retrieved chunks
  - hit_rate@k : 1.0 if any relevant source appears in top-k, else 0.0
  - mrr        : reciprocal rank of the first retrieved chunk whose source is relevant

`relevant_sources` (ground truth) is supplied per golden-set question;
`retrieved_sources` is the ordered list of source filenames of the retrieved chunks.
"""
from typing import Iterable, Sequence


def _as_set(items: Iterable[str]) -> set[str]:
    return set(items or [])


def recall_at_k(
    relevant_sources: Iterable[str], retrieved_sources: Sequence[str], k: int
) -> float:
    """Fraction of relevant sources present among the first k retrieved chunks."""
    relevant = _as_set(relevant_sources)
    if not relevant:
        return 0.0
    top = set(retrieved_sources[:k])
    return len(relevant & top) / len(relevant)


def hit_rate(
    relevant_sources: Iterable[str], retrieved_sources: Sequence[str], k: int
) -> float:
    """1.0 if at least one relevant source appears in the top-k, else 0.0."""
    relevant = _as_set(relevant_sources)
    if not relevant:
        return 0.0
    return 1.0 if (relevant & set(retrieved_sources[:k])) else 0.0


def mrr(relevant_sources: Iterable[str], retrieved_sources: Sequence[str]) -> float:
    """Reciprocal rank of the first retrieved chunk whose source is relevant."""
    relevant = _as_set(relevant_sources)
    if not relevant:
        return 0.0
    for i, src in enumerate(retrieved_sources, 1):
        if src in relevant:
            return 1.0 / i
    return 0.0


def retrieval_metrics(
    relevant_sources: Iterable[str],
    retrieved_sources: Sequence[str],
    ks: Sequence[int] | None = None,
) -> dict:
    """Compute the full metric bundle for one question (empty dict if no ground truth)."""
    if ks is None:
        ks = [1, 3, 5, 10]
    relevant = _as_set(relevant_sources)
    if not relevant:
        return {}
    out: dict = {}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(relevant, retrieved_sources, k), 3)
    out["mrr"] = round(mrr(relevant, retrieved_sources), 3)
    out["hit_rate@5"] = round(hit_rate(relevant, retrieved_sources, 5), 3)
    return out
