"""Ranking and classification metrics for implicit feedback evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from model.ncf import NCF


@dataclass
class MetricResult:
    ndcg: float
    precision: float
    recall: float
    map_score: float
    f1: float
    num_users: int

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        key = lambda name: f"{prefix}{name}" if prefix else name
        return {
            key("ndcg"): self.ndcg,
            key("precision"): self.precision,
            key("recall"): self.recall,
            key("map"): self.map_score,
            key("f1"): self.f1,
        }


def _dcg(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    positions = np.arange(1, relevances.size + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg_at_k(relevant: set[int], ranked_items: list[int], k: int) -> float:
    top_k = ranked_items[:k]
    relevances = np.array([1 if item in relevant else 0 for item in top_k], dtype=float)
    ideal = np.sort(relevances)[::-1]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(relevances) / ideal_dcg


def precision_at_k(relevant: set[int], ranked_items: list[int], k: int) -> float:
    top_k = ranked_items[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(relevant: set[int], ranked_items: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision(relevant: set[int], ranked_items: list[int]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(ranked_items, start=1):
        if item in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant)


def f1_at_k(relevant: set[int], ranked_items: list[int], k: int) -> float:
    """F1@k for top-k recommendations (implicit feedback is highly imbalanced)."""
    precision = precision_at_k(relevant, ranked_items, k)
    recall = recall_at_k(relevant, ranked_items, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate_model(
    model: NCF,
    eval_dict: dict[int, set[int]],
    train_items_by_user: dict[int, set[int]],
    num_items: int,
    device: torch.device,
    k: int = 10,
    max_users: int | None = None,
) -> MetricResult:
    model.eval()
    ndcg_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    map_scores: list[float] = []
    f1_scores: list[float] = []

    users = list(eval_dict.keys())
    if max_users is not None:
        users = users[:max_users]

    all_items = torch.arange(num_items, device=device)

    for user_idx in users:
        relevant = eval_dict[user_idx]
        if not relevant:
            continue

        seen = train_items_by_user.get(user_idx, set())
        candidate_items = [i for i in range(num_items) if i not in seen]
        if not candidate_items:
            continue

        user_tensor = torch.full((len(candidate_items),), user_idx, dtype=torch.long, device=device)
        item_tensor = torch.tensor(candidate_items, dtype=torch.long, device=device)
        scores = model.predict(user_tensor, item_tensor).cpu().numpy()

        ranked_indices = np.argsort(-scores)
        ranked_items = [candidate_items[i] for i in ranked_indices]

        ndcg_scores.append(ndcg_at_k(relevant, ranked_items, k))
        precision_scores.append(precision_at_k(relevant, ranked_items, k))
        recall_scores.append(recall_at_k(relevant, ranked_items, k))
        map_scores.append(average_precision(relevant, ranked_items))
        f1_scores.append(f1_at_k(relevant, ranked_items, k))

    if not ndcg_scores:
        return MetricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    return MetricResult(
        ndcg=float(np.mean(ndcg_scores)),
        precision=float(np.mean(precision_scores)),
        recall=float(np.mean(recall_scores)),
        map_score=float(np.mean(map_scores)),
        f1=float(np.mean(f1_scores)),
        num_users=len(ndcg_scores),
    )
