"""Safety metric aggregation for L1 ablations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BinarySafetyMetrics:
    threshold: float
    coverage: float
    tp: int
    fp: int
    tn: int
    fn: int
    n_unsafe_test: int
    n_safe_test: int
    false_positive_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    empirical_unsafe_coverage: float
    empirical_unsafe_coverage_ci_low: float
    empirical_unsafe_coverage_ci_high: float
    coverage_gap: float
    roc_auc: float
    pr_auc: float
    recall_intervention_delta_ge_005: float
    recall_intervention_delta_ge_010: float
    recall_intervention_delta_ge_025: float
    min_unsafe_floor: int
    metric_cell_valid: bool
    reported_false_negative_rate: float
    reported_empirical_unsafe_coverage: float
    reported_coverage_gap: float
    reported_pr_auc: float
    intervention_rate: float
    unsafe_rate: float
    sla_hit_rate: float
    sla_hits: int
    sla_total: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_binary_metrics(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    coverage: float,
    sla_hits: int = 0,
    sla_total: int = 0,
    intervention_deltas: np.ndarray | None = None,
    min_unsafe_floor: int = 30,
) -> BinarySafetyMetrics:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have same shape")
    pred = scores >= float(threshold)
    truth = labels == 1
    if intervention_deltas is None:
        intervention_deltas = np.zeros_like(scores)
    intervention_deltas = np.asarray(intervention_deltas, dtype=np.float64).reshape(-1)
    if intervention_deltas.shape != scores.shape:
        raise ValueError("intervention_deltas must have same shape as scores")

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth))
    fn = int(np.sum(~pred & truth))
    n_unsafe = int(np.sum(truth))
    n_safe = int(np.sum(~truth))
    empirical_coverage = _rate(tp, tp + fn)
    ci_low, ci_high = wilson_interval(tp, tp + fn)
    cell_valid = n_unsafe >= int(min_unsafe_floor)
    fnr = _rate(fn, fn + tp)
    gap = float(coverage - empirical_coverage)
    pr = pr_auc(scores, labels)
    return BinarySafetyMetrics(
        threshold=float(threshold),
        coverage=float(coverage),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        n_unsafe_test=n_unsafe,
        n_safe_test=n_safe,
        false_positive_rate=_rate(fp, fp + tn),
        false_negative_rate=fnr,
        precision=_rate(tp, tp + fp),
        recall=_rate(tp, tp + fn),
        empirical_unsafe_coverage=empirical_coverage,
        empirical_unsafe_coverage_ci_low=ci_low,
        empirical_unsafe_coverage_ci_high=ci_high,
        coverage_gap=gap,
        roc_auc=roc_auc(scores, labels),
        pr_auc=pr,
        recall_intervention_delta_ge_005=_severity_recall(
            pred, truth, intervention_deltas, 0.05
        ),
        recall_intervention_delta_ge_010=_severity_recall(
            pred, truth, intervention_deltas, 0.10
        ),
        recall_intervention_delta_ge_025=_severity_recall(
            pred, truth, intervention_deltas, 0.25
        ),
        min_unsafe_floor=int(min_unsafe_floor),
        metric_cell_valid=cell_valid,
        reported_false_negative_rate=fnr if cell_valid else float("nan"),
        reported_empirical_unsafe_coverage=(
            empirical_coverage if cell_valid else float("nan")
        ),
        reported_coverage_gap=gap if cell_valid else float("nan"),
        reported_pr_auc=pr if cell_valid else float("nan"),
        intervention_rate=float(np.mean(pred)) if pred.size else 0.0,
        unsafe_rate=float(np.mean(truth)) if truth.size else 0.0,
        sla_hit_rate=_rate(sla_hits, sla_total),
        sla_hits=int(sla_hits),
        sla_total=int(sla_total),
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = (
        z
        * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
        / denom
    )
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC by pairwise ranking.

    Returns NaN when only one class is present.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return float(wins / (pos.size * neg.size))


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Stepwise area under the precision-recall curve."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)

    # Include origin point and integrate precision as a step function over recall.
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    delta_recall = recall - recall_prev
    return float(np.sum(precision * delta_recall))


def _severity_recall(
    pred: np.ndarray,
    truth: np.ndarray,
    intervention_deltas: np.ndarray,
    threshold: float,
) -> float:
    mask = truth & (intervention_deltas >= threshold)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(pred[mask]))
