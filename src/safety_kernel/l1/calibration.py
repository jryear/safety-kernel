"""Conformal thresholding for L1 failure scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalThreshold:
    """Threshold chosen to cover unsafe calibration examples."""

    coverage: float
    threshold: float
    n_calibration: int
    n_unsafe: int
    unit: str
    effective_n_unsafe: int

    @property
    def target_false_negative_rate(self) -> float:
        return 1.0 - self.coverage


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    coverage: float,
    group_ids: np.ndarray | None = None,
) -> ConformalThreshold:
    """Choose a risk threshold from unsafe calibration scores.

    The critic predicts unsafe when score >= threshold. To target coverage of
    true unsafe samples, use the lower quantile of unsafe scores. Higher
    coverage lowers the threshold and typically increases false positives.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if not (0.0 < coverage < 1.0):
        raise ValueError("coverage must be in (0, 1)")

    unsafe_scores = _unsafe_calibration_scores(scores, labels, group_ids)
    if unsafe_scores.size == 0:
        # Fail closed: with no unsafe calibration evidence, flag everything.
        threshold = 0.0
    else:
        # Split-conformal finite-sample rank on unsafe scores. Sort descending
        # because higher score means higher risk; threshold at the k-th covered
        # unsafe unit. This is equivalent to the conformal quantile over
        # nonconformity score -risk_score.
        descending = np.sort(unsafe_scores)[::-1]
        rank = int(np.ceil((unsafe_scores.size + 1) * coverage))
        rank = min(max(rank, 1), unsafe_scores.size)
        threshold = float(descending[rank - 1])
    return ConformalThreshold(
        coverage=float(coverage),
        threshold=threshold,
        n_calibration=int(scores.size),
        n_unsafe=int(np.sum(labels == 1)),
        unit="episode_block" if group_ids is not None else "sample",
        effective_n_unsafe=int(unsafe_scores.size),
    )


def _unsafe_calibration_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray | None,
) -> np.ndarray:
    if group_ids is None:
        return np.asarray(scores[labels == 1], dtype=np.float64)

    group_ids = np.asarray(group_ids).reshape(-1)
    if group_ids.shape != scores.shape:
        raise ValueError("group_ids must have the same shape as scores")
    block_scores: list[float] = []
    for group in np.unique(group_ids):
        mask = group_ids == group
        unsafe = mask & (labels == 1)
        if not np.any(unsafe):
            continue
        # To cover every unsafe sample in a covered episode, the episode block
        # score is its weakest unsafe score.
        block_scores.append(float(np.min(scores[unsafe])))
    return np.asarray(block_scores, dtype=np.float64)
