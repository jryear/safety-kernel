"""Deterministic NumPy failure-manifold critic.

This deliberately starts with a small auditable logistic model rather than a
large neural net. The MVP question is safety metric shape under erosion and
conformal thresholds, not representation scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CriticFitStats:
    n_samples: int
    n_features: int
    positive_rate: float
    final_loss: float


class FailureManifoldCritic:
    """Logistic failure critic over rollout state-action features."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.15,
        l2: float = 1e-3,
        epochs: int = 500,
    ):
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.fit_stats_: CriticFitStats | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> CriticFitStats:
        X = _as_2d(X)
        y = _as_labels(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("cannot fit critic on zero samples")

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        Xn = (X - self.mean_) / self.scale_

        w = np.zeros(Xn.shape[1], dtype=np.float64)
        b = _logit(np.clip(y.mean(), 1e-3, 1.0 - 1e-3))

        for _ in range(self.epochs):
            z = Xn @ w + b
            p = _sigmoid(z)
            err = p - y
            grad_w = (Xn.T @ err) / Xn.shape[0] + self.l2 * w
            grad_b = float(err.mean())
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        self.weights_ = w
        self.bias_ = float(b)
        final_scores = self.predict_proba(X)
        final_loss = _binary_cross_entropy(final_scores, y)
        stats = CriticFitStats(
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            positive_rate=float(y.mean()),
            final_loss=float(final_loss),
        )
        self.fit_stats_ = stats
        return stats

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        self._require_fit()
        Xn = (X - self.mean_) / self.scale_
        return _sigmoid(Xn @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return self.predict_proba(X) >= float(threshold)

    def _require_fit(self) -> None:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("critic is not fit")


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {X.shape}")
    return X


def _as_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("labels must be binary 0/1")
    return y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _binary_cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())
