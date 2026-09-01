"""L1 failure-manifold MVP tests."""

import numpy as np
import pytest

from safety_kernel.l1.ablations import (
    AblationConfig,
    run_erosion_conformal_ablation,
    run_erosion_conformal_ablation_with_diagnostics,
)
from safety_kernel.l1.calibration import calibrate_threshold
from safety_kernel.l1.critic import FailureManifoldCritic
from safety_kernel.l1.metrics import compute_binary_metrics
from safety_kernel.l1.rollouts import collect_rollouts


def test_critic_learns_separable_boundary():
    rng = np.random.default_rng(0)
    X_safe = rng.normal(loc=-1.0, scale=0.1, size=(32, 4))
    X_unsafe = rng.normal(loc=1.0, scale=0.1, size=(32, 4))
    X = np.vstack([X_safe, X_unsafe])
    y = np.array([0] * 32 + [1] * 32)

    critic = FailureManifoldCritic(epochs=250)
    stats = critic.fit(X, y)
    scores = critic.predict_proba(X)

    assert stats.n_samples == 64
    assert scores[:32].mean() < 0.25
    assert scores[32:].mean() > 0.75


def test_conformal_threshold_lowers_with_higher_coverage():
    scores = np.array([0.1, 0.2, 0.45, 0.6, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1, 1, 1])

    loose = calibrate_threshold(scores, labels, coverage=0.8)
    tight = calibrate_threshold(scores, labels, coverage=0.95)

    assert tight.threshold <= loose.threshold
    assert tight.n_unsafe == 4
    assert tight.effective_n_unsafe == 4


def test_collect_rollouts_produces_binary_oracle_labels():
    data = collect_rollouts(
        n_episodes=1,
        max_steps=24,
        seed=123,
        erosion_scale=1.0,
        contract_path=None,
        shift_schedule=None,
    )

    assert data.X.shape == (24, 20)
    assert data.y.shape == (24,)
    assert set(data.y.tolist()).issubset({0, 1})
    assert data.intervention_deltas.shape == (24,)
    assert data.episode_ids.shape == (24,)
    assert isinstance(data.constraint_counts, dict)
    assert data.n_steps == 24


def test_metrics_report_auc_and_empirical_coverage():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    metrics = compute_binary_metrics(
        scores=scores,
        labels=labels,
        threshold=0.5,
        coverage=0.9,
        intervention_deltas=np.array([0.0, 0.0, 0.06, 0.30]),
    )

    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.empirical_unsafe_coverage == 1.0
    assert metrics.coverage_gap == pytest.approx(-0.1)
    assert metrics.recall_intervention_delta_ge_005 == 1.0


def test_tiny_ablation_grid_runs():
    rows = run_erosion_conformal_ablation(
        AblationConfig(
            erosion_scales=(0.75,),
            coverages=(0.8, 0.95),
            train_episodes=1,
            calibration_episodes=1,
            test_episodes=1,
            max_steps=32,
            seed=5,
            contract_path=None,
            use_shift_schedule=False,
        )
    )

    assert len(rows) == 2
    for row in rows:
        assert row["erosion_scale"] == 0.75
        assert row["coverage"] in {0.8, 0.95}
        assert "roc_auc" in row
        assert "pr_auc" in row
        assert "empirical_unsafe_coverage" in row
        assert "test_self_collision_rate" in row
        assert 0.0 <= row["false_positive_rate"] <= 1.0
        assert 0.0 <= row["false_negative_rate"] <= 1.0
        assert 0.0 <= row["intervention_rate"] <= 1.0


def test_ablation_emits_component_diagnostics():
    result = run_erosion_conformal_ablation_with_diagnostics(
        AblationConfig(
            erosion_scales=(1.0,),
            coverages=(0.9,),
            train_episodes=1,
            calibration_episodes=1,
            test_episodes=1,
            max_steps=32,
            seed=8,
            contract_path=None,
            use_shift_schedule=False,
        )
    )

    assert len(result.metrics_rows) == 1
    assert len(result.diagnostic_rows) == 3
    row = result.diagnostic_rows[0]
    assert "intervention_gt_tol_rate" in row
    assert "boundary_margin_rate" in row
    assert "near_miss_rate" in row
    assert "constraint_self_collision_rate" in row
    assert "intervention_delta_p95" in row
