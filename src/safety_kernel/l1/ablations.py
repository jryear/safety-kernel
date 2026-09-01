"""Ablation runner for erosion depth and conformal coverage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math

from safety_kernel.l1.calibration import calibrate_threshold
from safety_kernel.l1.critic import FailureManifoldCritic
from safety_kernel.l1.metrics import compute_binary_metrics
from safety_kernel.l1.rollouts import (
    DeterministicPPOBaseline,
    LegacyVelocitySaturatingBaseline,
    RolloutDataset,
    collect_rollouts,
    default_shift_schedule,
)


ACTION_HALF_RANGE_RAD_S = math.pi


@dataclass(frozen=True)
class AblationConfig:
    erosion_scales: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5)
    coverages: tuple[float, ...] = (0.8, 0.9, 0.95)
    train_episodes: int = 8
    calibration_episodes: int = 12
    test_episodes: int = 8
    max_steps: int = 220
    seed: int = 7
    contract_path: str | Path | None = None
    use_shift_schedule: bool = True
    intervention_tolerance: float = 0.05
    boundary_margin: float = 0.0
    calibration_unit: str = "thinned"
    calibration_stride: int = 10
    min_unsafe_floor: int = 30
    policy_kind: str = "boundary"


@dataclass(frozen=True)
class AblationResult:
    metrics_rows: list[dict]
    diagnostic_rows: list[dict]


def run_erosion_conformal_ablation_with_diagnostics(config: AblationConfig) -> AblationResult:
    """Run the MVP safety metric grid."""
    rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    shifts = default_shift_schedule() if config.use_shift_schedule else []

    for erosion_scale in config.erosion_scales:
        train_policy = _make_policy(config.policy_kind, seed=config.seed)
        calibration_policy = _make_policy(config.policy_kind, seed=config.seed + 1000)
        test_policy = _make_policy(config.policy_kind, seed=config.seed + 2000)
        train = collect_rollouts(
            n_episodes=config.train_episodes,
            max_steps=config.max_steps,
            seed=config.seed,
            erosion_scale=erosion_scale,
            contract_path=config.contract_path,
            shift_schedule=shifts,
            intervention_tolerance=config.intervention_tolerance,
            boundary_margin=config.boundary_margin,
            policy=train_policy,
        )
        calibration = collect_rollouts(
            n_episodes=config.calibration_episodes,
            max_steps=config.max_steps,
            seed=config.seed + 1000,
            erosion_scale=erosion_scale,
            contract_path=config.contract_path,
            shift_schedule=shifts,
            intervention_tolerance=config.intervention_tolerance,
            boundary_margin=config.boundary_margin,
            policy=calibration_policy,
        )
        test = collect_rollouts(
            n_episodes=config.test_episodes,
            max_steps=config.max_steps,
            seed=config.seed + 2000,
            erosion_scale=erosion_scale,
            contract_path=config.contract_path,
            shift_schedule=shifts,
            intervention_tolerance=config.intervention_tolerance,
            boundary_margin=config.boundary_margin,
            policy=test_policy,
        )

        diagnostic_rows.extend([
            _diagnostic_row(
                split="train",
                erosion_scale=erosion_scale,
                data=train,
                config=config,
            ),
            _diagnostic_row(
                split="calibration",
                erosion_scale=erosion_scale,
                data=calibration,
                config=config,
            ),
            _diagnostic_row(
                split="test",
                erosion_scale=erosion_scale,
                data=test,
                config=config,
            ),
        ])

        critic = FailureManifoldCritic()
        fit_stats = critic.fit(train.X, train.y)
        calibration_scores = critic.predict_proba(calibration.X)
        test_scores = critic.predict_proba(test.X)
        cal_scores_used, cal_labels_used, cal_groups_used = _calibration_view(
            scores=calibration_scores,
            labels=calibration.y,
            episode_ids=calibration.episode_ids,
            unit=config.calibration_unit,
            stride=config.calibration_stride,
        )

        for coverage in config.coverages:
            conformal = calibrate_threshold(
                cal_scores_used,
                cal_labels_used,
                coverage=coverage,
                group_ids=cal_groups_used,
            )
            metrics = compute_binary_metrics(
                scores=test_scores,
                labels=test.y,
                threshold=conformal.threshold,
                coverage=coverage,
                sla_hits=test.sla_hits,
                sla_total=test.sla_total,
                intervention_deltas=test.intervention_deltas,
                min_unsafe_floor=config.min_unsafe_floor,
            )
            row = {
                "erosion_scale": float(erosion_scale),
                "coverage": float(coverage),
                "threshold": float(conformal.threshold),
                "calibration_unit": conformal.unit,
                "requested_calibration_unit": config.calibration_unit,
                "calibration_stride": int(config.calibration_stride),
                "effective_calibration_unsafe_units": int(conformal.effective_n_unsafe),
                "intervention_tolerance": float(config.intervention_tolerance),
                "boundary_margin": float(config.boundary_margin),
                "policy_kind": config.policy_kind,
                "train_samples": int(train.n_steps),
                "calibration_samples": int(calibration.n_steps),
                "test_samples": int(test.n_steps),
                "train_positive_rate": float(fit_stats.positive_rate),
                "calibration_positive_rate": float(calibration.y.mean()) if calibration.y.size else 0.0,
                "test_positive_rate": float(test.y.mean()) if test.y.size else 0.0,
                "train_loss": float(fit_stats.final_loss),
                **_constraint_rates("test", test.constraint_counts, test.n_steps),
                **metrics.to_dict(),
            }
            rows.append(row)
    return AblationResult(metrics_rows=rows, diagnostic_rows=diagnostic_rows)


def run_erosion_conformal_ablation(config: AblationConfig) -> list[dict]:
    """Compatibility wrapper returning only thresholded metric rows."""
    return run_erosion_conformal_ablation_with_diagnostics(config).metrics_rows


def coerce_float_tuple(values: Iterable[str | float]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _constraint_rates(prefix: str, counts: dict[str, int], n_steps: int) -> dict[str, float | int]:
    names = [
        "velocity_clamp",
        "position_lookahead",
        "energy_cap",
        "self_collision",
    ]
    out: dict[str, float | int] = {}
    for name in names:
        count = int(counts.get(name, 0))
        out[f"{prefix}_{name}_count"] = count
        out[f"{prefix}_{name}_rate"] = float(count / n_steps) if n_steps else 0.0
    return out


def _diagnostic_row(
    *,
    split: str,
    erosion_scale: float,
    data: RolloutDataset,
    config: AblationConfig,
) -> dict:
    finite_margins = data.margins[data.margins == data.margins]
    row = {
        "split": split,
        "erosion_scale": float(erosion_scale),
        "samples": int(data.n_steps),
        "episodes": int(data.n_episodes),
        "positive_rate": float(data.y.mean()) if data.y.size else 0.0,
        "intervention_gt_tol_rate": _array_rate(data.intervention_labels),
        "boundary_margin_rate": _array_rate(data.boundary_labels),
        "near_miss_rate": _array_rate(data.near_miss_labels),
        "intervention_tolerance_rad_s": float(config.intervention_tolerance),
        "intervention_tolerance_frac_action_half_range": float(
            config.intervention_tolerance / ACTION_HALF_RANGE_RAD_S
        ),
        "boundary_margin": float(config.boundary_margin),
        "policy_kind": config.policy_kind,
        "intervention_delta_p50": _quantile(data.intervention_deltas, 0.50),
        "intervention_delta_p90": _quantile(data.intervention_deltas, 0.90),
        "intervention_delta_p95": _quantile(data.intervention_deltas, 0.95),
        "intervention_delta_p99": _quantile(data.intervention_deltas, 0.99),
        "intervention_delta_max": float(data.intervention_deltas.max())
        if data.intervention_deltas.size
        else 0.0,
        "margin_min_p01": _quantile(finite_margins, 0.01),
        "margin_min_p05": _quantile(finite_margins, 0.05),
        "margin_min_p50": _quantile(finite_margins, 0.50),
    }
    row.update(_constraint_rates("constraint", data.constraint_counts, data.n_steps))
    return row


def _array_rate(values) -> float:
    if len(values) == 0:
        return 0.0
    return float(sum(values) / len(values))


def _quantile(values, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    import numpy as np

    return float(np.quantile(values, q))


def _calibration_view(
    *,
    scores,
    labels,
    episode_ids,
    unit: str,
    stride: int,
):
    if unit == "sample":
        return scores, labels, None
    if unit == "episode_block":
        return scores, labels, episode_ids
    if unit == "thinned":
        stride = max(int(stride), 1)
        idx = list(range(0, len(labels), stride))
        return scores[idx], labels[idx], None
    raise ValueError(f"unknown calibration_unit: {unit!r}")


def _make_policy(policy_kind: str, *, seed: int):
    if policy_kind == "boundary":
        return DeterministicPPOBaseline(seed=seed)
    if policy_kind == "legacy_velocity":
        return LegacyVelocitySaturatingBaseline(seed=seed)
    raise ValueError(f"unknown policy_kind: {policy_kind!r}")
