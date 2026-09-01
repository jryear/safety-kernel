"""Deterministic rollout collection for L1 boundary learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from collections import Counter

import numpy as np

from safety_kernel.contract.shift_injector import ShiftSpec
from safety_kernel.env.cell_arm import CellArmEnv
from safety_kernel.l1.labels import feature_vector, label_action_with_l0r


@dataclass(frozen=True)
class RolloutDataset:
    X: np.ndarray
    y: np.ndarray
    margins: np.ndarray
    intervention_labels: np.ndarray
    intervention_deltas: np.ndarray
    boundary_labels: np.ndarray
    near_miss_labels: np.ndarray
    episode_ids: np.ndarray
    constraint_counts: dict[str, int]
    sla_hits: int
    sla_total: int
    n_steps: int
    n_episodes: int


class DeterministicPPOBaseline:
    """Deterministic actor used as the PPO-baseline adapter.

    The class mirrors the inference-only surface of a trained PPO actor:
    ``begin_episode`` fixes deterministic latent state and ``act`` maps an
    observation to a clipped action. A trained PPO export can replace this
    class without changing the L1 data/calibration pipeline.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        gain: float = 1.2,
        damping: float = 0.25,
        target_span: float = 1.5,
        action_limit_scale: float = 0.85,
    ):
        self.seed = int(seed)
        self.gain = float(gain)
        self.damping = float(damping)
        self.target_span = float(target_span)
        self.action_limit_scale = float(action_limit_scale)
        self._target_q = np.zeros(5, dtype=np.float64)
        self._phase = 0.0

    def begin_episode(self, episode: int) -> None:
        rng = np.random.default_rng(self.seed + int(episode) * 7919)
        self._target_q = rng.uniform(-self.target_span, self.target_span, size=5)
        self._phase = float(rng.uniform(0.0, 2.0 * np.pi))

    def act(self, obs: np.ndarray, step: int, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
        q = obs[:5]
        qdot = obs[5:10]
        t = float(step)
        probe = 0.15 * np.array(
            [
                np.sin(0.031 * t + self._phase),
                np.cos(0.047 * t + self._phase),
                np.sin(0.053 * t + 0.5 * self._phase),
                np.cos(0.029 * t + 0.25 * self._phase),
                np.sin(0.041 * t + 0.75 * self._phase),
            ],
            dtype=np.float64,
        )
        action = self.gain * (self._target_q - q) - self.damping * qdot + probe
        return np.clip(
            action,
            action_low * self.action_limit_scale,
            action_high * self.action_limit_scale,
        )


class LegacyVelocitySaturatingBaseline(DeterministicPPOBaseline):
    """Original actor shape kept for attribution runs.

    This actor clips at the physical action box and tends to activate tiny
    eroded velocity clamps. It is useful for separating label fixes from actor
    distribution changes.
    """

    def __init__(self, *, seed: int = 0):
        super().__init__(
            seed=seed,
            gain=1.35,
            damping=0.28,
            target_span=1.1,
            action_limit_scale=1.0,
        )

    def act(
        self,
        obs: np.ndarray,
        step: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> np.ndarray:
        q = obs[:5]
        qdot = obs[5:10]
        t = float(step)
        probe = 0.22 * np.array(
            [
                np.sin(0.031 * t + self._phase),
                np.cos(0.047 * t + self._phase),
                np.sin(0.053 * t + 0.5 * self._phase),
                np.cos(0.029 * t + 0.25 * self._phase),
                np.sin(0.041 * t + 0.75 * self._phase),
            ],
            dtype=np.float64,
        )
        action = self.gain * (self._target_q - q) - self.damping * qdot + probe
        return np.clip(action, action_low, action_high)


def collect_rollouts(
    *,
    n_episodes: int,
    max_steps: int,
    seed: int,
    erosion_scale: float = 1.0,
    contract_path: str | Path | None = None,
    shift_schedule: Iterable[ShiftSpec] | None = None,
    policy: DeterministicPPOBaseline | None = None,
    intervention_tolerance: float = 0.05,
    boundary_margin: float = 0.0,
) -> RolloutDataset:
    """Collect deterministic state-action samples and hard-oracle labels."""
    if erosion_scale <= 0.0:
        raise ValueError("erosion_scale must be positive")

    base_eps_q = 0.001
    base_eps_qdot = 0.01
    base_d_safe = 0.02
    env = CellArmEnv(
        seed=seed,
        max_steps=max_steps,
        eps_q=base_eps_q * erosion_scale,
        eps_qdot=base_eps_qdot * erosion_scale,
        d_safe=base_d_safe * erosion_scale,
        contract_path=contract_path,
        shift_schedule=list(shift_schedule) if shift_schedule else None,
    )
    actor = policy or DeterministicPPOBaseline(seed=seed)

    features: list[np.ndarray] = []
    labels: list[int] = []
    margins: list[float] = []
    intervention_labels: list[int] = []
    intervention_deltas: list[float] = []
    boundary_labels: list[int] = []
    near_miss_labels: list[int] = []
    episode_ids: list[int] = []
    constraint_counts: Counter[str] = Counter()
    sla_hits = 0
    sla_total = 0

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        actor.begin_episode(episode)
        episode_events: list[dict] = []
        for step in range(max_steps):
            action = actor.act(obs, step, env.action_space.low, env.action_space.high)
            label = label_action_with_l0r(
                env=env,
                action=action,
                intervention_tolerance=intervention_tolerance,
                boundary_margin=boundary_margin,
            )
            features.append(feature_vector(obs, action))
            labels.append(1 if label.unsafe else 0)
            margins.append(label.margin_min)
            intervention_labels.append(1 if label.intervention else 0)
            intervention_deltas.append(label.intervention_delta)
            boundary_labels.append(
                1
                if np.isfinite(label.margin_min)
                and label.margin_min <= float(boundary_margin)
                else 0
            )
            near_miss_labels.append(1 if label.rta_nearmiss else 0)
            episode_ids.append(episode)
            for constraint in label.constraints_active:
                constraint_counts[constraint] += 1

            obs, _, terminated, truncated, info = env.step(action)
            episode_events.extend(info.get("contract_events", []))
            if terminated or truncated:
                break

        hits, total = _sla_hits_for_episode(
            shift_schedule=list(shift_schedule) if shift_schedule else [],
            events=episode_events,
            horizon_steps=max_steps,
        )
        sla_hits += hits
        sla_total += total

    X = np.vstack(features) if features else np.zeros((0, 20), dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return RolloutDataset(
        X=X,
        y=y,
        margins=np.asarray(margins, dtype=np.float64),
        intervention_labels=np.asarray(intervention_labels, dtype=np.int64),
        intervention_deltas=np.asarray(intervention_deltas, dtype=np.float64),
        boundary_labels=np.asarray(boundary_labels, dtype=np.int64),
        near_miss_labels=np.asarray(near_miss_labels, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        constraint_counts=dict(constraint_counts),
        sla_hits=int(sla_hits),
        sla_total=int(sla_total),
        n_steps=int(y.size),
        n_episodes=int(n_episodes),
    )


def default_shift_schedule() -> list[ShiftSpec]:
    """A small deterministic SLA probe schedule for Track A."""
    return [
        ShiftSpec(
            start_step=60,
            end_step=125,
            shift_type="sensor_stale",
            params={"frame_age_ms": 500.0},
        ),
        ShiftSpec(
            start_step=180,
            end_step=240,
            shift_type="anchor_drift",
            params={"anchor": "fiducial_fixture", "drift_xyz": [0.05, 0.0, 0.0]},
        ),
    ]


def _sla_hits_for_episode(
    *,
    shift_schedule: list[ShiftSpec],
    events: list[dict],
    horizon_steps: int,
) -> tuple[int, int]:
    """Count whether each injected fault generated any state transition.

    Existing unit tests validate exact per-monitor SLA windows. The ablation
    metric keeps the paper plot compact: fault window detected or not.
    """
    if not shift_schedule:
        return 0, 0
    hits = 0
    injected = [shift for shift in shift_schedule if shift.start_step < horizon_steps]
    for shift in injected:
        latest_detection_step = shift.start_step + _expected_detection_ticks(shift)
        found = any(
            int(event.get("step", -1)) >= shift.start_step
            and int(event.get("step", -1)) <= latest_detection_step
            for event in events
        )
        hits += 1 if found else 0
    return hits, len(injected)


def _expected_detection_ticks(shift: ShiftSpec) -> int:
    """Expected shift detection budget in ticks for the default Track A probes."""
    if shift.shift_type == "sensor_stale":
        return 27  # front_cam_health: 25 debounce ticks + slack from Gate 1
    if shift.shift_type in {"anchor_drift", "base_drift"}:
        return 15  # 13 debounce ticks + slack from Gate 1
    if shift.shift_type == "safety_io_flip":
        return 2
    return 0
