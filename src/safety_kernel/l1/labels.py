"""Failure labels derived from the existing safety stack.

Labels are intentionally oracle-derived from L0-R/L0-C evidence. L1 learns
the boundary; it does not define safety by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from safety_kernel.l0r.projection import l0r_project


@dataclass(frozen=True)
class ActionLabel:
    """A single state-action label from the hard safety oracle."""

    unsafe: bool
    intervention: bool
    intervention_delta: float
    constraints_active: tuple[str, ...]
    margin_min: float
    rta_nearmiss: bool


def feature_vector(obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Build the critic feature vector from proprioception and action."""
    obs = np.asarray(obs, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if obs.shape != (15,):
        raise ValueError(f"expected obs shape (15,), got {obs.shape}")
    if action.shape != (5,):
        raise ValueError(f"expected action shape (5,), got {action.shape}")
    return np.concatenate([obs, action])


def label_action_with_l0r(
    *,
    env,
    action: np.ndarray,
    intervention_tolerance: float = 0.05,
    boundary_margin: float = 0.0,
) -> ActionLabel:
    """Label the current state/action using L0-R projection evidence.

    A sample is unsafe when it is near the hard boundary in a physically
    meaningful way: L0-R changes the action by more than
    ``intervention_tolerance``, the post-projection margin is at or below
    ``boundary_margin``, or the state is an RTA near-miss. Tiny eroded-box
    touches are tracked as interventions but do not become positive labels by
    default.
    """
    qdot_cmd = np.asarray(action, dtype=np.float64).copy()
    q = env.data.qpos[: env.n_joints].copy()
    M = env._get_mass_matrix()
    qdot_exec, info = l0r_project(
        qdot_cmd=qdot_cmd,
        q=q,
        M=M,
        model=env.model,
        data=env.data,
        params=env.l0r_params,
    )
    constraints = tuple(str(c) for c in info.get("constraints_active", ()))
    intervention_delta = float(np.linalg.norm(qdot_exec - qdot_cmd, ord=np.inf))
    intervention = bool(intervention_delta > intervention_tolerance)
    rta_nearmiss = bool(info.get("rta_nearmiss", False))
    margins = info.get("margin_to_L0R", {})
    margin_min = _min_margin(margins)
    unsafe = intervention or rta_nearmiss or (
        np.isfinite(margin_min) and margin_min <= float(boundary_margin)
    )
    return ActionLabel(
        unsafe=unsafe,
        intervention=intervention,
        intervention_delta=intervention_delta,
        constraints_active=constraints,
        margin_min=margin_min,
        rta_nearmiss=rta_nearmiss,
    )


def _min_margin(margins: dict[str, Any]) -> float:
    values: list[float] = []
    for value in margins.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return float("nan")
    return float(min(values))
