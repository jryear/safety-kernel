"""L0-R sequential projection: velocity clamp → position lookahead → energy cap → self-collision.

Implements Section 2.1.1 of the cumulative spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from safety_kernel.l0r.params import L0RParams
from safety_kernel.l0r.capsule import compute_collision_distances_at_q
from safety_kernel.l0r.errors import L0RSafetyInvariantError


def l0r_project(
    qdot_cmd: np.ndarray,
    q: np.ndarray,
    M: np.ndarray,
    model,
    data,
    params: L0RParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply L0-R sequential safety projection.

    Args:
        qdot_cmd: (n_joints,) proposed velocity command from L2.
        q: (n_joints,) current joint positions (observed).
        M: (n_joints, n_joints) mass matrix at current q.
        model: mujoco.MjModel — for FK in self-collision check.
        data: mujoco.MjData — for FK in self-collision check.
        params: L0RParams with physical limits, erosion, collision config.

    Returns:
        (qdot_exec, info) where info is a dict matching the
        safety_intervention NDJSON schema fields.
    """
    n = params.n_joints
    if qdot_cmd.shape != (n,):
        raise ValueError(f"qdot_cmd must have shape ({n},), got {qdot_cmd.shape}")
    if q.shape != (n,):
        raise ValueError(f"q must have shape ({n},), got {q.shape}")
    if M.shape != (n, n):
        raise ValueError(f"M must have shape ({n}, {n}), got {M.shape}")
    if not (
        np.all(np.isfinite(qdot_cmd))
        and np.all(np.isfinite(q))
        and np.all(np.isfinite(M))
    ):
        raise ValueError("L0-R inputs must be finite")

    qdot = qdot_cmd.copy()
    constraints_active: list[str] = []
    rta_nearmiss = False
    energy_scaling_factor = 1.0
    collision_info: list[dict] = []

    # Eroded limits (precomputed from params)
    qdot_max_e = params.qdot_max_eroded
    q_min_e = params.q_min_eroded
    q_max_e = params.q_max_eroded

    # ── Step 1: Velocity box clamp ──────────────────────────────────
    qdot_prev = qdot.copy()
    qdot = np.clip(qdot, -qdot_max_e, qdot_max_e)
    if not np.array_equal(qdot, qdot_prev):
        constraints_active.append("velocity_clamp")

    # ── Step 2: Position lookahead clamp ────────────────────────────
    qdot_prev = qdot.copy()
    dt = params.dt

    for i in range(n):
        q_i = q[i]
        # Feasible velocity interval from position limits
        v_lo = (q_min_e[i] - q_i) / dt
        v_hi = (q_max_e[i] - q_i) / dt

        # Detect if q_i is outside eroded bounds (flag RTA near-miss)
        if q_i > q_max_e[i] or q_i < q_min_e[i]:
            rta_nearmiss = True

        if v_lo > v_hi:
            # Both bounds violated simultaneously (shouldn't happen with
            # reasonable erosion, but handle defensively).
            midpoint = (q_min_e[i] + q_max_e[i]) / 2.0
            if q_i < midpoint:
                qdot[i] = qdot_max_e[i]  # max recovery positive
            else:
                qdot[i] = -qdot_max_e[i]  # max recovery negative
        else:
            qdot[i] = np.clip(qdot[i], v_lo, v_hi)

    # Re-clamp to eroded velocity limits (position clamp can widen)
    qdot = np.clip(qdot, -qdot_max_e, qdot_max_e)

    if not np.array_equal(qdot, qdot_prev):
        constraints_active.append("position_lookahead")

    # ── Step 3: Kinetic energy cap ──────────────────────────────────
    qdot_prev = qdot.copy()
    e_k = 0.5 * qdot @ M @ qdot

    if e_k > params.e_max_safe:
        scale = np.sqrt(params.e_max_safe / e_k)
        qdot *= scale
        energy_scaling_factor = scale
        constraints_active.append("energy_cap")
    else:
        energy_scaling_factor = 1.0

    energy_ratio = e_k / params.e_max if params.e_max > 0 else 0.0

    # ── Step 4: Self-collision check ────────────────────────────────
    q_next = q + qdot * dt
    collision_distances = compute_collision_distances_at_q(
        model, data, q_next, params.collision_pairs
    )

    collision_triggered = False
    for idx, dist in enumerate(collision_distances):
        pair_name = params.collision_pair_names[idx]
        collision_info.append({
            "pair_id": pair_name,
            "distance": float(dist),
        })
        if dist < params.d_safe:
            collision_triggered = True

    if collision_triggered:
        qdot[:] = 0.0
        constraints_active.append("self_collision")
        # Telemetry must describe the final output being checked. The first
        # distance set explains why the candidate was suppressed; after the
        # suppression, strict H3 evaluates q_next for the zero command.
        final_distances = compute_collision_distances_at_q(
            model, data, q, params.collision_pairs
        )
        collision_info = [
            {"pair_id": params.collision_pair_names[idx], "distance": float(dist)}
            for idx, dist in enumerate(final_distances)
        ]

    # ── Runtime verification: enforce all output invariants ─────────
    # This must execute even under ``python -O``; assertions are removed by
    # the optimizer and cannot be the safety boundary.
    _verify_output_constraints(qdot, q, M, model, data, params)

    # ── Build intervention info ─────────────────────────────────────
    # Compute margins for telemetry
    margins = {}
    margins["velocity"] = float(np.min(qdot_max_e - np.abs(qdot)))
    q_next_final = q + qdot * dt
    margins["position_upper"] = float(np.min(q_max_e - q_next_final))
    margins["position_lower"] = float(np.min(q_next_final - q_min_e))
    e_k_final = 0.5 * qdot @ M @ qdot
    margins["energy"] = float(params.e_max_safe - e_k_final)
    if collision_info:
        margins["self_collision"] = float(min(c["distance"] for c in collision_info))

    info = {
        "qdot_cmd": qdot_cmd.tolist(),
        "qdot_exec": qdot.tolist(),
        "constraints_active": constraints_active,
        "margin_to_L0R": margins,
        "energy_ratio": float(energy_ratio),
        "energy_scaling_factor": float(energy_scaling_factor),
        "collision_pairs": collision_info,
        "rta_nearmiss": rta_nearmiss,
    }

    return qdot, info


def _verify_output_constraints(
    qdot: np.ndarray,
    q: np.ndarray,
    M: np.ndarray,
    model,
    data,
    params: L0RParams,
) -> None:
    """Verify that the L0-R output satisfies every applicable constraint.

    When self-collision zeros velocity and q is already outside eroded
    position bounds, q_next = q still violates position — this is an
    inherent conflict (reported as both self_collision + rta_nearmiss).
    Position verification is skipped only when no finite one-tick recovery
    command exists because collision handling selected the safer zero command;
    the caller still receives the RTA near-miss signal.
    """
    EPS = 1e-8
    n = params.n_joints
    dt = params.dt
    qdot_max_e = params.qdot_max_eroded
    q_min_e = params.q_min_eroded
    q_max_e = params.q_max_eroded
    is_zero = not np.any(np.abs(qdot) > EPS)

    if not np.all(np.isfinite(qdot)):
        raise L0RSafetyInvariantError("L0-R output contains non-finite velocity")

    # 1. Velocity limits (always valid — zero satisfies)
    for i in range(n):
        if abs(qdot[i]) > qdot_max_e[i] + EPS:
            raise L0RSafetyInvariantError(
                f"Velocity violation j{i}: |{qdot[i]}| > {qdot_max_e[i]}"
            )

    # 2. Position lookahead
    # When q is already outside eroded bounds, max recovery velocity may
    # not be enough to return within one step — this is flagged as
    # rta_nearmiss. Skip position assertion for those joints.
    if not is_zero:
        q_next = q + qdot * dt
        for i in range(n):
            q_outside = (q[i] < q_min_e[i] - EPS) or (q[i] > q_max_e[i] + EPS)
            if q_outside:
                # Best-effort recovery: verify velocity moves toward safe region
                if q[i] < q_min_e[i]:
                    if qdot[i] < -EPS:
                        raise L0RSafetyInvariantError(
                            f"Recovery direction j{i}: should move positive but {qdot[i]}"
                        )
                else:
                    if qdot[i] > EPS:
                        raise L0RSafetyInvariantError(
                            f"Recovery direction j{i}: should move negative but {qdot[i]}"
                        )
            else:
                if q_next[i] < q_min_e[i] - EPS:
                    raise L0RSafetyInvariantError(
                        f"Position lower violation j{i}: {q_next[i]} < {q_min_e[i]}"
                    )
                if q_next[i] > q_max_e[i] + EPS:
                    raise L0RSafetyInvariantError(
                        f"Position upper violation j{i}: {q_next[i]} > {q_max_e[i]}"
                    )

    # 3. Energy cap (always valid — zero satisfies)
    e_k = 0.5 * qdot @ M @ qdot
    if e_k > params.e_max_safe + EPS:
        raise L0RSafetyInvariantError(
            f"Energy violation: {e_k} > {params.e_max_safe}"
        )

    # 4. Self-collision (only relevant when velocity is non-zero)
    if not is_zero:
        q_next = q + qdot * dt
        distances = compute_collision_distances_at_q(
            model, data, q_next, params.collision_pairs
        )
        for idx, dist in enumerate(distances):
            if dist < params.d_safe - EPS:
                raise L0RSafetyInvariantError(
                    f"Self-collision violation {params.collision_pair_names[idx]}: "
                    f"{dist} < {params.d_safe}"
                )
