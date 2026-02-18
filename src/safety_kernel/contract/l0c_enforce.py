"""L0-C enforcement: TCP speed cap (pre-L0-R).

Implements the DEGRADED-state TCP speed clamp from Section 4 / revocation_policy.
Runs before L0-R projection: qdot_cmd → l0c_enforce → l0r_project → qdot_exec.

Gate 1: TCP speed cap only (uniform scaling). No geometric keepout enforcement.
Deterministic, monotone (only shrinks commands), no QP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from safety_kernel.contract.state_machine import ContractState


def l0c_enforce(
    qdot_cmd: np.ndarray,
    J: np.ndarray,
    contract_state: ContractState,
    max_tcp_speed_mps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply L0-C TCP speed cap when in DEGRADED state.

    Args:
        qdot_cmd: (n_joints,) proposed velocity command from L2.
        J: (3, n_joints) TCP position Jacobian at current q.
        contract_state: Current contract state machine state.
        max_tcp_speed_mps: TCP speed limit from revocation_policy.on_degrade.caps.

    Returns:
        (qdot_l0c, info) where info contains enforcement details.
        qdot_l0c is always a copy (never mutates input).
    """
    qdot_l0c = qdot_cmd.copy()
    info: dict[str, Any] = {
        "l0c_active": False,
        "tcp_speed_before": 0.0,
        "tcp_speed_after": 0.0,
        "scaling_factor": 1.0,
        "constraints_active": [],
    }

    # Only enforce in DEGRADED state
    if contract_state != ContractState.DEGRADED:
        v_tcp = J @ qdot_cmd
        info["tcp_speed_before"] = float(np.linalg.norm(v_tcp))
        info["tcp_speed_after"] = info["tcp_speed_before"]
        return qdot_l0c, info

    # Compute TCP velocity
    v_tcp = J @ qdot_cmd
    speed = float(np.linalg.norm(v_tcp))
    info["tcp_speed_before"] = speed
    info["l0c_active"] = True

    if speed > max_tcp_speed_mps and speed > 0.0:
        scale = max_tcp_speed_mps / speed
        qdot_l0c *= scale
        info["scaling_factor"] = scale
        info["tcp_speed_after"] = max_tcp_speed_mps
        info["constraints_active"].append("tcp_speed_cap")
    else:
        info["tcp_speed_after"] = speed

    return qdot_l0c, info
