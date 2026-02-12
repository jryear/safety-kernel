"""L0-R parameters dataclass — loaded from MJCF model attributes + sensor spec."""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class L0RParams:
    """Immutable L0-R projection parameters for CellArm 5-DOF.

    All limits are *physical* limits from the MJCF model.
    Erosion is computed internally from eps_q, eps_qdot, and dt.
    """

    n_joints: int

    # Physical limits (from MJCF)
    q_min: np.ndarray        # (n_joints,) rad
    q_max: np.ndarray        # (n_joints,) rad
    qdot_max: np.ndarray     # (n_joints,) rad/s — symmetric ±

    # Energy
    e_max: float             # Joules — tunable kernel budget
    energy_margin: float     # multiplicative, e.g. 0.98

    # Sensor erosion
    eps_q: float             # rad — joint position uncertainty
    eps_qdot: float          # rad/s — joint velocity uncertainty

    # Self-collision
    collision_pairs: List[Tuple[int, int]]  # (geom_id_a, geom_id_b)
    collision_pair_names: List[str]         # human-readable, e.g. "link_0-link_3"
    d_safe: float            # m — surface-to-surface clearance margin

    # Timing
    dt: float                # seconds — control timestep

    @property
    def e_max_safe(self) -> float:
        return self.e_max * self.energy_margin

    @property
    def qdot_max_eroded(self) -> np.ndarray:
        return self.qdot_max - self.eps_qdot

    @property
    def q_min_eroded(self) -> np.ndarray:
        return self.q_min + self.eps_q + self.eps_qdot * self.dt

    @property
    def q_max_eroded(self) -> np.ndarray:
        return self.q_max - self.eps_q - self.eps_qdot * self.dt

    @classmethod
    def from_mjcf(
        cls,
        model,
        *,
        e_max: float = 5.0,
        energy_margin: float = 0.98,
        eps_q: float = 0.001,
        eps_qdot: float = 0.01,
        d_safe: float = 0.02,
        collision_pairs: list[tuple[str, str]] | None = None,
    ) -> L0RParams:
        """Build L0RParams from a MuJoCo model.

        Args:
            model: mujoco.MjModel
            collision_pairs: list of (geom_name_a, geom_name_b) pairs.
                If None, uses the CellArm default set.
        """
        import mujoco

        n_joints = model.nv
        assert n_joints == model.nu, "Expected 1:1 joint-actuator mapping"

        # Joint position limits from model
        q_min = model.jnt_range[:, 0].copy()
        q_max = model.jnt_range[:, 1].copy()

        # Velocity limits from actuator ctrlrange (velocity servos)
        qdot_max = model.actuator_ctrlrange[:, 1].copy()  # symmetric ±

        dt = model.opt.timestep

        # Resolve collision pairs
        if collision_pairs is None:
            collision_pairs = [
                ("geom_link_0", "geom_link_3"),
                ("geom_link_0", "geom_link_4"),
                ("geom_link_0", "geom_link_5"),
                ("geom_link_1", "geom_link_4"),
                ("geom_link_1", "geom_link_5"),
            ]

        pair_ids = []
        pair_names = []
        for name_a, name_b in collision_pairs:
            id_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name_a)
            id_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name_b)
            assert id_a >= 0, f"Geom not found: {name_a}"
            assert id_b >= 0, f"Geom not found: {name_b}"
            pair_ids.append((id_a, id_b))
            pair_names.append(f"{name_a}-{name_b}")

        return cls(
            n_joints=n_joints,
            q_min=q_min,
            q_max=q_max,
            qdot_max=qdot_max,
            e_max=e_max,
            energy_margin=energy_margin,
            eps_q=eps_q,
            eps_qdot=eps_qdot,
            collision_pairs=pair_ids,
            collision_pair_names=pair_names,
            d_safe=d_safe,
            dt=dt,
        )
