"""CellArm 5-DOF Gymnasium environment wrapper.

Implements Appendix D of the cumulative spec.
Provides truth/observed channels, mass matrix, FK, keepout distances.
"""

from __future__ import annotations

import pathlib
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from safety_kernel.l0r.capsule import compute_collision_distances
from safety_kernel.l0r.params import L0RParams
from safety_kernel.l0r.projection import l0r_project

# Path to MJCF model
_MODEL_PATH = pathlib.Path(__file__).resolve().parents[3] / "models" / "cell_arm.xml"

# Keepout boxes: (name, center, half_dims) — from Appendix D
_KEEPOUTS = [
    ("fixture_column", np.array([0.6, 0.35, 0.3]), np.array([0.1, 0.1, 0.3])),
    ("back_wall", np.array([-0.3, 0.0, 0.5]), np.array([0.05, 0.4, 0.5])),
]

# Allowed region for goal sampling
_ALLOWED_CENTER = np.array([0.4, 0.0, 0.5])
_ALLOWED_HALF = np.array([0.4, 0.4, 0.5])


def _point_to_box_signed_distance(
    point: np.ndarray, center: np.ndarray, half_dims: np.ndarray
) -> float:
    """Signed distance from a point to an axis-aligned box surface.

    Positive = outside, negative = penetrating.
    """
    d = np.abs(point - center) - half_dims
    outside = np.linalg.norm(np.maximum(d, 0.0))
    inside = np.min(np.maximum(d, d))  # max of all components when all negative
    # If any component > 0, we're outside
    if np.any(d > 0):
        return float(outside)
    else:
        return float(np.max(d))  # negative distance to nearest face


class CellArmEnv(gym.Env):
    """CellArm 5-DOF environment with L0-R safety projection.

    Observation: (q, qdot, tau) — 15D proprioceptive.
    Action: qdot_cmd — 5D joint velocity commands.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        model_path: str | pathlib.Path | None = None,
        e_max: float = 5.0,
        energy_margin: float = 0.98,
        eps_q: float = 0.001,
        eps_qdot: float = 0.01,
        d_safe: float = 0.02,
        max_steps: int = 500,
        seed: int | None = None,
        enable_l0r: bool = True,
    ):
        super().__init__()
        self._model_path = pathlib.Path(model_path) if model_path else _MODEL_PATH

        self.model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self.data = mujoco.MjData(self.model)
        self._max_steps = max_steps
        self._enable_l0r = enable_l0r

        self.n_joints = self.model.nv
        assert self.n_joints == 5

        # L0-R parameters
        self.l0r_params = L0RParams.from_mjcf(
            self.model,
            e_max=e_max,
            energy_margin=energy_margin,
            eps_q=eps_q,
            eps_qdot=eps_qdot,
            d_safe=d_safe,
        )

        # Spaces
        pi = np.float64(np.pi)
        obs_high = np.concatenate([
            np.full(5, pi),       # q: ±π
            np.full(5, pi),       # qdot: ±π
            np.full(5, 50.0),     # tau: max torque
        ])
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-pi * np.ones(5), high=pi * np.ones(5), dtype=np.float64
        )

        self._step_count = 0
        self._rng = np.random.default_rng(seed)

        # Pre-locate TCP site ID
        self._tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp"
        )
        assert self._tcp_site_id >= 0, "TCP site not found in model"

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Random initial configuration (modest range to avoid starting in collision)
        q_init = self._rng.uniform(-0.5, 0.5, size=self.n_joints)
        self.data.qpos[:] = q_init
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        qdot_cmd = np.clip(action, self.action_space.low, self.action_space.high)

        if self._enable_l0r:
            q = self.data.qpos[:self.n_joints].copy()
            M = self._get_mass_matrix()

            qdot_exec, intervention_info = l0r_project(
                qdot_cmd=qdot_cmd,
                q=q,
                M=M,
                model=self.model,
                data=self.data,
                params=self.l0r_params,
            )
        else:
            qdot_exec = qdot_cmd.copy()
            intervention_info = {
                "qdot_cmd": qdot_cmd.tolist(),
                "qdot_exec": qdot_exec.tolist(),
                "constraints_active": [],
                "margin_to_L0R": {},
                "energy_ratio": 0.0,
                "energy_scaling_factor": 1.0,
                "collision_pairs": [],
                "rta_nearmiss": False,
            }

        # Apply velocity command via actuators
        self.data.ctrl[:] = qdot_exec
        mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()
        info["intervention"] = intervention_info

        terminated = False
        truncated = self._step_count >= self._max_steps
        reward = 0.0  # No reward shaping at Gate 0

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Proprioceptive observation: (q, qdot, tau)."""
        q = self.data.qpos[:self.n_joints].copy()
        qdot = self.data.qvel[:self.n_joints].copy()
        tau = self.data.actuator_force[:self.n_joints].copy()
        return np.concatenate([q, qdot, tau])

    def _get_info(self) -> dict[str, Any]:
        """Truth and observed channels."""
        q = self.data.qpos[:self.n_joints].copy()
        qdot = self.data.qvel[:self.n_joints].copy()
        tau = self.data.actuator_force[:self.n_joints].copy()

        # TCP position
        tcp_pos = self.data.site_xpos[self._tcp_site_id].copy()

        # Keepout distances (TCP point → box surface)
        keepout_dists = {}
        for name, center, half_dims in _KEEPOUTS:
            keepout_dists[name] = round(
                _point_to_box_signed_distance(tcp_pos, center, half_dims), 6
            )

        # Self-collision surface distances
        collision_dists = compute_collision_distances(
            self.model, self.data, self.l0r_params.collision_pairs
        )
        collision_dist_dict = {}
        for idx, pair_name in enumerate(self.l0r_params.collision_pair_names):
            collision_dist_dict[pair_name] = round(collision_dists[idx], 6)

        truth = {
            "q": q.tolist(),
            "qdot": qdot.tolist(),
            "tau": tau.tolist(),
            "tcp_pos": tcp_pos.tolist(),
            "tcp_to_keepout_distances": keepout_dists,
            "self_collision_surface_distances": collision_dist_dict,
        }

        # Gates 0-3: observed == truth
        observed = {
            "q": q.tolist(),
            "qdot": qdot.tolist(),
            "tau": tau.tolist(),
        }

        return {
            "truth": truth,
            "observed": observed,
            "step": self._step_count,
        }

    def _get_mass_matrix(self) -> np.ndarray:
        """Compute and return the full mass matrix M(q)."""
        mujoco.mj_crb(self.model, self.data)
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def get_tcp_position(self) -> np.ndarray:
        """Get current TCP (tool center point) position in world frame."""
        return self.data.site_xpos[self._tcp_site_id].copy()

    def get_jacobian(self) -> np.ndarray:
        """Get TCP Jacobian (position rows only, 3×nv)."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._tcp_site_id)
        return jacp
