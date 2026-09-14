"""CellArm 5-DOF Gymnasium environment wrapper.

Implements Appendix D of the cumulative spec.
Provides truth/observed channels, mass matrix, FK, keepout distances.
Gate 1: optional contract integration (state machine, monitors, L0-C enforcement).
"""

from __future__ import annotations

import pathlib
from typing import Any, Mapping

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from safety_kernel.l0r.capsule import compute_collision_distances
from safety_kernel.l0r.params import L0RParams
from safety_kernel.l0r.projection import l0r_project
from safety_kernel.contract.state_machine import ContractState

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
        contract_path: str | pathlib.Path | None = None,
        shift_schedule: list | None = None,
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

        # ── Gate 1: Contract integration (optional) ──────────────────
        self._contract = None
        self._state_machine = None
        self._monitors = None
        self._shift_injector = None
        self._contract_events: list[dict] = []
        # A reset is only a request between ticks.  Its monitor evaluation and
        # any transition occur once, inside the next decision tick, so no
        # monitor state can advance outside independently replayable evidence.
        self._pending_reset_input = False
        self._keepouts = list(_KEEPOUTS)  # Default: hardcoded

        if contract_path is not None:
            from safety_kernel.contract.loader import load_contract_file
            from safety_kernel.contract.monitors import create_monitors
            from safety_kernel.contract.state_machine import ContractStateMachine
            from safety_kernel.contract.shift_injector import ShiftInjector

            lc = load_contract_file(pathlib.Path(contract_path))
            self._contract = lc.contract
            loop_rate_hz = 1.0 / self.model.opt.timestep
            self._state_machine = ContractStateMachine(loop_rate_hz)
            self._monitors = create_monitors(self._contract, loop_rate_hz)
            self._shift_injector = ShiftInjector(
                self._contract, shift_schedule,
            )
            # Use contract keepouts instead of hardcoded
            self._keepouts = [
                (k.name, k.center, k.half_dims)
                for k in self._contract.keepouts
            ]

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
        self._contract_events = []
        self._pending_reset_input = False

        # Per-episode reset of contract components
        if self._state_machine is not None:
            self._state_machine.reset_episode()
        if self._monitors is not None:
            for m in self._monitors:
                m.reset_episode()
        if self._shift_injector is not None:
            self._shift_injector.reset_episode()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Action must have shape {self.action_space.shape}, got {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("Action must contain only finite values")

        qdot_proposal = np.clip(
            action_array, self.action_space.low, self.action_space.high
        )
        tick = self._step_count
        runtime_pre_state = self._contract_state_value()
        pre_state = runtime_pre_state
        pre_step_info = self._get_info()
        pre_step_timestamp_ns = int(float(self.data.time) * 1_000_000_000)

        # ── Contract layer: evidence + state transition ──────────────
        # Contract authority is decided at the start of the zero-based tick,
        # before L0-C/L0-R enforcement and before the plant can move.
        step_events: list[dict] = []
        reset_input = self._pending_reset_input
        self._pending_reset_input = False
        monitor_inputs: dict[str, Any] | None = None
        monitor_results: dict[str, dict] | None = None
        if self._monitors is not None and self._state_machine is not None:
            evaluated_events, monitor_inputs, monitor_results = self._evaluate_contract_layer(tick)
            step_events.extend(evaluated_events)
            if reset_input and monitor_results is not None:
                from safety_kernel.contract.state_machine import _build_monitors_status

                # Native reset authority is stricter than a raw predicate
                # check: a debounced monitor must also be untriggered.
                all_passing = all(
                    not result["predicate_violated"] and not result["triggered"]
                    for result in monitor_results.values()
                )
                reset_events = self._state_machine.operator_reset(
                    tick, _build_monitors_status(monitor_results), all_passing
                ) if self._state_machine.state in (ContractState.REVOKED, ContractState.RTA_STOP) else []
                step_events.extend(reset_events)
                self._contract_events.extend(reset_events)

        # ── L0-C enforcement (pre-L0-R) ──────────────────────────────
        l0c_info = None
        qdot_authorized = qdot_proposal.copy()
        qdot_l0c = qdot_authorized.copy()
        authority_reason = "no_contract"
        if self._contract is not None and self._state_machine is not None:
            from safety_kernel.contract.l0c_enforce import l0c_enforce

            if self._state_machine.state in (
                ContractState.REVOKED,
                ContractState.RTA_STOP,
            ):
                # Authority suppression deliberately occurs *after* L0-C/L0-R.
                # This preserves a real proposal→L0-R→final trace and makes a
                # nonzero rejected proposal observable as an RTA intervention.
                qdot_l0c = qdot_proposal.copy()
                authority_reason = self._authority_reason(step_events)
                l0c_info = {
                    "l0c_active": False,
                    "tcp_speed_before": 0.0,
                    "tcp_speed_after": 0.0,
                    "scaling_factor": 1.0,
                    "constraints_active": [],
                }
            else:
                J = self.get_jacobian()
                qdot_l0c, l0c_info = l0c_enforce(
                    qdot_authorized, J,
                    self._state_machine.state,
                    self._contract.revocation_policy.on_degrade_max_tcp_speed_mps,
                )
                l0c_info["max_tcp_speed_mps"] = (
                    self._contract.revocation_policy.on_degrade_max_tcp_speed_mps
                )
                # Command-stage Jacobian evidence is required to independently
                # derive TCP speed; it is not a physical containment claim.
                l0c_info["tcp_jacobian"] = J.tolist()
                authority_reason = f"state_{self._state_machine.state.value.lower()}"

        # ── L0-R projection ──────────────────────────────────────────
        if self._enable_l0r:
            q = self.data.qpos[:self.n_joints].copy()
            M = self._get_mass_matrix()

            qdot_exec, intervention_info = l0r_project(
                qdot_cmd=qdot_l0c,
                q=q,
                M=M,
                model=self.model,
                data=self.data,
                params=self.l0r_params,
            )
        else:
            qdot_exec = qdot_l0c.copy()
            intervention_info = {
                "qdot_cmd": qdot_l0c.tolist(),
                "qdot_exec": qdot_exec.tolist(),
                "constraints_active": [],
                "margin_to_L0R": {},
                "energy_ratio": 0.0,
                "energy_scaling_factor": 1.0,
                "collision_pairs": [],
                "rta_nearmiss": False,
            }
        qdot_l0r = qdot_exec.copy()

        # An L0-R near-miss is an immediate RTA trigger. Suppress the command
        # on this same tick; recording the transition after mj_step would leave
        # one untrusted action authorized after the panic condition was known.
        if (
            intervention_info.get("rta_nearmiss", False)
            and self._state_machine is not None
        ):
            from safety_kernel.contract.state_machine import _build_monitors_status

            monitors_status = _build_monitors_status(monitor_results or {})
            nearmiss_events = self._state_machine.notify_l0r_nearmiss(
                tick, monitors_status,
            )
            step_events.extend(nearmiss_events)
            self._contract_events.extend(nearmiss_events)
            qdot_exec = np.zeros_like(qdot_exec)
            intervention_info["qdot_exec"] = qdot_exec.tolist()
            authority_reason = "l0r_nearmiss"

        # REVOKED and RTA_STOP have no L2 authority.  This is after L0-R by
        # construction; therefore the final vector is a persistent exact zero
        # while the preceding L0-R output remains available for causality.
        post_state = self._contract_state_value()
        if post_state in {ContractState.REVOKED.value, ContractState.RTA_STOP.value}:
            if authority_reason == "no_contract" or authority_reason.startswith("state_"):
                authority_reason = self._authority_reason(step_events)
            qdot_exec = np.zeros_like(qdot_l0r)
            intervention_info["qdot_exec"] = qdot_exec.tolist()
            final_collision_distances = compute_collision_distances(
                self.model, self.data, self.l0r_params.collision_pairs
            )
            intervention_info["collision_pairs"] = [
                {"pair_id": name, "distance": float(distance)}
                for name, distance in zip(
                    self.l0r_params.collision_pair_names, final_collision_distances
                )
            ]

        # Apply velocity command via actuators
        self.data.ctrl[:] = qdot_exec
        mujoco.mj_step(self.model, self.data)

        self._step_count = tick + 1

        obs = self._get_obs()
        info = self._get_info()
        # The telemetry ABI is zero-based and describes the decision that just
        # occurred, not the count of completed MuJoCo steps.
        info["step"] = tick
        info["intervention"] = intervention_info
        post_state = self._contract_state_value()
        authorized = post_state not in {
            ContractState.REVOKED.value,
            ContractState.RTA_STOP.value,
        }
        info["command_trace"] = {
            "proposal": qdot_proposal.tolist(),
            "authorized_command": qdot_authorized.tolist(),
            "after_l0c": qdot_l0c.tolist(),
            "after_l0r": qdot_l0r.tolist(),
            "final_command": qdot_exec.tolist(),
        }
        info["authority_decision"] = {
            "pre_state": pre_state,
            "post_state": post_state,
            "authorized": authorized,
            "reason": authority_reason,
            "proposal": qdot_proposal.tolist(),
            "final_command": qdot_exec.tolist(),
        }
        info["authority_info"] = {
            "proposal": qdot_proposal.tolist(),
            "l0c_command": qdot_l0c.tolist(),
            "l0r_command": qdot_l0r.tolist(),
            "final_command": qdot_exec.tolist(),
            "state_pre": pre_state,
            "state_post": post_state,
            "authorized": authorized,
            "reason": authority_reason,
            "reset_input": reset_input,
            "monitor_inputs": monitor_inputs,
            "monitor_results": monitor_results,
        }
        info["pre_step_truth"] = pre_step_info["truth"]
        info["pre_step_observed"] = pre_step_info["observed"]
        info["simulated_pre_step_timestamp_ns"] = pre_step_timestamp_ns
        q_pre = np.asarray(pre_step_info["observed"]["q"], dtype=np.float64)
        qdot_pre = np.asarray(pre_step_info["observed"]["qdot"], dtype=np.float64)
        info["pre_step_believed"] = {
            "q_interval": np.stack(
                (q_pre - self.l0r_params.eps_q, q_pre + self.l0r_params.eps_q),
                axis=1,
            ).tolist(),
            "qdot_interval": np.stack(
                (
                    qdot_pre - self.l0r_params.eps_qdot,
                    qdot_pre + self.l0r_params.eps_qdot,
                ),
                axis=1,
            ).tolist(),
        }
        info["l0r_evidence"] = {
            "q": pre_step_info["truth"]["q"],
            "qdot_exec": qdot_exec.tolist(),
            "qdot_max_eroded": self.l0r_params.qdot_max_eroded.tolist(),
            "q_min_eroded": self.l0r_params.q_min_eroded.tolist(),
            "q_max_eroded": self.l0r_params.q_max_eroded.tolist(),
            "dt": float(self.l0r_params.dt),
            "mass_matrix": M.tolist() if self._enable_l0r else self._get_mass_matrix().tolist(),
            "e_max_safe": float(self.l0r_params.e_max_safe),
            "d_safe": float(self.l0r_params.d_safe),
            "collision_distances": {
                item["pair_id"]: item["distance"]
                for item in intervention_info.get("collision_pairs", [])
            },
        }
        if l0c_info is not None:
            info["l0c"] = l0c_info
        if step_events:
            info["contract_events"] = step_events

        terminated = False
        truncated = self._step_count >= self._max_steps
        reward = 0.0  # No reward shaping at Gate 0

        return obs, reward, terminated, truncated, info

    def _contract_state_value(self) -> str | None:
        """Return the current contract state without inventing one in Gym-only mode."""
        if self._state_machine is None:
            return None
        return self._state_machine.state.value

    def _authority_reason(self, events: list[dict]) -> str:
        """Return the deterministic monitor/state reason for denied authority."""
        if events:
            return events[-1]["trigger_monitor"]
        return f"state_{self._state_machine.state.value.lower()}"

    def operator_reset(self) -> list[dict]:
        """Queue one explicit reset request for evaluation in the next tick.

        This method deliberately performs no monitor evaluation or transition:
        every evaluation affecting reset authority is emitted exactly once as
        that next tick's decision evidence.  Repeated requests are represented
        by repeated tick records and cannot consume hidden hysteresis state.
        """
        if self._state_machine is None or self._monitors is None:
            raise RuntimeError("Operator reset requires contract integration")

        if self._state_machine.state not in (ContractState.REVOKED, ContractState.RTA_STOP):
            raise RuntimeError("Operator reset is legal only from REVOKED or RTA_STOP")
        self._pending_reset_input = True
        return []

    def _evaluate_contract_layer(
        self,
        step: int,
    ) -> tuple[list[dict], dict[str, Any] | None, dict[str, dict] | None]:
        """Evaluate monitor evidence and drive the state machine for ``step``."""
        monitor_inputs, monitor_results = self._collect_monitor_evidence(step)

        if monitor_results is None:
            # A monitor result cannot be inferred from absent evidence. Make the
            # missing-evidence condition an immediate RTA trigger rather than
            # allowing debounce/default values to preserve task authority.
            monitor_results = {
                "monitor_evidence_missing": {
                    "triggered": True,
                    "should_trigger": True,
                    "should_clear": False,
                    "severity": "rta_stop",
                    "predicate_violated": True,
                }
            }

        # Drive state machine with monitor results
        events = self._state_machine.process_monitor_events(step, monitor_results)

        # Check for DEGRADED→ACTIVE: set recovery guard with clearing monitor's hysteresis
        for event in events:
            if event["from_state"] == "DEGRADED" and event["to_state"] == "ACTIVE":
                clearing_name = event["trigger_monitor"]
                for m in self._monitors:
                    if m.name == clearing_name:
                        self._state_machine.set_recovery_guard(step, m.hysteresis_ticks)
                        break

        self._contract_events.extend(events)
        # The state machine receives a local fail-closed sentinel, but the
        # wire evidence remains unavailable rather than falsely describing a
        # partial input or synthetic monitor result as real telemetry.
        if monitor_inputs is None or "monitor_evidence_missing" in monitor_results:
            return events, None, None
        return events, monitor_inputs, monitor_results

    def _collect_monitor_evidence(
        self, step: int,
    ) -> tuple[dict[str, Any] | None, dict[str, dict] | None]:
        """Return current monitor inputs/results, preserving unavailable evidence."""
        if self._shift_injector is None:
            return None, None

        try:
            monitor_inputs = self._shift_injector.get_monitor_inputs(step)
        except (KeyError, TypeError, ValueError):
            return None, None
        if not self._monitor_evidence_complete(monitor_inputs):
            return None, None

        try:
            monitor_results: dict[str, dict] = {}
            for m in self._monitors:
                result = m.evaluate(monitor_inputs)
                monitor_results[m.name] = {
                    "triggered": result.triggered,
                    "should_trigger": result.should_trigger,
                    "should_clear": result.should_clear,
                    "severity": m.severity,
                    "predicate_violated": result.predicate_violated,
                }
            return monitor_inputs, monitor_results
        except (KeyError, TypeError, ValueError, IndexError):
            return None, None

    def _monitor_evidence_complete(self, inputs: Any) -> bool:
        """Validate that every configured monitor has finite current evidence."""
        if not isinstance(inputs, Mapping):
            return False

        for config in self._contract.monitors:
            if config.type == "anchor_pose_consistency":
                poses = inputs.get("anchor_poses")
                if not isinstance(poses, Mapping):
                    return False
                for anchor_name in config.params.get("anchors", []):
                    pose = poses.get(anchor_name)
                    if not self._finite_vector_mapping(pose, "xyz") or not self._finite_vector_mapping(pose, "rpy"):
                        return False
            elif config.type == "safety_io_health":
                safety_io = inputs.get("safety_io")
                signal = config.params.get("signal")
                if (
                    not isinstance(safety_io, Mapping)
                    or signal not in safety_io
                    or not isinstance(safety_io[signal], bool)
                ):
                    return False
            elif config.type == "sensor_health":
                sensor = inputs.get("sensor_health")
                if not isinstance(sensor, Mapping):
                    return False
                if not self._finite_scalar(sensor.get("frame_age_ms")):
                    return False
                if config.params.get("monotonic_timestamps", True) and not self._finite_scalar(sensor.get("timestamp")):
                    return False
            elif config.type == "robot_mount_integrity":
                pose = inputs.get("base_pose_estimate")
                if not self._finite_vector_mapping(pose, "xyz") or not self._finite_vector_mapping(pose, "rpy"):
                    return False
            else:
                return False
        return True

    @staticmethod
    def _finite_scalar(value: Any) -> bool:
        return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))

    @classmethod
    def _finite_vector_mapping(cls, value: Any, key: str) -> bool:
        if not isinstance(value, Mapping):
            return False
        vector = value.get(key)
        try:
            array = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return array.shape == (3,) and bool(np.all(np.isfinite(array)))

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
        for name, center, half_dims in self._keepouts:
            keepout_dists[name] = _point_to_box_signed_distance(
                tcp_pos, center, half_dims
            )

        # Self-collision surface distances
        collision_dists = compute_collision_distances(
            self.model, self.data, self.l0r_params.collision_pairs
        )
        collision_dist_dict = {}
        for idx, pair_name in enumerate(self.l0r_params.collision_pair_names):
            collision_dist_dict[pair_name] = float(collision_dists[idx])

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

        info = {
            "truth": truth,
            "observed": observed,
            "step": self._step_count,
        }

        # Gate 1: contract state + monitor status
        if self._state_machine is not None:
            info["contract_state"] = self._state_machine.state.value
        if self._monitors is not None:
            info["monitors_status"] = {
                m.name: {"triggered": m.triggered}
                for m in self._monitors
            }

        return info

    def _get_mass_matrix(self) -> np.ndarray:
        """Compute and return the full mass matrix M(q)."""
        mujoco.mj_crb(self.model, self.data)
        M = np.zeros((self.model.nv, self.model.nv))
        try:
            # MuJoCo >= 3.13 takes MjData directly and no longer exposes qM.
            mujoco.mj_fullM(self.model, self.data, M)
        except TypeError:
            # Compatibility with the older binding signature used when Gate 0
            # was originally implemented.
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
