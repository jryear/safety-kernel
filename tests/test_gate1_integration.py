"""Gate 1 integration tests.

H3 predicates: detection SLA, no illegal transitions, no flapping.
Contract geometry loading, L0-R near-miss → RTA_STOP, backward compat.
"""

import pathlib

import numpy as np
import pytest

from safety_kernel.env.cell_arm import CellArmEnv
from safety_kernel.contract.shift_injector import ShiftSpec

_CONTRACTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "contracts"
_TRACK_A = _CONTRACTS_DIR / "cell_arm_track_a.yaml"

LOOP_RATE_HZ = 50.0  # CellArm default


def _make_env(shift_schedule=None, seed=42, max_steps=500):
    """Create env with contract loaded."""
    return CellArmEnv(
        contract_path=_TRACK_A,
        shift_schedule=shift_schedule,
        seed=seed,
        max_steps=max_steps,
    )


def _run_steps(env, n_steps, action=None):
    """Run n_steps, return list of info dicts."""
    if action is None:
        action = np.zeros(5)
    infos = []
    for _ in range(n_steps):
        _, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        if terminated or truncated:
            break
    return infos


def _find_transition(infos, to_state):
    """Find the first step where a transition to to_state occurred."""
    for info in infos:
        for event in info.get("contract_events", []):
            if event["to_state"] == to_state:
                return event["step"]
    return None


class TestContractLoadsGeometry:
    """Contract keepouts replace hardcoded values."""

    def test_keepouts_from_contract(self):
        env = _make_env()
        env.reset(seed=0)
        assert len(env._keepouts) == 2
        names = {k[0] for k in env._keepouts}
        assert "fixture_column" in names
        assert "back_wall" in names

    def test_contract_state_in_info(self):
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(5))
        assert info["contract_state"] == "ACTIVE"

    def test_monitors_status_in_info(self):
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(5))
        assert "monitors_status" in info
        assert "anchors_consistent" in info["monitors_status"]


class TestBackwardCompat:
    """Gate 0 mode (no contract) still works."""

    def test_no_contract_works(self):
        env = CellArmEnv(seed=42)
        obs, info = env.reset(seed=0)
        assert obs.shape == (15,)
        obs, _, _, _, info = env.step(np.zeros(5))
        assert "contract_state" not in info

    def test_gate0_acceptance_still_passes(self):
        """Run the same kernel correctness smoke test as Gate 0."""
        env = CellArmEnv(seed=42, max_steps=200)
        n_episodes = 5
        total_violations = 0

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            for step in range(200):
                action = env.action_space.sample()
                obs, _, terminated, truncated, info = env.step(action)
                intervention = info.get("intervention", {})
                # Check L0-R guarantees
                qdot_exec = np.array(intervention.get("qdot_exec", np.zeros(5)))
                for i in range(5):
                    if abs(qdot_exec[i]) > env.l0r_params.qdot_max_eroded[i] + 1e-8:
                        total_violations += 1
                if terminated or truncated:
                    break

        assert total_violations == 0


class TestDetectionSLA:
    """H3 predicate (b): detection SLA per monitor type."""

    def test_sla_gate_interlock(self):
        """gate_interlock_ok: rta_stop severity, 0ms debounce → RTA_STOP at injection step + 1."""
        shift = ShiftSpec(
            start_step=50, end_step=None,
            shift_type="safety_io_flip",
            params={"signal": "gate_closed"},
        )
        env = _make_env(shift_schedule=[shift], max_steps=200)
        env.reset(seed=0)
        infos = _run_steps(env, 200)

        transition_step = _find_transition(infos, "RTA_STOP")
        assert transition_step is not None
        # 0ms debounce → trigger 1 step after first violation
        assert transition_step <= 52, f"SLA violated: triggered at step {transition_step}"

    def test_sla_estop(self):
        """estop_chain_ok: 0ms debounce → trigger at injection step + 1."""
        shift = ShiftSpec(
            start_step=50, end_step=None,
            shift_type="safety_io_flip",
            params={"signal": "estop_ok"},
        )
        env = _make_env(shift_schedule=[shift], max_steps=200)
        env.reset(seed=0)
        infos = _run_steps(env, 200)

        # estop is rta_stop severity → goes to RTA_STOP
        transition_step = _find_transition(infos, "RTA_STOP")
        assert transition_step is not None
        assert transition_step <= 52, f"SLA violated: triggered at step {transition_step}"

    def test_sla_anchors_consistent(self):
        """anchors_consistent: 250ms debounce = 13 ticks → trigger by step 50+14=64."""
        shift = ShiftSpec(
            start_step=50, end_step=None,
            shift_type="anchor_drift",
            params={"anchor": "fiducial_fixture", "drift_xyz": [0.05, 0.0, 0.0]},
        )
        env = _make_env(shift_schedule=[shift], max_steps=200)
        env.reset(seed=0)
        infos = _run_steps(env, 200)

        # anchors_consistent is revoke severity
        transition_step = _find_transition(infos, "REVOKED")
        assert transition_step is not None
        # 13 debounce ticks + 1 = detection at step 50 + 14 = 64
        assert transition_step <= 65, f"SLA violated: triggered at step {transition_step}"

    def test_sla_front_cam_health(self):
        """front_cam_health: 500ms debounce = 25 ticks → trigger by step 50+26=76."""
        shift = ShiftSpec(
            start_step=50, end_step=None,
            shift_type="sensor_stale",
            params={"frame_age_ms": 500.0},
        )
        env = _make_env(shift_schedule=[shift], max_steps=200)
        env.reset(seed=0)
        infos = _run_steps(env, 200)

        transition_step = _find_transition(infos, "DEGRADED")
        assert transition_step is not None
        # 25 debounce ticks + 1 = detection by step 50 + 26 = 76
        assert transition_step <= 77, f"SLA violated: triggered at step {transition_step}"

    def test_sla_robot_mount_integrity(self):
        """robot_mount_integrity: 250ms debounce = 13 ticks."""
        shift = ShiftSpec(
            start_step=50, end_step=None,
            shift_type="base_drift",
            params={"drift_xyz": [0.01, 0.0, 0.0]},
        )
        env = _make_env(shift_schedule=[shift], max_steps=200)
        env.reset(seed=0)
        infos = _run_steps(env, 200)

        transition_step = _find_transition(infos, "REVOKED")
        assert transition_step is not None
        assert transition_step <= 65, f"SLA violated: triggered at step {transition_step}"


class TestNoIllegalTransitions:
    """H3 predicate (c): every transition matches the transition table."""

    LEGAL_TRANSITIONS = {
        ("ACTIVE", "DEGRADED"),
        ("ACTIVE", "REVOKED"),
        ("ACTIVE", "RTA_STOP"),
        ("DEGRADED", "REVOKED"),
        ("DEGRADED", "ACTIVE"),
        ("DEGRADED", "RTA_STOP"),
        ("REVOKED", "RTA_STOP"),
        ("REVOKED", "ACTIVE"),
        ("RTA_STOP", "ACTIVE"),
    }

    def test_smoke_no_shifts(self):
        """No shifts → no transitions (all monitors pass)."""
        env = _make_env(max_steps=200)
        env.reset(seed=0)
        infos = _run_steps(env, 200)
        for info in infos:
            for event in info.get("contract_events", []):
                pair = (event["from_state"], event["to_state"])
                assert pair in self.LEGAL_TRANSITIONS, f"Illegal transition: {pair}"

    def test_with_anchor_drift(self):
        """Anchor drift → transitions must be legal."""
        shift = ShiftSpec(
            start_step=30, end_step=80,
            shift_type="anchor_drift",
            params={"anchor": "fid_a", "drift_xyz": [0.05, 0.0, 0.0]},
        )
        env = _make_env(shift_schedule=[shift], max_steps=200)
        env.reset(seed=0)
        infos = _run_steps(env, 200)
        for info in infos:
            for event in info.get("contract_events", []):
                pair = (event["from_state"], event["to_state"])
                assert pair in self.LEGAL_TRANSITIONS, f"Illegal transition: {pair}"


class TestActiveStateWithNoFaults:
    """With no shifts, system stays ACTIVE for entire episode."""

    def test_stays_active(self):
        env = _make_env(max_steps=200)
        env.reset(seed=0)
        infos = _run_steps(env, 200)
        for info in infos:
            assert info.get("contract_state") == "ACTIVE"
            assert len(info.get("contract_events", [])) == 0


class TestEpisodeReset:
    """Contract state resets properly across episodes."""

    def test_reset_clears_state(self):
        """After a fault triggers a transition, reset restores ACTIVE."""
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="safety_io_flip",
            params={"signal": "estop_ok"},
        )
        env = _make_env(shift_schedule=[shift], max_steps=50)
        env.reset(seed=0)
        _run_steps(env, 10)
        # Should be in RTA_STOP (estop is rta_stop severity, 0 debounce)
        assert env._state_machine.state.value == "RTA_STOP"

        # Reset
        env.reset(seed=1)
        assert env._state_machine.state.value == "ACTIVE"
