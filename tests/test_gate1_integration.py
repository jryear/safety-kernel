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
            params={"anchor": "fiducial_fixture", "drift_xyz": [0.05, 0.0, 0.0]},
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


class TestAuthorityBoundary:
    """Contract authority is decided before the current tick moves the plant."""

    _ACTION = np.full(5, 0.5)

    def test_immediate_rta_fault_suppresses_first_nonzero_action(self):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="safety_io_flip",
            params={"signal": "estop_ok"},
        )
        env = _make_env(shift_schedule=[shift])
        env.reset(seed=0)

        _, _, _, _, info = env.step(self._ACTION)

        assert info["contract_state"] == "RTA_STOP"
        np.testing.assert_array_equal(info["intervention"]["qdot_exec"], np.zeros(5))
        assert info["l0c"]["l0c_active"] is False

    def test_rta_stop_suppresses_repeated_nonzero_actions(self):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="safety_io_flip",
            params={"signal": "gate_closed"},
        )
        env = _make_env(shift_schedule=[shift])
        env.reset(seed=0)

        infos = [env.step(self._ACTION)[-1] for _ in range(3)]

        assert all(info["contract_state"] == "RTA_STOP" for info in infos)
        for info in infos:
            np.testing.assert_array_equal(info["intervention"]["qdot_exec"], np.zeros(5))

    def test_revocation_suppresses_action_on_the_trigger_tick(self):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="anchor_drift",
            params={"anchor": "fiducial_fixture", "drift_xyz": [0.05, 0.0, 0.0]},
        )
        env = _make_env(shift_schedule=[shift])
        env.reset(seed=0)

        _run_steps(env, 13)
        _, _, _, _, info = env.step(self._ACTION)

        assert info["contract_state"] == "REVOKED"
        np.testing.assert_array_equal(info["intervention"]["qdot_exec"], np.zeros(5))

    def test_degraded_cap_applies_on_the_trigger_tick(self):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="sensor_stale",
            params={"frame_age_ms": 500.0},
        )
        env = _make_env(shift_schedule=[shift])
        env.reset(seed=0)

        _run_steps(env, 25)
        action = np.full(5, np.pi)
        jacobian = env.get_jacobian().copy()
        assert np.linalg.norm(jacobian @ action) > 0.6
        _, _, _, _, info = env.step(action)

        assert info["contract_state"] == "DEGRADED"
        assert info["l0c"]["l0c_active"] is True
        assert info["l0c"]["tcp_speed_after"] <= 0.6
        assert "tcp_speed_cap" in info["l0c"]["constraints_active"]
        assert np.linalg.norm(jacobian @ info["command_trace"]["final_command"]) <= 0.6

    def test_l0r_nearmiss_stops_before_the_plant_step(self):
        env = _make_env()
        env.reset(seed=0)
        env.data.qpos[0] = env.l0r_params.q_max_eroded[0] + 0.1

        _, _, _, _, info = env.step(self._ACTION)

        assert info["contract_state"] == "RTA_STOP"
        assert info["contract_events"][0]["trigger_monitor"] == "l0r_nearmiss"
        np.testing.assert_array_equal(info["intervention"]["qdot_exec"], np.zeros(5))

    def test_operator_reset_requires_current_passing_evidence(self):
        shift = ShiftSpec(
            start_step=0, end_step=2,
            shift_type="safety_io_flip",
            params={"signal": "estop_ok"},
        )
        env = _make_env(shift_schedule=[shift])
        env.reset(seed=0)
        env.step(self._ACTION)
        assert env._state_machine.state.value == "RTA_STOP"

        # The original fault is still present at tick 1, so reset is denied.
        assert env.operator_reset() == []
        assert env._state_machine.state.value == "RTA_STOP"

        # Tick 1 consumes the denied request in logged evidence.  A fresh
        # request is evaluated only inside tick 2, after the fault has cleared.
        env.step(self._ACTION)
        assert env._state_machine.state.value == "RTA_STOP"
        assert env.operator_reset() == []
        events = env.step(self._ACTION)[-1]["contract_events"]
        assert env._state_machine.state.value == "ACTIVE"
        assert events[0]["trigger_monitor"] == "operator_reset"

    def test_operator_reset_is_preserved_in_next_decision_evidence(self):
        shift = ShiftSpec(start_step=0, end_step=2, shift_type="safety_io_flip", params={"signal": "estop_ok"})
        env = _make_env(shift_schedule=[shift])
        env.reset(seed=0)
        env.step(self._ACTION)
        env.step(self._ACTION)
        assert env.operator_reset() == []
        info = env.step(self._ACTION)[-1]
        authority = info["authority_info"]
        assert authority["reset_input"] is True
        assert authority["state_pre"] == "RTA_STOP"
        assert authority["state_post"] == "ACTIVE"
        assert info["contract_events"][0]["trigger_monitor"] == "operator_reset"

    def test_missing_monitor_evidence_stops_without_debounce(self):
        env = _make_env()
        env.reset(seed=0)
        env._shift_injector.get_monitor_inputs = lambda step: {}

        _, _, _, _, info = env.step(self._ACTION)

        assert info["contract_state"] == "RTA_STOP"
        assert info["contract_events"][0]["trigger_monitor"] == "monitor_evidence_missing"
        np.testing.assert_array_equal(info["intervention"]["qdot_exec"], np.zeros(5))

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_action_is_rejected(self, value):
        env = _make_env()
        env.reset(seed=0)
        action = self._ACTION.copy()
        action[0] = value

        with pytest.raises(ValueError, match="finite"):
            env.step(action)

    def test_command_trace_and_authority_decision_preserve_all_boundaries(self):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="safety_io_flip",
            params={"signal": "estop_ok"},
        )
        env = _make_env(shift_schedule=[shift])
        env.reset(seed=0)

        _, _, _, _, info = env.step(np.full(5, 99.0))

        trace = info["command_trace"]
        decision = info["authority_decision"]
        np.testing.assert_array_equal(trace["proposal"], env.action_space.high)
        # RTA authority suppression is intentionally after L0-C/L0-R so the
        # evidence retains the actual proposal-to-final veto boundary.
        np.testing.assert_array_equal(trace["authorized_command"], trace["proposal"])
        np.testing.assert_array_equal(trace["after_l0c"], trace["proposal"])
        np.testing.assert_array_equal(trace["final_command"], np.zeros(5))
        assert decision == {
            "pre_state": "ACTIVE",
            "post_state": "RTA_STOP",
            "authorized": False,
            "reason": "estop_chain_ok",
            "proposal": trace["proposal"],
            "final_command": trace["final_command"],
        }
