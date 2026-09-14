"""Shift injector tests.

Tests shift activation timing, each shift type, multiple shifts,
perfect baseline correctness, and snapshot round-trip.
"""

import numpy as np
import pytest
import pathlib

from safety_kernel.contract.errors import ShiftValidationError
from safety_kernel.contract.loader import load_contract_file
from safety_kernel.contract.shift_injector import ShiftInjector, ShiftSpec

_CONTRACTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "contracts"
_TRACK_A = _CONTRACTS_DIR / "cell_arm_track_a.yaml"


@pytest.fixture
def contract():
    return load_contract_file(_TRACK_A).contract


class TestPerfectBaseline:
    """Without shifts, all inputs match contract expectations."""

    def test_anchor_poses_match(self, contract):
        inj = ShiftInjector(contract)
        inputs = inj.get_monitor_inputs(0)
        for anchor in contract.anchors:
            observed = inputs["anchor_poses"][anchor.name]
            np.testing.assert_allclose(observed["xyz"], anchor.expected_pose.xyz)
            np.testing.assert_allclose(observed["rpy"], anchor.expected_pose.rpy)

    def test_safety_io_matches(self, contract):
        inj = ShiftInjector(contract)
        inputs = inj.get_monitor_inputs(0)
        for monitor in contract.monitors:
            if monitor.type == "safety_io_health":
                signal = monitor.params["signal"]
                expected = monitor.params["expected"]
                assert inputs["safety_io"][signal] == expected

    def test_sensor_health_fresh(self, contract):
        inj = ShiftInjector(contract)
        inputs = inj.get_monitor_inputs(0)
        assert inputs["sensor_health"]["frame_age_ms"] < 100  # Well within any threshold

    def test_base_pose_matches(self, contract):
        inj = ShiftInjector(contract)
        inputs = inj.get_monitor_inputs(0)
        np.testing.assert_allclose(
            inputs["base_pose_estimate"]["xyz"],
            contract.robot_base_pose.xyz,
        )

    def test_monotonic_timestamps(self, contract):
        inj = ShiftInjector(contract)
        prev_ts = -1.0
        for step in range(100):
            inputs = inj.get_monitor_inputs(step)
            ts = inputs["sensor_health"]["timestamp"]
            assert ts > prev_ts
            prev_ts = ts


class TestShiftActivation:
    """Shifts activate at start_step, deactivate at end_step."""

    def test_inactive_before_start(self, contract):
        shift = ShiftSpec(
            start_step=50, end_step=100,
            shift_type="anchor_drift",
            params={"anchor": contract.anchors[0].name, "drift_xyz": [0.1, 0, 0]},
        )
        inj = ShiftInjector(contract, [shift])
        assert len(inj.get_active_shifts(49)) == 0

    def test_active_at_start(self, contract):
        shift = ShiftSpec(
            start_step=50, end_step=100,
            shift_type="anchor_drift",
            params={"anchor": contract.anchors[0].name, "drift_xyz": [0.1, 0, 0]},
        )
        inj = ShiftInjector(contract, [shift])
        assert len(inj.get_active_shifts(50)) == 1

    def test_active_during(self, contract):
        shift = ShiftSpec(
            start_step=50, end_step=100,
            shift_type="anchor_drift",
            params={"anchor": contract.anchors[0].name, "drift_xyz": [0.1, 0, 0]},
        )
        inj = ShiftInjector(contract, [shift])
        assert len(inj.get_active_shifts(75)) == 1

    def test_inactive_at_end(self, contract):
        shift = ShiftSpec(
            start_step=50, end_step=100,
            shift_type="anchor_drift",
            params={"anchor": contract.anchors[0].name, "drift_xyz": [0.1, 0, 0]},
        )
        inj = ShiftInjector(contract, [shift])
        assert len(inj.get_active_shifts(100)) == 0

    def test_none_end_step_stays_active(self, contract):
        shift = ShiftSpec(
            start_step=50, end_step=None,
            shift_type="anchor_drift",
            params={"anchor": contract.anchors[0].name, "drift_xyz": [0.1, 0, 0]},
        )
        inj = ShiftInjector(contract, [shift])
        assert len(inj.get_active_shifts(1000)) == 1


class TestAnchorDrift:
    """anchor_drift shift type."""

    def test_drift_applied(self, contract):
        anchor_name = contract.anchors[0].name
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="anchor_drift",
            params={"anchor": anchor_name, "drift_xyz": [0.05, 0.0, 0.0]},
        )
        inj = ShiftInjector(contract, [shift])
        inputs = inj.get_monitor_inputs(0)
        expected_xyz = contract.anchors[0].expected_pose.xyz + np.array([0.05, 0, 0])
        np.testing.assert_allclose(inputs["anchor_poses"][anchor_name]["xyz"], expected_xyz)

    def test_no_drift_before_start(self, contract):
        anchor_name = contract.anchors[0].name
        shift = ShiftSpec(
            start_step=10, end_step=None,
            shift_type="anchor_drift",
            params={"anchor": anchor_name, "drift_xyz": [0.05, 0, 0]},
        )
        inj = ShiftInjector(contract, [shift])
        inputs = inj.get_monitor_inputs(5)
        np.testing.assert_allclose(
            inputs["anchor_poses"][anchor_name]["xyz"],
            contract.anchors[0].expected_pose.xyz,
        )


class TestSafetyIOFlip:
    """safety_io_flip shift type."""

    def test_signal_flipped(self, contract):
        # Find a safety_io monitor to get its signal name
        io_monitor = None
        for m in contract.monitors:
            if m.type == "safety_io_health":
                io_monitor = m
                break
        assert io_monitor is not None
        signal = io_monitor.params["signal"]
        expected = io_monitor.params["expected"]

        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="safety_io_flip",
            params={"signal": signal},
        )
        inj = ShiftInjector(contract, [shift])
        inputs = inj.get_monitor_inputs(0)
        assert inputs["safety_io"][signal] == (not expected)


class TestSensorStale:
    """sensor_stale shift type."""

    def test_frame_age_override(self, contract):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="sensor_stale",
            params={"frame_age_ms": 500.0},
        )
        inj = ShiftInjector(contract, [shift])
        inputs = inj.get_monitor_inputs(0)
        assert inputs["sensor_health"]["frame_age_ms"] == 500.0

    def test_timestamp_override(self, contract):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="sensor_stale",
            params={"timestamp_override": -1.0},
        )
        inj = ShiftInjector(contract, [shift])
        inputs = inj.get_monitor_inputs(10)
        assert inputs["sensor_health"]["timestamp"] == -1.0


class TestBaseDrift:
    """base_drift shift type."""

    def test_drift_xyz(self, contract):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="base_drift",
            params={"drift_xyz": [0.01, 0.0, 0.0]},
        )
        inj = ShiftInjector(contract, [shift])
        inputs = inj.get_monitor_inputs(0)
        expected_xyz = contract.robot_base_pose.xyz + np.array([0.01, 0, 0])
        np.testing.assert_allclose(inputs["base_pose_estimate"]["xyz"], expected_xyz)

    def test_drift_rpy(self, contract):
        shift = ShiftSpec(
            start_step=0, end_step=None,
            shift_type="base_drift",
            params={"drift_rpy": [0.02, 0.0, 0.0]},
        )
        inj = ShiftInjector(contract, [shift])
        inputs = inj.get_monitor_inputs(0)
        expected_rpy = contract.robot_base_pose.rpy + np.array([0.02, 0, 0])
        np.testing.assert_allclose(inputs["base_pose_estimate"]["rpy"], expected_rpy)


class TestMultipleShifts:
    """Multiple simultaneous shifts."""

    def test_two_shifts_both_applied(self, contract):
        anchor_name = contract.anchors[0].name
        io_monitor = None
        for m in contract.monitors:
            if m.type == "safety_io_health":
                io_monitor = m
                break

        shifts = [
            ShiftSpec(
                start_step=0, end_step=None,
                shift_type="anchor_drift",
                params={"anchor": anchor_name, "drift_xyz": [0.05, 0, 0]},
            ),
            ShiftSpec(
                start_step=0, end_step=None,
                shift_type="safety_io_flip",
                params={"signal": io_monitor.params["signal"]},
            ),
        ]
        inj = ShiftInjector(contract, shifts)
        inputs = inj.get_monitor_inputs(0)

        # Both shifts applied
        expected_xyz = contract.anchors[0].expected_pose.xyz + np.array([0.05, 0, 0])
        np.testing.assert_allclose(inputs["anchor_poses"][anchor_name]["xyz"], expected_xyz)
        assert inputs["safety_io"][io_monitor.params["signal"]] == (not io_monitor.params["expected"])


class TestSnapshot:
    """Snapshot round-trip."""

    def test_round_trip(self, contract):
        shifts = [
            ShiftSpec(start_step=10, end_step=50, shift_type="anchor_drift",
                      params={"anchor": contract.anchors[0].name, "drift_xyz": [0.1, 0, 0]}),
        ]
        inj = ShiftInjector(contract, shifts)
        inj.get_monitor_inputs(25)
        snap = inj.snapshot()
        assert snap["step_counter"] == 25
        assert len(snap["schedule"]) == 1
        assert snap["schedule"][0]["shift_type"] == "anchor_drift"


class TestReset:
    """Per-episode reset."""

    def test_reset_clears_step(self, contract):
        inj = ShiftInjector(contract)
        inj.get_monitor_inputs(50)
        inj.reset_episode()
        assert inj._step_counter == 0


class TestShiftValidation:
    """Invalid shifts must fail before they can silently omit monitor evidence."""

    @pytest.mark.parametrize(
        "shift",
        [
            ShiftSpec(0, None, "anchor_drift", {"anchor": "unknown", "drift_xyz": [0, 0, 0]}),
            ShiftSpec(0, None, "safety_io_flip", {"signal": "unknown"}),
            ShiftSpec(0, None, "sensor_stale", {"frame_age_ms": np.inf}),
            ShiftSpec(0, None, "base_drift", {"drift_xyz": [0, 0]}),
            ShiftSpec(0, None, "sensor_stale", {"ignored": 1}),
        ],
    )
    def test_rejects_unknown_or_malformed_shift(self, contract, shift):
        with pytest.raises(ShiftValidationError):
            ShiftInjector(contract, [shift])

    def test_rejects_nonfinite_shift_for_known_target(self, contract):
        shift = ShiftSpec(
            0,
            None,
            "anchor_drift",
            {"anchor": contract.anchors[0].name, "drift_xyz": [np.nan, 0, 0]},
        )
        with pytest.raises(ShiftValidationError, match="finite"):
            ShiftInjector(contract, [shift])
