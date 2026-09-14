"""Monitor unit tests.

Tests debounce/hysteresis timing, predicate logic, and per-episode reset.
"""

import numpy as np
import pytest

from safety_kernel.contract.schema import MonitorConfig, Anchor, Pose
from safety_kernel.contract.monitors import (
    AnchorPoseConsistencyMonitor,
    SafetyIOHealthMonitor,
    SensorHealthMonitor,
    RobotMountIntegrityMonitor,
    Monitor,
    create_monitors,
)

LOOP_RATE_HZ = 50.0  # 50 Hz → dt = 20ms


def _make_config(
    name: str = "test",
    type: str = "safety_io_health",
    debounce_ms: float = 0,
    hysteresis_ms: float = 0,
    severity: str = "rta_stop",
    params: dict | None = None,
) -> MonitorConfig:
    return MonitorConfig(
        name=name,
        type=type,
        inputs=["test"],
        params=params or {"signal": "test", "expected": True},
        debounce_ms=debounce_ms,
        hysteresis_ms=hysteresis_ms,
        severity=severity,
        action="safe_stop",
    )


class TestDebounceHysteresisConversion:
    """Verify tick conversion from ms at 50Hz."""

    def test_zero_debounce(self):
        config = _make_config(debounce_ms=0)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        assert m.debounce_ticks == 0

    def test_250ms_debounce(self):
        """250ms at 50Hz → ceil(250 * 50/1000) = ceil(12.5) = 13 ticks."""
        config = _make_config(debounce_ms=250)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        assert m.debounce_ticks == 13

    def test_500ms_debounce(self):
        """500ms at 50Hz → ceil(500 * 50/1000) = 25 ticks."""
        config = _make_config(debounce_ms=500)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        assert m.debounce_ticks == 25

    def test_500ms_hysteresis(self):
        config = _make_config(hysteresis_ms=500)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        assert m.hysteresis_ticks == 25

    def test_1000ms_hysteresis(self):
        config = _make_config(hysteresis_ms=1000)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        assert m.hysteresis_ticks == 50


class TestSafetyIOHealth:
    """safety_io_health monitor: binary signal mismatch."""

    def test_ok_signal(self):
        config = _make_config(params={"signal": "estop_ok", "expected": True})
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        result = m.evaluate({"safety_io": {"estop_ok": True}})
        assert not result.predicate_violated
        assert not result.triggered
        assert not result.should_trigger

    def test_violated_signal(self):
        config = _make_config(params={"signal": "estop_ok", "expected": True})
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        result = m.evaluate({"safety_io": {"estop_ok": False}})
        assert result.predicate_violated
        assert result.triggered  # 0 debounce → immediate
        assert result.should_trigger

    def test_immediate_trigger_zero_debounce(self):
        """With 0ms debounce, trigger on first violation."""
        config = _make_config(debounce_ms=0)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        result = m.evaluate({"safety_io": {"test": False}})
        assert result.should_trigger
        assert result.triggered

    def test_immediate_clear_zero_hysteresis(self):
        """With 0ms hysteresis, clear on first OK after trigger."""
        config = _make_config(debounce_ms=0, hysteresis_ms=0)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)

        # Trigger
        m.evaluate({"safety_io": {"test": False}})
        assert m.triggered

        # Clear
        result = m.evaluate({"safety_io": {"test": True}})
        assert result.should_clear
        assert not result.triggered


class TestDebounceExactTiming:
    """Step-exact debounce timing tests."""

    def test_250ms_debounce_triggers_at_step_13(self):
        """250ms at 50Hz: 13 ticks debounce. Trigger at tick 14 (0-indexed: step 13)."""
        config = _make_config(debounce_ms=250)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        assert m.debounce_ticks == 13

        # Steps 0-12: violated but not yet triggered
        for step in range(13):
            result = m.evaluate({"safety_io": {"test": False}})
            assert result.predicate_violated
            assert not result.should_trigger, f"Premature trigger at step {step}"
            assert not result.triggered

        # Step 13: trigger fires (14th consecutive violation)
        result = m.evaluate({"safety_io": {"test": False}})
        assert result.should_trigger
        assert result.triggered

    def test_debounce_resets_on_ok(self):
        """If violation clears during debounce, counter resets."""
        config = _make_config(debounce_ms=250)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)

        # 10 violations
        for _ in range(10):
            m.evaluate({"safety_io": {"test": False}})
        assert not m.triggered

        # 1 OK → resets counter
        m.evaluate({"safety_io": {"test": True}})

        # Need full 14 more violations
        for step in range(13):
            result = m.evaluate({"safety_io": {"test": False}})
            assert not result.should_trigger

        result = m.evaluate({"safety_io": {"test": False}})
        assert result.should_trigger


class TestHysteresisExactTiming:
    """Step-exact hysteresis timing tests."""

    def test_500ms_hysteresis_clears_at_step_25(self):
        """500ms at 50Hz: 25 ticks hysteresis. Clear at tick 26 after last trigger."""
        config = _make_config(debounce_ms=0, hysteresis_ms=500)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)

        # Trigger
        m.evaluate({"safety_io": {"test": False}})
        assert m.triggered

        # Steps 0-24 OK: still triggered
        for step in range(25):
            result = m.evaluate({"safety_io": {"test": True}})
            assert not result.should_clear, f"Premature clear at step {step}"
            assert result.triggered

        # Step 25: clear fires (26th consecutive OK)
        result = m.evaluate({"safety_io": {"test": True}})
        assert result.should_clear
        assert not result.triggered

    def test_hysteresis_resets_on_violation(self):
        """If violation recurs during hysteresis, OK counter resets."""
        config = _make_config(debounce_ms=0, hysteresis_ms=500)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)

        # Trigger
        m.evaluate({"safety_io": {"test": False}})
        assert m.triggered

        # 20 OK ticks
        for _ in range(20):
            m.evaluate({"safety_io": {"test": True}})
        assert m.triggered  # Not cleared yet

        # Violation again → resets OK counter
        m.evaluate({"safety_io": {"test": False}})

        # Need full 26 more OK ticks
        for step in range(25):
            result = m.evaluate({"safety_io": {"test": True}})
            assert not result.should_clear

        result = m.evaluate({"safety_io": {"test": True}})
        assert result.should_clear


class TestAnchorPoseConsistency:
    """anchor_pose_consistency monitor tests."""

    def _make_monitor(self, debounce_ms=0):
        config = _make_config(
            type="anchor_pose_consistency",
            debounce_ms=debounce_ms,
            severity="revoke",
            params={
                "anchors": ["fid_a", "fid_b"],
                "aggregation": "any_required_out_of_tolerance",
            },
        )
        anchors = (
            Anchor(
                name="fid_a", type="fiducial", sensor="cam",
                expected_pose=Pose(xyz=np.array([1.0, 0.0, 0.0]), rpy=np.zeros(3)),
                pos_tol=0.01, rot_tol=0.03, required=True,
            ),
            Anchor(
                name="fid_b", type="fiducial", sensor="cam",
                expected_pose=Pose(xyz=np.array([0.0, 1.0, 0.0]), rpy=np.zeros(3)),
                pos_tol=0.01, rot_tol=0.03, required=True,
            ),
        )
        return AnchorPoseConsistencyMonitor(config, LOOP_RATE_HZ, anchors)

    def test_both_anchors_ok(self):
        m = self._make_monitor()
        inputs = {
            "anchor_poses": {
                "fid_a": {"xyz": [1.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
                "fid_b": {"xyz": [0.0, 1.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            }
        }
        result = m.evaluate(inputs)
        assert not result.predicate_violated

    def test_one_anchor_pos_drift(self):
        m = self._make_monitor()
        inputs = {
            "anchor_poses": {
                "fid_a": {"xyz": [1.02, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},  # 0.02 > 0.01
                "fid_b": {"xyz": [0.0, 1.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            }
        }
        result = m.evaluate(inputs)
        assert result.predicate_violated

    def test_one_anchor_rot_drift(self):
        m = self._make_monitor()
        inputs = {
            "anchor_poses": {
                "fid_a": {"xyz": [1.0, 0.0, 0.0], "rpy": [0.04, 0.0, 0.0]},  # 0.04 > 0.03
                "fid_b": {"xyz": [0.0, 1.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            }
        }
        result = m.evaluate(inputs)
        assert result.predicate_violated

    def test_missing_anchor(self):
        m = self._make_monitor()
        inputs = {
            "anchor_poses": {
                "fid_a": {"xyz": [1.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
                # fid_b missing
            }
        }
        result = m.evaluate(inputs)
        assert result.predicate_violated


class TestSensorHealth:
    """sensor_health monitor tests."""

    def _make_monitor(self, debounce_ms=0):
        config = _make_config(
            type="sensor_health",
            debounce_ms=debounce_ms,
            severity="degrade",
            params={
                "max_frame_age_ms": 100,
                "monotonic_timestamps": True,
                "calibration_checksum": "SIM",
            },
        )
        return SensorHealthMonitor(config, LOOP_RATE_HZ)

    def test_healthy_sensor(self):
        m = self._make_monitor()
        result = m.evaluate({"sensor_health": {
            "frame_age_ms": 10.0,
            "timestamp": 1.0,
        }})
        assert not result.predicate_violated

    def test_stale_frame(self):
        m = self._make_monitor()
        result = m.evaluate({"sensor_health": {
            "frame_age_ms": 200.0,  # > 100
            "timestamp": 1.0,
        }})
        assert result.predicate_violated

    def test_non_monotonic_timestamp(self):
        m = self._make_monitor()
        # First call OK
        m.evaluate({"sensor_health": {"frame_age_ms": 10.0, "timestamp": 100.0}})
        # Second call with earlier timestamp
        result = m.evaluate({"sensor_health": {"frame_age_ms": 10.0, "timestamp": 50.0}})
        assert result.predicate_violated

    def test_rollback_does_not_replace_last_good_timestamp(self):
        m = self._make_monitor()
        assert not m.evaluate({"sensor_health": {"frame_age_ms": 10.0, "timestamp": 2.0}}).predicate_violated
        assert m.evaluate({"sensor_health": {"frame_age_ms": 10.0, "timestamp": 1.0}}).predicate_violated
        assert m.evaluate({"sensor_health": {"frame_age_ms": 10.0, "timestamp": 1.5}}).predicate_violated
        assert not m.evaluate({"sensor_health": {"frame_age_ms": 10.0, "timestamp": 2.1}}).predicate_violated

    def test_reset_clears_timestamp(self):
        m = self._make_monitor()
        m.evaluate({"sensor_health": {"frame_age_ms": 10.0, "timestamp": 100.0}})
        m.reset_episode()
        # After reset, any timestamp should be OK (no prev to compare)
        result = m.evaluate({"sensor_health": {"frame_age_ms": 10.0, "timestamp": 1.0}})
        assert not result.predicate_violated


class TestRobotMountIntegrity:
    """robot_mount_integrity monitor tests."""

    def _make_monitor(self, debounce_ms=0):
        config = _make_config(
            type="robot_mount_integrity",
            debounce_ms=debounce_ms,
            severity="revoke",
            params={
                "base_pose_drift_pos_m": 0.005,
                "base_pose_drift_rot_rad": 0.01,
            },
        )
        expected_pose = Pose(xyz=np.zeros(3), rpy=np.zeros(3))
        return RobotMountIntegrityMonitor(config, LOOP_RATE_HZ, expected_pose)

    def test_no_drift(self):
        m = self._make_monitor()
        result = m.evaluate({"base_pose_estimate": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}})
        assert not result.predicate_violated

    def test_pos_drift_exceeded(self):
        m = self._make_monitor()
        result = m.evaluate({"base_pose_estimate": {"xyz": [0.01, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}})
        assert result.predicate_violated  # 0.01 > 0.005

    def test_rot_drift_exceeded(self):
        m = self._make_monitor()
        result = m.evaluate({"base_pose_estimate": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.02, 0.0, 0.0]}})
        assert result.predicate_violated  # 0.02 > 0.01

    def test_rolling_window_clears(self):
        m = self._make_monitor()
        # One bad reading
        m.evaluate({"base_pose_estimate": {"xyz": [0.01, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}})
        # 50 good readings to push bad one out of window
        for _ in range(50):
            result = m.evaluate({"base_pose_estimate": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}})
        assert not result.predicate_violated


class TestMonitorReset:
    """Per-episode reset behavior."""

    def test_reset_clears_triggered(self):
        config = _make_config(debounce_ms=0)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        m.evaluate({"safety_io": {"test": False}})
        assert m.triggered
        m.reset_episode()
        assert not m.triggered

    def test_reset_clears_counters(self):
        config = _make_config(debounce_ms=250)
        m = SafetyIOHealthMonitor(config, LOOP_RATE_HZ)
        for _ in range(10):
            m.evaluate({"safety_io": {"test": False}})
        m.reset_episode()
        # After reset, need full debounce again
        for step in range(13):
            result = m.evaluate({"safety_io": {"test": False}})
            assert not result.should_trigger
        result = m.evaluate({"safety_io": {"test": False}})
        assert result.should_trigger


class TestCreateMonitors:
    """create_monitors factory from contract."""

    def test_creates_from_track_a(self):
        from safety_kernel.contract.loader import load_contract_file
        import pathlib
        contract_path = pathlib.Path(__file__).resolve().parents[1] / "contracts" / "cell_arm_track_a.yaml"
        lc = load_contract_file(contract_path)
        monitors = create_monitors(lc.contract, LOOP_RATE_HZ)
        assert len(monitors) == 5
        names = {m.name for m in monitors}
        assert "anchors_consistent" in names
        assert "gate_interlock_ok" in names
        assert "front_cam_health" in names
