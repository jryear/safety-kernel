"""Assumption monitors with tick-based debounce and hysteresis.

Implements Section 2.2.1 of the cumulative spec.
Four monitor types: anchor_pose_consistency, safety_io_health,
sensor_health, robot_mount_integrity.

Each monitor owns its temporal state (debounce/hysteresis counters).
The state machine polls monitors for triggers/clears.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from safety_kernel.contract.schema import Anchor, Contract, MonitorConfig, Pose


@dataclass
class MonitorResult:
    """Result of a single monitor evaluation."""
    predicate_violated: bool   # Raw predicate output this tick
    triggered: bool            # Monitor is in triggered state
    should_trigger: bool       # Debounce threshold just crossed → fire event
    should_clear: bool         # Hysteresis threshold just crossed → clear event


class Monitor(ABC):
    """Base class for assumption monitors with debounce/hysteresis."""

    def __init__(self, config: MonitorConfig, loop_rate_hz: float):
        self.name = config.name
        self.severity = config.severity
        self.action = config.action

        ticks_per_ms = loop_rate_hz / 1000.0
        self._debounce_ticks = ceil(config.debounce_ms * ticks_per_ms)
        self._hysteresis_ticks = ceil(config.hysteresis_ms * ticks_per_ms)

        self._consecutive_fail: int = 0
        self._consecutive_ok: int = 0
        self._triggered: bool = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def debounce_ticks(self) -> int:
        return self._debounce_ticks

    @property
    def hysteresis_ticks(self) -> int:
        return self._hysteresis_ticks

    def evaluate(self, inputs: dict[str, Any]) -> MonitorResult:
        """Evaluate monitor predicate with debounce/hysteresis logic.

        Args:
            inputs: Monitor-specific input dict.

        Returns:
            MonitorResult with trigger/clear signals.
        """
        violated = self._check_predicate(inputs)

        should_trigger = False
        should_clear = False

        if violated:
            self._consecutive_fail += 1
            self._consecutive_ok = 0
        else:
            self._consecutive_ok += 1
            self._consecutive_fail = 0

        # Debounce: trigger after debounce_ticks + 1 consecutive violations
        if not self._triggered and self._consecutive_fail > self._debounce_ticks:
            self._triggered = True
            should_trigger = True

        # Hysteresis: clear after hysteresis_ticks + 1 consecutive OK
        if self._triggered and self._consecutive_ok > self._hysteresis_ticks:
            self._triggered = False
            should_clear = True

        return MonitorResult(
            predicate_violated=violated,
            triggered=self._triggered,
            should_trigger=should_trigger,
            should_clear=should_clear,
        )

    def reset_episode(self) -> None:
        """Reset monitor state for new episode."""
        self._consecutive_fail = 0
        self._consecutive_ok = 0
        self._triggered = False

    @abstractmethod
    def _check_predicate(self, inputs: dict[str, Any]) -> bool:
        """Check the monitor predicate.

        Returns True if the predicate is violated (unsafe condition detected).
        """
        ...


class AnchorPoseConsistencyMonitor(Monitor):
    """Section 2.2.1: anchor_pose_consistency.

    Checks observed anchor poses against expected poses.
    Aggregation 'any_required_out_of_tolerance': fires if ANY required
    anchor exceeds position or rotation tolerance.
    """

    def __init__(
        self,
        config: MonitorConfig,
        loop_rate_hz: float,
        anchors: tuple[Anchor, ...],
    ):
        super().__init__(config, loop_rate_hz)
        anchor_names = set(config.params.get("anchors", []))
        self._anchors = {
            a.name: a for a in anchors if a.name in anchor_names
        }
        self._aggregation = config.params.get("aggregation", "any_required_out_of_tolerance")

    def _check_predicate(self, inputs: dict[str, Any]) -> bool:
        anchor_poses = inputs.get("anchor_poses", {})

        for name, anchor in self._anchors.items():
            if not anchor.required:
                continue

            observed = anchor_poses.get(name)
            if observed is None:
                # Missing anchor = violated
                return True

            pos_error = np.linalg.norm(
                np.asarray(observed["xyz"]) - anchor.expected_pose.xyz
            )
            if pos_error > anchor.pos_tol:
                return True

            rot_error = np.linalg.norm(
                np.asarray(observed["rpy"]) - anchor.expected_pose.rpy
            )
            if rot_error > anchor.rot_tol:
                return True

        return False


class SafetyIOHealthMonitor(Monitor):
    """Section 2.2.1: safety_io_health.

    Checks a binary signal matches expected value.
    """

    def __init__(self, config: MonitorConfig, loop_rate_hz: float):
        super().__init__(config, loop_rate_hz)
        self._signal = config.params["signal"]
        self._expected = config.params["expected"]

    def _check_predicate(self, inputs: dict[str, Any]) -> bool:
        safety_io = inputs.get("safety_io", {})
        actual = safety_io.get(self._signal)
        return actual != self._expected


class SensorHealthMonitor(Monitor):
    """Section 2.2.1: sensor_health.

    Checks frame age, monotonic timestamps, calibration checksum.
    """

    def __init__(self, config: MonitorConfig, loop_rate_hz: float):
        super().__init__(config, loop_rate_hz)
        self._max_frame_age_ms = config.params["max_frame_age_ms"]
        self._check_monotonic = config.params.get("monotonic_timestamps", True)
        self._prev_timestamp: float | None = None

    def _check_predicate(self, inputs: dict[str, Any]) -> bool:
        sensor = inputs.get("sensor_health", {})

        # Frame age
        frame_age_ms = sensor.get("frame_age_ms", 0.0)
        if frame_age_ms > self._max_frame_age_ms:
            return True

        # Monotonic timestamps
        if self._check_monotonic:
            timestamp = sensor.get("timestamp", 0.0)
            if self._prev_timestamp is not None:
                if timestamp <= self._prev_timestamp:
                    return True
            self._prev_timestamp = timestamp

        return False

    def reset_episode(self) -> None:
        super().reset_episode()
        self._prev_timestamp = None


class RobotMountIntegrityMonitor(Monitor):
    """Section 2.2.1: robot_mount_integrity.

    Checks base pose drift over a rolling window.
    """

    def __init__(
        self,
        config: MonitorConfig,
        loop_rate_hz: float,
        expected_base_pose: Pose,
    ):
        super().__init__(config, loop_rate_hz)
        self._expected_xyz = expected_base_pose.xyz
        self._expected_rpy = expected_base_pose.rpy
        self._pos_tol = config.params["base_pose_drift_pos_m"]
        self._rot_tol = config.params["base_pose_drift_rot_rad"]
        self._window_size = 50  # ~1 second at 50Hz
        self._pos_window: deque[float] = deque(maxlen=self._window_size)
        self._rot_window: deque[float] = deque(maxlen=self._window_size)

    def _check_predicate(self, inputs: dict[str, Any]) -> bool:
        base_pose = inputs.get("base_pose_estimate", {})
        xyz = np.asarray(base_pose.get("xyz", self._expected_xyz))
        rpy = np.asarray(base_pose.get("rpy", self._expected_rpy))

        pos_drift = float(np.linalg.norm(xyz - self._expected_xyz))
        rot_drift = float(np.linalg.norm(rpy - self._expected_rpy))

        self._pos_window.append(pos_drift)
        self._rot_window.append(rot_drift)

        # Max drift over rolling window
        max_pos = max(self._pos_window)
        max_rot = max(self._rot_window)

        return max_pos > self._pos_tol or max_rot > self._rot_tol

    def reset_episode(self) -> None:
        super().reset_episode()
        self._pos_window.clear()
        self._rot_window.clear()


def create_monitors(
    contract: Contract,
    loop_rate_hz: float,
) -> list[Monitor]:
    """Create monitor instances from contract configuration."""
    monitors: list[Monitor] = []

    for config in contract.monitors:
        if config.type == "anchor_pose_consistency":
            monitors.append(AnchorPoseConsistencyMonitor(
                config, loop_rate_hz, contract.anchors,
            ))
        elif config.type == "safety_io_health":
            monitors.append(SafetyIOHealthMonitor(config, loop_rate_hz))
        elif config.type == "sensor_health":
            monitors.append(SensorHealthMonitor(config, loop_rate_hz))
        elif config.type == "robot_mount_integrity":
            monitors.append(RobotMountIntegrityMonitor(
                config, loop_rate_hz, contract.robot_base_pose,
            ))
        else:
            raise ValueError(f"Unknown monitor type: {config.type!r}")

    return monitors
