"""Shift injector: deterministic fault injection for SLA testing.

Owns both the "perfect baseline" (contract expected values) and
deterministic drift injection. Produces monitor inputs directly.

Shift types:
  - anchor_drift: adds drift_xyz to specified anchor's observed pose
  - safety_io_flip: inverts specified signal (gate_closed, estop_ok)
  - sensor_stale: sets frame_age_ms to large value or breaks monotonicity
  - base_drift: adds drift_xyz, drift_rpy to base pose estimate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from safety_kernel.contract.schema import Contract


@dataclass
class ShiftSpec:
    """A single fault injection specification."""
    start_step: int
    end_step: int | None  # None = until episode end
    shift_type: str       # "anchor_drift", "safety_io_flip", "sensor_stale", "base_drift"
    params: dict[str, Any] = field(default_factory=dict)


class ShiftInjector:
    """Deterministic fault injector for monitor SLA testing.

    Without shifts, get_monitor_inputs() returns the perfect baseline
    matching contract expectations. With shifts, affected channels are
    modified deterministically.

    Per-episode lifecycle: call reset_episode() at episode boundary.
    """

    def __init__(
        self,
        contract: Contract,
        schedule: list[ShiftSpec] | None = None,
    ):
        self._contract = contract
        self._schedule = list(schedule) if schedule else []
        self._step_counter: int = 0

    def reset_episode(self) -> None:
        """Reset for new episode."""
        self._step_counter = 0

    def get_monitor_inputs(self, step: int) -> dict[str, Any]:
        """Return monitor inputs: perfect baseline + active shifts.

        Args:
            step: Current simulation step.

        Returns:
            Dict with keys matching what each monitor expects:
            - "anchor_poses": {name: {"xyz": [...], "rpy": [...]}}
            - "safety_io": {signal_name: bool}
            - "sensor_health": {"frame_age_ms": float, "timestamp": float}
            - "base_pose_estimate": {"xyz": [...], "rpy": [...]}
        """
        self._step_counter = step
        inputs = self._perfect_baseline(step)

        for shift in self._active_shifts(step):
            self._apply_shift(inputs, shift, step)

        return inputs

    def get_active_shifts(self, step: int) -> list[ShiftSpec]:
        """Return list of shifts active at this step."""
        return self._active_shifts(step)

    def snapshot(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "step_counter": self._step_counter,
            "schedule": [
                {
                    "start_step": s.start_step,
                    "end_step": s.end_step,
                    "shift_type": s.shift_type,
                    "params": s.params,
                }
                for s in self._schedule
            ],
        }

    # ---- internals ----

    def _active_shifts(self, step: int) -> list[ShiftSpec]:
        """Filter schedule to shifts active at this step."""
        active = []
        for s in self._schedule:
            if step < s.start_step:
                continue
            if s.end_step is not None and step >= s.end_step:
                continue
            active.append(s)
        return active

    def _perfect_baseline(self, step: int) -> dict[str, Any]:
        """Build perfect (no-fault) monitor inputs from contract."""
        c = self._contract

        # Anchor poses: exactly at expected positions
        anchor_poses = {}
        for anchor in c.anchors:
            anchor_poses[anchor.name] = {
                "xyz": anchor.expected_pose.xyz.tolist(),
                "rpy": anchor.expected_pose.rpy.tolist(),
            }

        # Safety IO: all signals at expected values
        safety_io = {}
        for monitor in c.monitors:
            if monitor.type == "safety_io_health":
                signal = monitor.params.get("signal", "")
                expected = monitor.params.get("expected", True)
                safety_io[signal] = expected

        # Sensor health: fresh frame, monotonic timestamp
        dt_ms = 1000.0 / 50.0  # 20ms at 50Hz
        sensor_health = {
            "frame_age_ms": 5.0,  # Well within any max_frame_age_ms
            "timestamp": float(step) * dt_ms,  # Monotonically increasing
        }

        # Base pose: exactly at expected position
        base_pose_estimate = {
            "xyz": c.robot_base_pose.xyz.tolist(),
            "rpy": c.robot_base_pose.rpy.tolist(),
        }

        return {
            "anchor_poses": anchor_poses,
            "safety_io": safety_io,
            "sensor_health": sensor_health,
            "base_pose_estimate": base_pose_estimate,
        }

    @staticmethod
    def _apply_shift(
        inputs: dict[str, Any],
        shift: ShiftSpec,
        step: int,
    ) -> None:
        """Apply a single shift to the inputs dict (in-place)."""
        if shift.shift_type == "anchor_drift":
            _apply_anchor_drift(inputs, shift.params)
        elif shift.shift_type == "safety_io_flip":
            _apply_safety_io_flip(inputs, shift.params)
        elif shift.shift_type == "sensor_stale":
            _apply_sensor_stale(inputs, shift.params)
        elif shift.shift_type == "base_drift":
            _apply_base_drift(inputs, shift.params)
        else:
            raise ValueError(f"Unknown shift type: {shift.shift_type!r}")


def _apply_anchor_drift(inputs: dict, params: dict) -> None:
    """Add drift_xyz to specified anchor's observed pose."""
    anchor_name = params["anchor"]
    drift_xyz = np.array(params["drift_xyz"])
    poses = inputs.get("anchor_poses", {})
    if anchor_name in poses:
        xyz = np.array(poses[anchor_name]["xyz"])
        poses[anchor_name]["xyz"] = (xyz + drift_xyz).tolist()


def _apply_safety_io_flip(inputs: dict, params: dict) -> None:
    """Invert specified signal."""
    signal = params["signal"]
    io = inputs.get("safety_io", {})
    if signal in io:
        io[signal] = not io[signal]


def _apply_sensor_stale(inputs: dict, params: dict) -> None:
    """Set frame_age_ms to large value or break monotonicity."""
    sensor = inputs.get("sensor_health", {})
    if "frame_age_ms" in params:
        sensor["frame_age_ms"] = params["frame_age_ms"]
    if "timestamp_override" in params:
        sensor["timestamp"] = params["timestamp_override"]


def _apply_base_drift(inputs: dict, params: dict) -> None:
    """Add drift to base pose estimate."""
    base = inputs.get("base_pose_estimate", {})
    if "drift_xyz" in params:
        xyz = np.array(base.get("xyz", [0, 0, 0]))
        base["xyz"] = (xyz + np.array(params["drift_xyz"])).tolist()
    if "drift_rpy" in params:
        rpy = np.array(base.get("rpy", [0, 0, 0]))
        base["rpy"] = (rpy + np.array(params["drift_rpy"])).tolist()
