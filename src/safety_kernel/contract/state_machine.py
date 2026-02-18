"""Contract state machine with flap detection.

Implements Section 4 of the cumulative spec.
States: ACTIVE, DEGRADED, REVOKED, RTA_STOP.
12-row transition table with hysteresis guard and flap limiter.

Event dicts use exact Appendix C.2 field names so Gate 2 is
"write to file + hash chain," not "rename schema."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContractState(Enum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    REVOKED = "REVOKED"
    RTA_STOP = "RTA_STOP"


# Trigger types
TRIGGER_DEGRADE = "degrade"
TRIGGER_REVOKE = "revoke"
TRIGGER_RTA_STOP = "rta_stop"
TRIGGER_L0R_NEARMISS = "l0r_nearmiss"
TRIGGER_CLEAR = "clear"
TRIGGER_RESET = "reset"

# Severity values (match MonitorConfig.severity and Appendix C.2)
SEVERITY_DEGRADE = "degrade"
SEVERITY_REVOKE = "revoke"
SEVERITY_RTA_STOP = "rta_stop"
SEVERITY_CLEAR = "clear"
SEVERITY_RESET = "reset"

# 12-row transition table (Section 4.1, cumulative spec source of truth).
_TRANSITIONS: dict[tuple[ContractState, str], ContractState] = {
    (ContractState.ACTIVE, TRIGGER_DEGRADE): ContractState.DEGRADED,
    (ContractState.ACTIVE, TRIGGER_REVOKE): ContractState.REVOKED,
    (ContractState.ACTIVE, TRIGGER_RTA_STOP): ContractState.RTA_STOP,
    (ContractState.ACTIVE, TRIGGER_L0R_NEARMISS): ContractState.RTA_STOP,
    (ContractState.DEGRADED, TRIGGER_REVOKE): ContractState.REVOKED,
    (ContractState.DEGRADED, TRIGGER_CLEAR): ContractState.ACTIVE,
    (ContractState.DEGRADED, TRIGGER_RTA_STOP): ContractState.RTA_STOP,
    (ContractState.DEGRADED, TRIGGER_L0R_NEARMISS): ContractState.RTA_STOP,
    (ContractState.REVOKED, TRIGGER_RTA_STOP): ContractState.RTA_STOP,
    (ContractState.REVOKED, TRIGGER_L0R_NEARMISS): ContractState.RTA_STOP,
    (ContractState.REVOKED, TRIGGER_RESET): ContractState.ACTIVE,
    (ContractState.RTA_STOP, TRIGGER_RESET): ContractState.ACTIVE,
}

MAX_FLAP_PAIRS = 4


class ContractStateMachine:
    """Contract state machine with flap detection (Section 4).

    Per-episode lifecycle: call reset_episode() at episode boundary.
    """

    def __init__(self, loop_rate_hz: float, max_flap_pairs: int = MAX_FLAP_PAIRS):
        self._loop_rate_hz = loop_rate_hz
        self._max_flap_pairs = max_flap_pairs
        self._state = ContractState.ACTIVE
        self._step: int = 0

        # Flap tracking
        self._flap_count: int = 0
        self._recovery_guard_end: int | None = None

        self._pending_events: list[dict[str, Any]] = []

    @property
    def state(self) -> ContractState:
        return self._state

    @property
    def flap_count(self) -> int:
        return self._flap_count

    def reset_episode(self) -> None:
        """Reset for new episode."""
        self._state = ContractState.ACTIVE
        self._step = 0
        self._flap_count = 0
        self._recovery_guard_end = None
        self._pending_events.clear()

    def process_monitor_events(
        self,
        step: int,
        monitor_results: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Process all monitor results for one step.

        Args:
            step: Current simulation step.
            monitor_results: {monitor_name: {"triggered": bool, "should_trigger": bool,
                             "should_clear": bool, "severity": str, "predicate_violated": bool}}

        Returns:
            List of contract_state_change event dicts (Appendix C.2).
        """
        self._step = step
        self._pending_events.clear()
        monitors_status = _build_monitors_status(monitor_results)

        # Determine highest-priority trigger this step
        rta_stop_trigger = None
        revoke_trigger = None
        degrade_trigger = None
        clear_candidates: list[str] = []

        for name, result in monitor_results.items():
            severity = result["severity"]
            if result.get("should_trigger"):
                if severity == SEVERITY_RTA_STOP:
                    rta_stop_trigger = name
                elif severity == SEVERITY_REVOKE:
                    revoke_trigger = name
                elif severity == SEVERITY_DEGRADE:
                    degrade_trigger = name
            if result.get("should_clear") and severity == SEVERITY_DEGRADE:
                clear_candidates.append(name)

        # Apply in priority order (strict=False: no-op if already in terminal state)
        if rta_stop_trigger is not None:
            self._do_transition(
                TRIGGER_RTA_STOP, rta_stop_trigger, SEVERITY_RTA_STOP,
                step, monitors_status, strict=False,
            )
        elif revoke_trigger is not None:
            self._do_transition(
                TRIGGER_REVOKE, revoke_trigger, SEVERITY_REVOKE,
                step, monitors_status, strict=False,
            )
        elif degrade_trigger is not None:
            self._do_transition(
                TRIGGER_DEGRADE, degrade_trigger, SEVERITY_DEGRADE,
                step, monitors_status, strict=False,
            )

        # Clear check: ALL degrade monitors must be non-triggered
        if (
            self._state == ContractState.DEGRADED
            and clear_candidates
            and self._all_degrade_monitors_clear(monitor_results)
        ):
            self._do_transition(
                TRIGGER_CLEAR, clear_candidates[-1], SEVERITY_CLEAR,
                step, monitors_status,
            )

        return list(self._pending_events)

    def notify_l0r_nearmiss(
        self,
        step: int,
        monitors_status: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """L0-R near-miss → RTA_STOP from any state.

        No-op if already in RTA_STOP (already in safest state).
        """
        self._step = step
        self._pending_events.clear()
        if self._state == ContractState.RTA_STOP:
            return []
        self._do_transition(
            TRIGGER_L0R_NEARMISS, "l0r_nearmiss", SEVERITY_RTA_STOP,
            step, monitors_status,
        )
        return list(self._pending_events)

    def operator_reset(
        self,
        step: int,
        monitors_status: dict[str, Any],
        all_monitors_passing: bool,
    ) -> list[dict[str, Any]]:
        """Operator reset from REVOKED or RTA_STOP.

        Requires all monitors passing. Raises if called from wrong state.
        Returns empty list if rejected (monitors failing).
        """
        self._step = step
        self._pending_events.clear()

        if self._state not in (ContractState.REVOKED, ContractState.RTA_STOP):
            raise IllegalTransitionError(
                f"Operator reset illegal from state {self._state.value}"
            )
        if not all_monitors_passing:
            return []

        self._do_transition(
            TRIGGER_RESET, "operator_reset", SEVERITY_RESET,
            step, monitors_status,
        )
        return list(self._pending_events)

    def set_recovery_guard(self, recovery_step: int, hysteresis_ticks: int) -> None:
        """Set flap recovery guard after DEGRADED→ACTIVE.

        Called by the integration layer with the clearing monitor's
        hysteresis_ticks. Flap = re-degradation before guard_end.
        """
        self._recovery_guard_end = recovery_step + hysteresis_ticks

    def snapshot(self) -> dict[str, Any]:
        """Serialize state for persistence/resumability."""
        return {
            "state": self._state.value,
            "step": self._step,
            "flap_count": self._flap_count,
            "recovery_guard_end": self._recovery_guard_end,
        }

    @classmethod
    def from_snapshot(
        cls, data: dict[str, Any], loop_rate_hz: float,
    ) -> ContractStateMachine:
        """Restore from snapshot."""
        sm = cls(loop_rate_hz)
        sm._state = ContractState(data["state"])
        sm._step = data["step"]
        sm._flap_count = data["flap_count"]
        sm._recovery_guard_end = data["recovery_guard_end"]
        return sm

    # ---- internals ----

    def _do_transition(
        self,
        trigger: str,
        trigger_monitor: str,
        trigger_severity: str,
        step: int,
        monitors_status: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        """Execute a transition. Validates against table, handles flaps.

        If strict=False, silently returns on missing transition (idempotent).
        If strict=True, raises IllegalTransitionError.
        """
        key = (self._state, trigger)
        to_state = _TRANSITIONS.get(key)
        if to_state is None:
            if not strict:
                return
            raise IllegalTransitionError(
                f"No transition from {self._state.value} on trigger {trigger!r}"
            )

        from_state = self._state

        # ACTIVE→DEGRADED: check flap guard
        if from_state == ContractState.ACTIVE and to_state == ContractState.DEGRADED:
            if self._is_flap(step):
                self._flap_count += 1
                if self._flap_count >= self._max_flap_pairs:
                    # Flap limiter: emit A→D then D→RTA_STOP
                    self._state = ContractState.DEGRADED
                    self._pending_events.append(_make_event(
                        from_state, ContractState.DEGRADED,
                        trigger_monitor, trigger_severity,
                        step, monitors_status,
                    ))
                    self._state = ContractState.RTA_STOP
                    self._pending_events.append(_make_event(
                        ContractState.DEGRADED, ContractState.RTA_STOP,
                        "flap_limiter", SEVERITY_RTA_STOP,
                        step, monitors_status,
                    ))
                    return

        # Normal transition
        self._state = to_state
        self._pending_events.append(_make_event(
            from_state, to_state, trigger_monitor, trigger_severity,
            step, monitors_status,
        ))

        # DEGRADED→ACTIVE: record recovery step (guard set by integration layer)
        if from_state == ContractState.DEGRADED and to_state == ContractState.ACTIVE:
            # Default guard: just the recovery step. Integration layer
            # should call set_recovery_guard() with the monitor's hysteresis.
            self._recovery_guard_end = step

    def _is_flap(self, step: int) -> bool:
        """Is this ACTIVE→DEGRADED within the recovery guard window?"""
        return (
            self._recovery_guard_end is not None
            and step < self._recovery_guard_end
        )

    @staticmethod
    def _all_degrade_monitors_clear(
        monitor_results: dict[str, dict[str, Any]],
    ) -> bool:
        """All degrade-severity monitors must be non-triggered."""
        for result in monitor_results.values():
            if result["severity"] == SEVERITY_DEGRADE and result.get("triggered", False):
                return False
        return True


def _build_monitors_status(
    monitor_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build monitors_status snapshot for Appendix C.2."""
    return {
        name: {
            "triggered": result.get("triggered", False),
            "violated": result.get("predicate_violated", False),
        }
        for name, result in monitor_results.items()
    }


def _make_event(
    from_state: ContractState,
    to_state: ContractState,
    trigger_monitor: str,
    trigger_severity: str,
    step: int,
    monitors_status: dict[str, Any],
) -> dict[str, Any]:
    """Appendix C.2 contract_state_change event dict.

    timestamp_ns, episode_id, prev_hash filled by telemetry layer.
    """
    return {
        "event_type": "contract_state_change",
        "timestamp_ns": None,
        "episode_id": None,
        "step": step,
        "prev_hash": None,
        "from_state": from_state.value,
        "to_state": to_state.value,
        "trigger_monitor": trigger_monitor,
        "trigger_severity": trigger_severity,
        "monitors_status": monitors_status,
    }


class IllegalTransitionError(Exception):
    """Raised when a transition is not in the transition table."""
    pass
