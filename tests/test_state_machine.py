"""State machine tests.

Tests the 12-row transition table, flap detection, operator reset,
per-episode reset, snapshot round-trip, and event emission (Appendix C.2).
"""

import pytest

from safety_kernel.contract.state_machine import (
    ContractState,
    ContractStateMachine,
    IllegalTransitionError,
    TRIGGER_DEGRADE,
    TRIGGER_REVOKE,
    TRIGGER_RTA_STOP,
    TRIGGER_L0R_NEARMISS,
    TRIGGER_CLEAR,
    TRIGGER_RESET,
    SEVERITY_DEGRADE,
    SEVERITY_REVOKE,
    SEVERITY_RTA_STOP,
    SEVERITY_CLEAR,
    SEVERITY_RESET,
    MAX_FLAP_PAIRS,
    _make_event,
)

LOOP_RATE_HZ = 50.0


def _empty_status() -> dict:
    """Empty monitors_status for transitions that don't depend on it."""
    return {}


def _make_degrade_trigger(name: str = "test_monitor") -> dict:
    """Monitor result dict that should_trigger with degrade severity."""
    return {
        name: {
            "triggered": True,
            "should_trigger": True,
            "should_clear": False,
            "severity": SEVERITY_DEGRADE,
            "predicate_violated": True,
        }
    }


def _make_clear(name: str = "test_monitor") -> dict:
    """Monitor result dict that should_clear with degrade severity, not triggered."""
    return {
        name: {
            "triggered": False,
            "should_trigger": False,
            "should_clear": True,
            "severity": SEVERITY_DEGRADE,
            "predicate_violated": False,
        }
    }


def _make_revoke_trigger(name: str = "test_monitor") -> dict:
    return {
        name: {
            "triggered": True,
            "should_trigger": True,
            "should_clear": False,
            "severity": SEVERITY_REVOKE,
            "predicate_violated": True,
        }
    }


def _make_rta_stop_trigger(name: str = "test_monitor") -> dict:
    return {
        name: {
            "triggered": True,
            "should_trigger": True,
            "should_clear": False,
            "severity": SEVERITY_RTA_STOP,
            "predicate_violated": True,
        }
    }


def _make_quiet(name: str = "test_monitor", severity: str = SEVERITY_DEGRADE) -> dict:
    """Monitor that is not triggering, not clearing."""
    return {
        name: {
            "triggered": False,
            "should_trigger": False,
            "should_clear": False,
            "severity": severity,
            "predicate_violated": False,
        }
    }


class TestInitialState:
    def test_starts_active(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        assert sm.state == ContractState.ACTIVE

    def test_flap_count_zero(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        assert sm.flap_count == 0


class TestDeterministicTriggerSelection:
    """Same-severity simultaneous faults select by monitor name, not map order."""

    def test_equal_severity_uses_lexicographic_monitor_name(self):
        first = {
            **_make_revoke_trigger("zeta"),
            **_make_revoke_trigger("alpha"),
        }
        second = {
            **_make_revoke_trigger("alpha"),
            **_make_revoke_trigger("zeta"),
        }

        first_events = ContractStateMachine(LOOP_RATE_HZ).process_monitor_events(0, first)
        second_events = ContractStateMachine(LOOP_RATE_HZ).process_monitor_events(0, second)

        assert first_events[0]["trigger_monitor"] == "alpha"
        assert second_events[0]["trigger_monitor"] == "alpha"


class TestTransitionTable:
    """Test all 12 rows of the transition table from Section 4.1."""

    # ACTIVE → DEGRADED
    def test_active_degrade(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        events = sm.process_monitor_events(0, _make_degrade_trigger())
        assert sm.state == ContractState.DEGRADED
        assert len(events) == 1
        assert events[0]["from_state"] == "ACTIVE"
        assert events[0]["to_state"] == "DEGRADED"

    # ACTIVE → REVOKED
    def test_active_revoke(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        events = sm.process_monitor_events(0, _make_revoke_trigger())
        assert sm.state == ContractState.REVOKED
        assert events[0]["trigger_severity"] == SEVERITY_REVOKE

    # ACTIVE → RTA_STOP (rta_stop severity)
    def test_active_rta_stop(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        events = sm.process_monitor_events(0, _make_rta_stop_trigger())
        assert sm.state == ContractState.RTA_STOP

    # ACTIVE → RTA_STOP (L0-R near-miss)
    def test_active_l0r_nearmiss(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        events = sm.notify_l0r_nearmiss(0, _empty_status())
        assert sm.state == ContractState.RTA_STOP
        assert events[0]["trigger_monitor"] == "l0r_nearmiss"

    # DEGRADED → REVOKED
    def test_degraded_revoke(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        assert sm.state == ContractState.DEGRADED
        events = sm.process_monitor_events(1, _make_revoke_trigger())
        assert sm.state == ContractState.REVOKED
        assert events[0]["from_state"] == "DEGRADED"

    # DEGRADED → ACTIVE (clear)
    def test_degraded_clear(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        events = sm.process_monitor_events(1, _make_clear())
        assert sm.state == ContractState.ACTIVE
        assert events[0]["trigger_severity"] == SEVERITY_CLEAR

    # DEGRADED → RTA_STOP
    def test_degraded_rta_stop(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        events = sm.process_monitor_events(1, _make_rta_stop_trigger())
        assert sm.state == ContractState.RTA_STOP

    # DEGRADED → RTA_STOP (L0-R near-miss)
    def test_degraded_l0r_nearmiss(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        events = sm.notify_l0r_nearmiss(1, _empty_status())
        assert sm.state == ContractState.RTA_STOP

    # REVOKED → RTA_STOP
    def test_revoked_rta_stop(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_revoke_trigger())
        events = sm.process_monitor_events(1, _make_rta_stop_trigger())
        assert sm.state == ContractState.RTA_STOP
        assert events[0]["from_state"] == "REVOKED"

    # REVOKED → RTA_STOP (L0-R near-miss)
    def test_revoked_l0r_nearmiss(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_revoke_trigger())
        events = sm.notify_l0r_nearmiss(1, _empty_status())
        assert sm.state == ContractState.RTA_STOP

    # REVOKED → ACTIVE (operator reset)
    def test_revoked_operator_reset(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_revoke_trigger())
        events = sm.operator_reset(1, _empty_status(), all_monitors_passing=True)
        assert sm.state == ContractState.ACTIVE
        assert len(events) == 1
        assert events[0]["trigger_monitor"] == "operator_reset"
        assert events[0]["trigger_severity"] == SEVERITY_RESET

    # RTA_STOP → ACTIVE (operator reset)
    def test_rta_stop_operator_reset(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_rta_stop_trigger())
        events = sm.operator_reset(1, _empty_status(), all_monitors_passing=True)
        assert sm.state == ContractState.ACTIVE


class TestIllegalTransitions:
    """Transitions not in the table must raise."""

    def test_active_clear_illegal(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        # Attempt ACTIVE→ACTIVE via clear — not in table
        # This would require the clear path, but the state check prevents it
        # since we're in ACTIVE, the clear branch won't trigger.
        # Instead, test that operator_reset from ACTIVE raises.
        with pytest.raises(IllegalTransitionError):
            sm.operator_reset(0, _empty_status(), all_monitors_passing=True)

    def test_degraded_reset_illegal(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        with pytest.raises(IllegalTransitionError):
            sm.operator_reset(1, _empty_status(), all_monitors_passing=True)

    def test_active_reset_illegal(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        with pytest.raises(IllegalTransitionError):
            sm.operator_reset(0, _empty_status(), all_monitors_passing=True)


class TestOperatorResetRejection:
    """Operator reset requires all monitors passing."""

    def test_rejected_when_monitors_failing(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_revoke_trigger())
        events = sm.operator_reset(1, _empty_status(), all_monitors_passing=False)
        assert len(events) == 0
        assert sm.state == ContractState.REVOKED  # Unchanged

    def test_accepted_when_monitors_passing(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_revoke_trigger())
        events = sm.operator_reset(1, _empty_status(), all_monitors_passing=True)
        assert len(events) == 1
        assert sm.state == ContractState.ACTIVE


class TestClearRequiresAllDegradeMonitorsOK:
    """DEGRADED→ACTIVE requires ALL degrade monitors to be non-triggered."""

    def test_clear_blocked_by_other_degrade_monitor(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        # Two degrade monitors: sensor_a triggers, sensor_b quiet
        results = {
            "sensor_a": {
                "triggered": True, "should_trigger": True, "should_clear": False,
                "severity": SEVERITY_DEGRADE, "predicate_violated": True,
            },
            "sensor_b": {
                "triggered": False, "should_trigger": False, "should_clear": False,
                "severity": SEVERITY_DEGRADE, "predicate_violated": False,
            },
        }
        sm.process_monitor_events(0, results)
        assert sm.state == ContractState.DEGRADED

        # sensor_a clears, but sensor_b also triggers
        results2 = {
            "sensor_a": {
                "triggered": False, "should_trigger": False, "should_clear": True,
                "severity": SEVERITY_DEGRADE, "predicate_violated": False,
            },
            "sensor_b": {
                "triggered": True, "should_trigger": False, "should_clear": False,
                "severity": SEVERITY_DEGRADE, "predicate_violated": True,
            },
        }
        events = sm.process_monitor_events(1, results2)
        # Should stay DEGRADED because sensor_b is still triggered
        assert sm.state == ContractState.DEGRADED
        assert len(events) == 0

    def test_clear_allowed_when_all_clear(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        results = {
            "sensor_a": {
                "triggered": True, "should_trigger": True, "should_clear": False,
                "severity": SEVERITY_DEGRADE, "predicate_violated": True,
            },
        }
        sm.process_monitor_events(0, results)
        assert sm.state == ContractState.DEGRADED

        results2 = {
            "sensor_a": {
                "triggered": False, "should_trigger": False, "should_clear": True,
                "severity": SEVERITY_DEGRADE, "predicate_violated": False,
            },
        }
        events = sm.process_monitor_events(1, results2)
        assert sm.state == ContractState.ACTIVE


class TestFlapping:
    """Flap detection: ACTIVE↔DEGRADED within recovery guard → RTA_STOP."""

    def _degrade_and_clear(self, sm, step, guard_ticks=10):
        """Degrade then clear at consecutive steps, set recovery guard."""
        sm.process_monitor_events(step, _make_degrade_trigger())
        assert sm.state == ContractState.DEGRADED
        sm.process_monitor_events(step + 1, _make_clear())
        assert sm.state == ContractState.ACTIVE
        sm.set_recovery_guard(step + 1, guard_ticks)

    def test_no_flap_outside_guard(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        self._degrade_and_clear(sm, 0, guard_ticks=10)
        # Re-degrade AFTER guard_end (step 11 >= step 1 + 10 = 11)
        sm.process_monitor_events(11, _make_degrade_trigger())
        assert sm.state == ContractState.DEGRADED
        assert sm.flap_count == 0

    def test_flap_inside_guard(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        self._degrade_and_clear(sm, 0, guard_ticks=10)
        # Re-degrade WITHIN guard (step 5 < 1 + 10 = 11)
        sm.process_monitor_events(5, _make_degrade_trigger())
        assert sm.state == ContractState.DEGRADED
        assert sm.flap_count == 1

    def test_four_flaps_forces_rta_stop(self):
        """4 flap pairs → RTA_STOP (spec Section 4.3: max N=4).

        First degrade has no recovery guard → not a flap.
        Need 5 total D→A→D cycles to get 4 flaps.
        """
        sm = ContractStateMachine(LOOP_RATE_HZ, max_flap_pairs=4)
        step = 0

        # Cycle 0: initial degrade (no prior guard → not a flap)
        sm.process_monitor_events(step, _make_degrade_trigger())
        assert sm.state == ContractState.DEGRADED
        step += 1
        sm.process_monitor_events(step, _make_clear())
        assert sm.state == ContractState.ACTIVE
        sm.set_recovery_guard(step, 100)
        step += 1
        assert sm.flap_count == 0

        # Cycles 1-4: each re-degrade is within guard → flap
        for i in range(1, 5):
            sm.process_monitor_events(step, _make_degrade_trigger())
            step += 1
            if i < 4:
                assert sm.state == ContractState.DEGRADED, f"cycle {i}"
                assert sm.flap_count == i
                sm.process_monitor_events(step, _make_clear())
                assert sm.state == ContractState.ACTIVE
                sm.set_recovery_guard(step, 100)
                step += 1
            else:
                # 4th flap: should force RTA_STOP
                assert sm.state == ContractState.RTA_STOP
                assert sm.flap_count == 4

    def test_flap_limiter_emits_two_events(self):
        """Flap limiter emits A→D then D→RTA_STOP."""
        sm = ContractStateMachine(LOOP_RATE_HZ, max_flap_pairs=1)

        # First cycle: no flap
        sm.process_monitor_events(0, _make_degrade_trigger())
        sm.process_monitor_events(1, _make_clear())
        sm.set_recovery_guard(1, 100)

        # Second degrade = 1st flap → limiter fires
        events = sm.process_monitor_events(2, _make_degrade_trigger())
        assert sm.state == ContractState.RTA_STOP
        assert len(events) == 2
        assert events[0]["from_state"] == "ACTIVE"
        assert events[0]["to_state"] == "DEGRADED"
        assert events[1]["from_state"] == "DEGRADED"
        assert events[1]["to_state"] == "RTA_STOP"
        assert events[1]["trigger_monitor"] == "flap_limiter"

    def test_flap_count_resets_on_episode_reset(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        sm.process_monitor_events(1, _make_clear())
        sm.set_recovery_guard(1, 100)
        sm.process_monitor_events(2, _make_degrade_trigger())
        assert sm.flap_count == 1
        sm.reset_episode()
        assert sm.flap_count == 0
        assert sm.state == ContractState.ACTIVE


class TestPriorityOrder:
    """When multiple monitors trigger in one step, highest severity wins."""

    def test_rta_stop_beats_revoke(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        results = {
            **_make_revoke_trigger("revoke_mon"),
            **_make_rta_stop_trigger("rta_mon"),
        }
        events = sm.process_monitor_events(0, results)
        assert sm.state == ContractState.RTA_STOP
        assert events[0]["trigger_monitor"] == "rta_mon"

    def test_revoke_beats_degrade(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        results = {
            **_make_degrade_trigger("deg_mon"),
            **_make_revoke_trigger("rev_mon"),
        }
        events = sm.process_monitor_events(0, results)
        assert sm.state == ContractState.REVOKED
        assert events[0]["trigger_monitor"] == "rev_mon"


class TestNoopSteps:
    """Steps with no triggers or clears emit no events."""

    def test_quiet_step_active(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        events = sm.process_monitor_events(0, _make_quiet())
        assert sm.state == ContractState.ACTIVE
        assert len(events) == 0

    def test_quiet_step_degraded(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        # Quiet: monitor still triggered but not should_trigger again
        results = {
            "test_monitor": {
                "triggered": True, "should_trigger": False, "should_clear": False,
                "severity": SEVERITY_DEGRADE, "predicate_violated": True,
            }
        }
        events = sm.process_monitor_events(1, results)
        assert sm.state == ContractState.DEGRADED
        assert len(events) == 0


class TestEpisodeReset:
    """Per-episode reset behavior."""

    def test_reset_restores_active(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_revoke_trigger())
        assert sm.state == ContractState.REVOKED
        sm.reset_episode()
        assert sm.state == ContractState.ACTIVE

    def test_reset_clears_recovery_guard(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.set_recovery_guard(10, 25)
        sm.reset_episode()
        # After reset, re-degradation should not be a flap
        sm.process_monitor_events(0, _make_degrade_trigger())
        assert sm.flap_count == 0


class TestSnapshot:
    """Snapshot round-trip."""

    def test_round_trip(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        sm.process_monitor_events(1, _make_clear())
        sm.set_recovery_guard(1, 25)
        sm.process_monitor_events(5, _make_degrade_trigger())  # flap
        assert sm.flap_count == 1

        snap = sm.snapshot()
        sm2 = ContractStateMachine.from_snapshot(snap, LOOP_RATE_HZ)
        assert sm2.state == sm.state
        assert sm2.flap_count == sm.flap_count
        assert sm2.snapshot() == snap


class TestEventShape:
    """Events match Appendix C.2 field names."""

    def test_event_has_all_fields(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        events = sm.process_monitor_events(42, _make_degrade_trigger("cam_health"))
        assert len(events) == 1
        e = events[0]
        # Appendix C.2 required fields
        assert e["event_type"] == "contract_state_change"
        assert e["step"] == 42
        assert e["from_state"] == "ACTIVE"
        assert e["to_state"] == "DEGRADED"
        assert e["trigger_monitor"] == "cam_health"
        assert e["trigger_severity"] == SEVERITY_DEGRADE
        assert "monitors_status" in e
        # Gate 2 placeholders
        assert e["prev_hash"] is None
        assert e["timestamp_ns"] is None
        assert e["episode_id"] is None

    def test_monitors_status_snapshot(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        events = sm.process_monitor_events(0, _make_degrade_trigger("test_mon"))
        ms = events[0]["monitors_status"]
        assert "test_mon" in ms
        assert ms["test_mon"]["triggered"] is True
        assert ms["test_mon"]["violated"] is True


class TestL0RNearmissFromEveryState:
    """L0-R near-miss → RTA_STOP from any state (Section 4.2)."""

    def test_from_active(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.notify_l0r_nearmiss(0, _empty_status())
        assert sm.state == ContractState.RTA_STOP

    def test_from_degraded(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_degrade_trigger())
        sm.notify_l0r_nearmiss(1, _empty_status())
        assert sm.state == ContractState.RTA_STOP

    def test_from_revoked(self):
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_revoke_trigger())
        sm.notify_l0r_nearmiss(1, _empty_status())
        assert sm.state == ContractState.RTA_STOP

    def test_from_rta_stop_is_noop(self):
        """Already in RTA_STOP → L0-R near-miss is a no-op (already safest)."""
        sm = ContractStateMachine(LOOP_RATE_HZ)
        sm.process_monitor_events(0, _make_rta_stop_trigger())
        assert sm.state == ContractState.RTA_STOP
        events = sm.notify_l0r_nearmiss(1, _empty_status())
        assert sm.state == ContractState.RTA_STOP
        assert len(events) == 0  # No-op, no event
