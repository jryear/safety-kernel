"""L0-C enforcement tests.

Tests TCP speed cap: uniform scaling, monotonicity, state gating.
Pure-function tests — no MuJoCo dependency.
"""

import numpy as np
import pytest

from safety_kernel.contract.l0c_enforce import l0c_enforce
from safety_kernel.contract.state_machine import ContractState


# Identity Jacobian: TCP velocity = joint velocity (for simplicity)
J_IDENTITY = np.eye(3, 5)  # 3×5: first 3 joints map 1:1 to TCP xyz

# Simple Jacobian where joint 0 maps to x, joint 1 to y
J_SIMPLE = np.zeros((3, 5))
J_SIMPLE[0, 0] = 1.0  # joint 0 → x
J_SIMPLE[1, 1] = 1.0  # joint 1 → y
J_SIMPLE[2, 2] = 1.0  # joint 2 → z

V_MAX = 0.6  # m/s (matches Track A contract)


class TestPassthroughNonDegraded:
    """L0-C only enforces in DEGRADED state."""

    def test_active_passthrough(self):
        qdot = np.array([1.0, 1.0, 1.0, 0.0, 0.0])  # speed = sqrt(3) ≈ 1.73
        qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.ACTIVE, V_MAX)
        np.testing.assert_array_equal(qdot_out, qdot)
        assert not info["constraints_active"]
        assert info["scaling_factor"] == 1.0

    def test_revoked_passthrough(self):
        qdot = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.REVOKED, V_MAX)
        np.testing.assert_array_equal(qdot_out, qdot)

    def test_rta_stop_passthrough(self):
        qdot = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.RTA_STOP, V_MAX)
        np.testing.assert_array_equal(qdot_out, qdot)


class TestSpeedCap:
    """TCP speed cap in DEGRADED state."""

    def test_below_cap_passthrough(self):
        # speed = 0.3 < 0.6
        qdot = np.array([0.3, 0.0, 0.0, 0.0, 0.0])
        qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.DEGRADED, V_MAX)
        np.testing.assert_array_equal(qdot_out, qdot)
        assert info["scaling_factor"] == 1.0
        assert not info["constraints_active"]

    def test_above_cap_scales(self):
        # speed = 1.0 > 0.6
        qdot = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.DEGRADED, V_MAX)
        # Should be scaled to 0.6
        assert info["scaling_factor"] == pytest.approx(0.6)
        assert "tcp_speed_cap" in info["constraints_active"]
        v_tcp = J_SIMPLE @ qdot_out
        assert np.linalg.norm(v_tcp) == pytest.approx(V_MAX)

    def test_uniform_scaling(self):
        """All joints scaled by same factor (monotone, uniform)."""
        qdot = np.array([0.8, 0.6, 0.0, 0.5, 0.3])
        qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.DEGRADED, V_MAX)
        scale = info["scaling_factor"]
        np.testing.assert_allclose(qdot_out, qdot * scale)

    def test_exact_cap_no_scaling(self):
        """Speed exactly at cap → no scaling."""
        qdot = np.array([0.6, 0.0, 0.0, 0.0, 0.0])
        qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.DEGRADED, V_MAX)
        np.testing.assert_array_equal(qdot_out, qdot)
        assert info["scaling_factor"] == 1.0

    def test_zero_velocity(self):
        """Zero velocity → no scaling, no crash."""
        qdot = np.zeros(5)
        qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.DEGRADED, V_MAX)
        np.testing.assert_array_equal(qdot_out, qdot)
        assert info["scaling_factor"] == 1.0


class TestMonotonicity:
    """L0-C only shrinks commands, never amplifies."""

    def test_output_leq_input(self):
        for _ in range(100):
            qdot = np.random.randn(5) * 2.0
            qdot_out, info = l0c_enforce(qdot, J_SIMPLE, ContractState.DEGRADED, V_MAX)
            scale = info["scaling_factor"]
            assert scale <= 1.0 + 1e-10


class TestNoCopyMutation:
    """l0c_enforce must not mutate the input array."""

    def test_input_unchanged(self):
        qdot = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        original = qdot.copy()
        l0c_enforce(qdot, J_SIMPLE, ContractState.DEGRADED, V_MAX)
        np.testing.assert_array_equal(qdot, original)


class TestInfoFields:
    """Info dict has expected fields."""

    def test_all_fields_present(self):
        qdot = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        _, info = l0c_enforce(qdot, J_SIMPLE, ContractState.DEGRADED, V_MAX)
        assert "l0c_active" in info
        assert "tcp_speed_before" in info
        assert "tcp_speed_after" in info
        assert "scaling_factor" in info
        assert "constraints_active" in info
        assert info["l0c_active"] is True

    def test_active_state_not_active(self):
        qdot = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        _, info = l0c_enforce(qdot, J_SIMPLE, ContractState.ACTIVE, V_MAX)
        assert info["l0c_active"] is False
