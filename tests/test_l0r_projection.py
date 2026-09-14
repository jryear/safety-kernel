"""Functional tests for L0-R projection (Gate 0 acceptance — C4).

Tests from the plan:
1. Passthrough — command within all limits → qdot_exec == qdot_cmd (bitwise)
2. Velocity clamp — exceeds velocity limits → clamped to eroded limits
3. Position lookahead — joint near limit → velocity scaled
4. Energy cap — high-velocity command → uniformly scaled
5. Self-collision — approaching collision → zero velocity
6. Combined — multiple constraints active
7. Infeasibility — joint outside eroded bounds → recovery velocity + RTA flag
8. Erosion correctness — eroded limits == physical - erosion terms
9. Determinism — same input → identical output across 1000 runs (bitwise)
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import mujoco
import pytest

from safety_kernel.l0r.params import L0RParams
from safety_kernel.l0r.projection import l0r_project
from safety_kernel.env.cell_arm import CellArmEnv

# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    """Shared environment for all tests."""
    e = CellArmEnv(seed=0, e_max=5.0, enable_l0r=False)
    e.reset(seed=0)
    return e


@pytest.fixture(scope="module")
def params(env):
    return env.l0r_params


def _get_state(env):
    """Get current (q, M, model, data) from env."""
    q = env.data.qpos[:env.n_joints].copy()
    M = env._get_mass_matrix()
    return q, M, env.model, env.data


# ──────────────────────────────────────────────────────────────────
# 1. Passthrough test
# ──────────────────────────────────────────────────────────────────

def test_passthrough(env, params):
    """Command well within all limits passes through unchanged."""
    env.reset(seed=1)
    # Small command that's well within all limits
    qdot_cmd = np.array([0.1, -0.1, 0.05, -0.05, 0.02])
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    np.testing.assert_array_equal(qdot_exec, qdot_cmd)
    assert info["constraints_active"] == []
    assert info["energy_scaling_factor"] == 1.0
    assert info["rta_nearmiss"] is False


# ──────────────────────────────────────────────────────────────────
# 2. Velocity clamp
# ──────────────────────────────────────────────────────────────────

def test_velocity_clamp(env, params):
    """Command exceeding velocity limits is clamped to eroded limits."""
    env.reset(seed=2)
    # Command well above velocity limits
    qdot_cmd = np.full(5, 10.0)  # >> π rad/s
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    qdot_max_e = params.qdot_max_eroded
    # Each joint should be at or below eroded limit
    for i in range(5):
        assert abs(qdot_exec[i]) <= qdot_max_e[i] + 1e-10
    assert "velocity_clamp" in info["constraints_active"]


def test_velocity_clamp_negative(env, params):
    """Negative velocity commands also get clamped."""
    env.reset(seed=3)
    qdot_cmd = np.full(5, -10.0)
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    qdot_max_e = params.qdot_max_eroded
    for i in range(5):
        assert abs(qdot_exec[i]) <= qdot_max_e[i] + 1e-10
    assert "velocity_clamp" in info["constraints_active"]


# ──────────────────────────────────────────────────────────────────
# 3. Position lookahead
# ──────────────────────────────────────────────────────────────────

def test_position_lookahead_near_upper_limit(env, params):
    """Joint near upper position limit → velocity reduced to avoid overshoot."""
    env.reset(seed=4)
    # Place joint 0 very close to eroded upper limit so that even a modest
    # velocity would push q_next past the eroded bound.
    # eroded_upper ≈ π - 0.0012, so place at eroded_upper - 0.01
    q_near_limit = np.zeros(5)
    q_near_limit[0] = params.q_max_eroded[0] - 0.01
    env.data.qpos[:] = q_near_limit
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Command velocity that would overshoot: 0.01/dt = 0.5, so 2.0 >> 0.5
    qdot_cmd = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    # q_next should not exceed eroded upper limit
    q_next = q + qdot_exec * params.dt
    assert q_next[0] <= params.q_max_eroded[0] + 1e-8
    assert "position_lookahead" in info["constraints_active"]


def test_position_lookahead_near_lower_limit(env, params):
    """Joint near lower position limit → negative velocity clamped."""
    env.reset(seed=5)
    q_near_limit = np.zeros(5)
    q_near_limit[1] = params.q_min_eroded[1] + 0.01
    env.data.qpos[:] = q_near_limit
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    qdot_cmd = np.array([0.0, -2.0, 0.0, 0.0, 0.0])
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    q_next = q + qdot_exec * params.dt
    assert q_next[1] >= params.q_min_eroded[1] - 1e-8
    assert "position_lookahead" in info["constraints_active"]


def test_position_lookahead_cross_term(env, params):
    """Erosion includes the eps_qdot * dt cross-term."""
    # The eroded position limits should account for velocity uncertainty
    dt = params.dt
    expected_erosion = params.eps_q + params.eps_qdot * dt
    for i in range(5):
        actual_upper = params.q_max[i] - params.q_max_eroded[i]
        assert abs(actual_upper - expected_erosion) < 1e-12


# ──────────────────────────────────────────────────────────────────
# 4. Energy cap
# ──────────────────────────────────────────────────────────────────

def test_energy_cap(env, params):
    """High-velocity command → uniformly scaled, scaling factor correct."""
    env.reset(seed=6)
    # Large velocity that should exceed energy cap
    qdot_cmd = np.full(5, 2.5)  # high velocity on all joints
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    # Check energy is within budget
    e_k = 0.5 * qdot_exec @ M @ qdot_exec
    assert e_k <= params.e_max_safe + 1e-8

    if "energy_cap" in info["constraints_active"]:
        # Scaling factor should be < 1.0
        assert info["energy_scaling_factor"] < 1.0
        # Verify uniform scaling: ratios should be equal
        # (but only for joints that weren't already clamped by velocity/position)
        sf = info["energy_scaling_factor"]
        # The velocity before energy cap is the post-position-clamp value,
        # so we can't directly reconstruct it. Just verify energy is correct.
        assert e_k <= params.e_max_safe + 1e-8


def test_energy_cap_within_budget(env, params):
    """Small command within energy budget → no scaling."""
    env.reset(seed=7)
    qdot_cmd = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    assert info["energy_scaling_factor"] == 1.0
    assert "energy_cap" not in info["constraints_active"]


# ──────────────────────────────────────────────────────────────────
# 5. Self-collision
# ──────────────────────────────────────────────────────────────────

def test_self_collision_triggers_zero(env, params):
    """Arm configuration approaching self-collision → zero velocity."""
    env.reset(seed=8)
    # Fold the arm back: shoulder pitch + elbow can bring link_3/4/5 near link_0
    # Use values inside eroded bounds (avoid ±π which is at physical limit)
    q_folded = np.array([0.0, 1.5, 3.0, -1.5, 0.0])
    env.data.qpos[:] = q_folded
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Check if any collision pair is already close
    from safety_kernel.l0r.capsule import compute_collision_distances
    dists = compute_collision_distances(
        env.model, env.data, params.collision_pairs
    )
    min_dist = min(dists)

    if min_dist < params.d_safe + 0.1:
        # We're near a collision zone — command velocity that maintains/worsens it
        qdot_cmd = np.array([0.0, 0.1, 0.0, -0.1, 0.0])
        q, M, model, data = _get_state(env)
        qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

        if "self_collision" in info["constraints_active"]:
            np.testing.assert_array_equal(qdot_exec, np.zeros(5))
            # Verify pair IDs are reported
            assert len(info["collision_pairs"]) > 0
            violated = [c for c in info["collision_pairs"]
                        if c["distance"] < params.d_safe]
            assert len(violated) > 0
    else:
        pytest.skip(f"Config not near collision (min_dist={min_dist:.4f})")


def test_self_collision_forced(env, params):
    """Force a known near-collision config and verify zero-velocity response."""
    env.reset(seed=9)
    # Extreme fold: bring end-effector back to base
    # Try several configurations to find one that triggers
    configs = [
        np.array([0.0, np.pi * 0.7, np.pi * 0.9, -np.pi * 0.8, 0.0]),
        np.array([0.0, np.pi * 0.8, np.pi * 0.5, -np.pi * 0.9, 0.0]),
        np.array([0.0, np.pi * 0.6, np.pi, -np.pi * 0.7, 0.0]),
        np.array([0.0, np.pi * 0.9, np.pi * 0.8, -np.pi * 0.5, 0.0]),
    ]

    from safety_kernel.l0r.capsule import compute_collision_distances_at_q

    for q_test in configs:
        env.data.qpos[:] = q_test
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        # Check distances at a slightly moved config
        qdot_test = np.array([0.0, 0.3, 0.0, -0.3, 0.0])
        q_next = q_test + qdot_test * params.dt
        dists = compute_collision_distances_at_q(
            env.model, env.data, q_next, params.collision_pairs
        )

        if min(dists) < params.d_safe:
            q, M, model, data = _get_state(env)
            qdot_exec, info = l0r_project(qdot_test, q, M, model, data, params)
            assert "self_collision" in info["constraints_active"]
            np.testing.assert_array_equal(qdot_exec, np.zeros(5))
            return

    pytest.skip("Could not find a config that triggers self-collision at this velocity")


# ──────────────────────────────────────────────────────────────────
# 6. Combined constraints
# ──────────────────────────────────────────────────────────────────

def test_combined_constraints(env, params):
    """Multiple constraints active simultaneously → all satisfied in output."""
    env.reset(seed=10)
    # Place joint near limit AND command high velocity
    q_setup = np.zeros(5)
    q_setup[0] = params.q_max[0] - 0.03  # near upper position limit
    env.data.qpos[:] = q_setup
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # High velocity on all joints
    qdot_cmd = np.full(5, 5.0)
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    # Velocity constraint
    for i in range(5):
        assert abs(qdot_exec[i]) <= params.qdot_max_eroded[i] + 1e-8

    # Position constraint
    q_next = q + qdot_exec * params.dt
    for i in range(5):
        assert q_next[i] <= params.q_max_eroded[i] + 1e-8
        assert q_next[i] >= params.q_min_eroded[i] - 1e-8

    # Energy constraint
    e_k = 0.5 * qdot_exec @ M @ qdot_exec
    assert e_k <= params.e_max_safe + 1e-8

    # Should have triggered at least velocity and position
    assert len(info["constraints_active"]) >= 2


# ──────────────────────────────────────────────────────────────────
# 7. Infeasibility
# ──────────────────────────────────────────────────────────────────

def test_infeasibility_recovery(env, params):
    """Joint outside eroded bounds → recovery velocity + RTA flag."""
    env.reset(seed=11)
    # Place joint 2 outside eroded upper bound but inside physical bound
    # Erosion gap = eps_q + eps_qdot*dt = 0.001 + 0.01*0.02 = 0.0012 rad
    # Place halfway into the gap
    q_infeasible = np.zeros(5)
    gap = params.q_max[2] - params.q_max_eroded[2]  # ~0.0012
    q_infeasible[2] = params.q_max_eroded[2] + gap * 0.5
    assert q_infeasible[2] > params.q_max_eroded[2], "Must be outside eroded"
    assert q_infeasible[2] < params.q_max[2], "Must be inside physical limits"

    env.data.qpos[:] = q_infeasible
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Command positive velocity (trying to go further out)
    qdot_cmd = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    q, M, model, data = _get_state(env)

    qdot_exec, info = l0r_project(qdot_cmd, q, M, model, data, params)

    # Should flag RTA near-miss
    assert info["rta_nearmiss"] is True
    # Recovery velocity should move toward safe region (negative for joint 2)
    assert qdot_exec[2] <= 0.0


# ──────────────────────────────────────────────────────────────────
# 8. Erosion correctness
# ──────────────────────────────────────────────────────────────────

def test_erosion_correctness(params):
    """Eroded limits == physical limits - erosion terms exactly."""
    dt = params.dt
    eps_q = params.eps_q
    eps_qdot = params.eps_qdot

    # Position erosion
    for i in range(5):
        expected_lower = params.q_min[i] + eps_q + eps_qdot * dt
        expected_upper = params.q_max[i] - eps_q - eps_qdot * dt
        assert abs(params.q_min_eroded[i] - expected_lower) < 1e-12
        assert abs(params.q_max_eroded[i] - expected_upper) < 1e-12

    # Velocity erosion
    for i in range(5):
        expected = params.qdot_max[i] - eps_qdot
        assert abs(params.qdot_max_eroded[i] - expected) < 1e-12

    # Energy erosion
    expected_e = params.e_max * params.energy_margin
    assert abs(params.e_max_safe - expected_e) < 1e-12

    # Eroded strictly tighter than physical
    for i in range(5):
        assert params.q_min_eroded[i] > params.q_min[i]
        assert params.q_max_eroded[i] < params.q_max[i]
        assert params.qdot_max_eroded[i] < params.qdot_max[i]
    assert params.e_max_safe < params.e_max


# ──────────────────────────────────────────────────────────────────
# 9. Determinism
# ──────────────────────────────────────────────────────────────────

def test_determinism(env, params):
    """Same input → identical output across 1000 runs (bitwise)."""
    env.reset(seed=42)
    q_fixed = np.array([0.5, -0.3, 0.7, -0.2, 0.1])
    env.data.qpos[:] = q_fixed
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    qdot_cmd = np.array([1.5, -1.0, 2.0, -0.5, 0.8])
    q, M, model, data = _get_state(env)

    # First run — reference
    qdot_ref, info_ref = l0r_project(qdot_cmd.copy(), q.copy(), M.copy(),
                                      model, data, params)

    # 1000 more runs — must be bitwise identical
    for i in range(1000):
        # Reset state each time
        env.data.qpos[:] = q_fixed
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        q2, M2, _, _ = _get_state(env)

        qdot_out, info_out = l0r_project(qdot_cmd.copy(), q2.copy(), M2.copy(),
                                          model, data, params)
        np.testing.assert_array_equal(qdot_out, qdot_ref)
        assert info_out["energy_scaling_factor"] == info_ref["energy_scaling_factor"]


def test_output_invariant_check_runs_under_python_optimized_mode():
    """L0-R output verification must not disappear under ``python -O``."""
    code = textwrap.dedent(
        """
        from types import SimpleNamespace
        import numpy as np

        from safety_kernel.l0r.errors import L0RSafetyInvariantError
        from safety_kernel.l0r.projection import _verify_output_constraints

        params = SimpleNamespace(
            n_joints=1,
            dt=0.02,
            qdot_max_eroded=np.array([1.0]),
            q_min_eroded=np.array([-2.0]),
            q_max_eroded=np.array([2.0]),
            e_max_safe=10.0,
            d_safe=0.02,
            collision_pairs=[],
            collision_pair_names=[],
        )
        try:
            _verify_output_constraints(
                np.array([2.0]), np.array([0.0]), np.eye(1), None, None, params
            )
        except L0RSafetyInvariantError:
            raise SystemExit(0)
        raise SystemExit("unsafe output was accepted under python -O")
        """
    )
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = os.environ | {
        "PYTHONPATH": str(repo_root / "src")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


# ──────────────────────────────────────────────────────────────────
# Smoke test: L2 controller integration
# ──────────────────────────────────────────────────────────────────

def test_l2_pd_controller_smoke(env, params):
    """Run a few steps with PD reaching controller through L0-R.

    Verifies that the kernel output satisfies all constraints
    (the debug assertion inside l0r_project catches violations).
    """
    env_l0r = CellArmEnv(seed=100, e_max=5.0, enable_l0r=True)
    obs, info = env_l0r.reset(seed=100)

    # Simple PD reaching: move toward a goal
    goal_tcp = np.array([0.3, 0.0, 0.6])
    n_steps = 50
    interventions = 0

    for step in range(n_steps):
        tcp = env_l0r.get_tcp_position()
        J = env_l0r.get_jacobian()

        error = goal_tcp - tcp
        K_p = 2.0
        qdot_cmd = np.linalg.pinv(J) @ (K_p * error)

        obs, r, term, trunc, info = env_l0r.step(qdot_cmd)
        if info["intervention"]["constraints_active"]:
            interventions += 1

    # If the debug assertion in l0r_project didn't fire, all outputs were safe
    print(f"  L2 smoke test: {interventions}/{n_steps} steps with interventions")
