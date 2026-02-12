#!/usr/bin/env python3
"""L0-R projection performance benchmark (C4b).

Standalone script — not part of pytest suite.
Runs 1000 random projections, reports latency statistics.
Target: < 1ms per call.
"""

import time
import numpy as np
import mujoco

from safety_kernel.env.cell_arm import CellArmEnv
from safety_kernel.l0r.projection import l0r_project


def main():
    N = 1000
    env = CellArmEnv(seed=0, enable_l0r=False)
    params = env.l0r_params
    rng = np.random.default_rng(42)

    # Pregenerate random states and commands
    q_samples = rng.uniform(-1.5, 1.5, size=(N, 5))
    qdot_cmds = rng.uniform(-2.0, 2.0, size=(N, 5))

    latencies = []

    for i in range(N):
        env.data.qpos[:] = q_samples[i]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        q = env.data.qpos[:5].copy()
        M = env._get_mass_matrix()

        t0 = time.perf_counter_ns()
        qdot_exec, info = l0r_project(
            qdot_cmds[i].copy(), q, M, env.model, env.data, params
        )
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)  # ms

    latencies = np.array(latencies)

    print("=" * 60)
    print("L0-R Projection Benchmark")
    print("=" * 60)
    print(f"  Calls:    {N}")
    print(f"  Median:   {np.median(latencies):.3f} ms")
    print(f"  P95:      {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99:      {np.percentile(latencies, 99):.3f} ms")
    print(f"  Max:      {np.max(latencies):.3f} ms")
    print(f"  Min:      {np.min(latencies):.3f} ms")
    total_s = np.sum(latencies) / 1000
    print(f"  Total:    {total_s:.3f} s")
    print(f"  Rate:     {N / total_s:.0f} calls/s")
    print("-" * 60)

    target = 1.0  # ms
    if np.percentile(latencies, 99) < target:
        print(f"  PASS — P99 ({np.percentile(latencies, 99):.3f} ms) < {target} ms target")
    else:
        print(f"  INFO — P99 ({np.percentile(latencies, 99):.3f} ms) >= {target} ms target")
        print(f"         (diagnostic only, not a gate blocker)")
    print("=" * 60)


if __name__ == "__main__":
    main()
