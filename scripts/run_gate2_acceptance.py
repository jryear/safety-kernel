#!/usr/bin/env python3
"""Produce the deterministic 100 x 200 Gate 2 command-evidence run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from safety_kernel.contract.shift_injector import ShiftSpec
from safety_kernel.env.cell_arm import CellArmEnv
from safety_kernel.telemetry import RunBundle, TelemetryRun


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--contract", type=Path, default=Path("contracts/cell_arm_track_a.yaml")
    )
    args = parser.parse_args()
    if args.episodes < 2 or args.steps < 30:
        parser.error("acceptance requires at least two episodes and 30 steps")

    contract = {
        "contract_envelope_version": "ffsk.contract-envelope.v1",
        "original_yaml": args.contract.read_text(encoding="utf-8"),
    }
    manifest = {
        "bundle_kind": "acceptance",
        "total_episodes": args.episodes,
        "steps_per_episode": args.steps,
        "seed_schedule": list(range(args.episodes)),
        "flap_limit": 4,
        "loop_rate_hz": 50,
        "monitor_slas": {
            "safety_io_flip:estop_ok": {
                "max_steps": 0,
                "to_state": "RTA_STOP",
                "monitor": "estop_chain_ok",
            },
            "sensor_stale": {
                "max_steps": 26,
                "to_state": "DEGRADED",
                "monitor": "front_cam_health",
            },
        },
        "command_authority": "software_command_suppression_only",
        "completion_timestamp_ns": args.episodes * args.steps * 20_000_000,
    }
    probe = CellArmEnv(contract_path=args.contract, max_steps=args.steps, seed=0)
    params = probe.l0r_params
    environment = {
        "environment": "CellArmEnv",
        "model": "models/cell_arm.xml",
        "model_xml": probe._model_path.read_text(encoding="utf-8"),
        "model_sha256": hashlib.sha256(probe._model_path.read_bytes()).hexdigest(),
        "control_timestep_seconds": float(probe.model.opt.timestep),
        "l0r_parameters": {
            "n_joints": 5, "dt": float(params.dt),
            "qdot_max_eroded": params.qdot_max_eroded.tolist(),
            "q_min_eroded": params.q_min_eroded.tolist(),
            "q_max_eroded": params.q_max_eroded.tolist(),
            "e_max_safe": float(params.e_max_safe), "d_safe": float(params.d_safe),
            "collision_pair_names": list(params.collision_pair_names),
        },
    }
    telemetry = TelemetryRun(
        RunBundle(
            args.out,
            manifest=manifest,
            contract=contract,
            environment=environment,
        ),
        total_episodes=args.episodes,
    )
    for episode in range(args.episodes):
        shift = ShiftSpec(
            start_step=0,
            end_step=None,
            shift_type="safety_io_flip" if episode == 0 else "sensor_stale",
            params={"signal": "estop_ok"} if episode == 0 else {"frame_age_ms": 500.0},
        )
        env = CellArmEnv(
            contract_path=args.contract,
            shift_schedule=[shift],
            max_steps=args.steps,
            seed=episode,
        )
        env.reset(seed=episode)
        telemetry.start_episode(
            f"episode-{episode:03d}", episode, planned_steps=args.steps
        )
        for step in range(args.steps):
            phase = 0.013 * episode + 0.071 * step
            proposal = (
                0.05 * np.sin(phase + np.arange(5, dtype=np.float64))
                if episode == 0
                else np.full(5, np.pi, dtype=np.float64)
            )
            _, _, _, _, info = env.step(proposal)
            shifts = ()
            if step == 0:
                shifts = ({
                    "shift_type": "safety_io_flip" if episode == 0 else "sensor_stale",
                    "params": {"signal": "estop_ok"} if episode == 0 else {"frame_age_ms": 500.0},
                    "affected_channels": ["safety_io"] if episode == 0 else ["sensor_health"],
                },)
            telemetry.record_step(info, shifts=shifts)
        telemetry.end_episode(
            synthetic_revocation=False,
            synthetic_revocation_params=None,
        )
    telemetry.complete(
        timestamp_ns=int(args.episodes * args.steps * params.dt * 1_000_000_000)
    )
    # Finalization is deliberately a fresh interpreter after producer sealing.
    completed = subprocess.run(
        [sys.executable, "-m", "safety_kernel", "finalize", str(args.out)],
        check=False, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    receipt_path = args.out / "completion_receipt.json"
    report = {
        "bundle": str(args.out.resolve()),
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "events_sha256": hashlib.sha256(
            (args.out / "events.ndjson").read_bytes()
        ).hexdigest(),
        "completion_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        if receipt_path.is_file()
        else None,
        "checker": result,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    # Gate 2 accepts structural evidence and reports H3 separately. A strict
    # H3 failure is a Gate 3 result, not permission to discard the Gate 2 run.
    return 0 if result["evidence_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
