"""Package CLI for producing, checking, and structurally replaying bundles."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from safety_kernel.telemetry.checker import check_bundle, finalize_bundle, structural_replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m safety_kernel")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="create a new immutable Gate 2 telemetry bundle")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True, help="JSON run manifest")
    run.add_argument("--contract", type=Path, required=True, help="JSON contract evidence")
    run.add_argument("--environment", type=Path, required=True, help="JSON environment evidence")
    run.add_argument("--adapter", help="module:callable; callable receives TelemetryRun")
    check = commands.add_parser("check", help="independently validate bundle evidence")
    check.add_argument("bundle", type=Path)
    finalize = commands.add_parser("finalize", help="independently check then write immutable checker and receipt artifacts")
    finalize.add_argument("bundle", type=Path)
    replay = commands.add_parser("replay", help="replay ordering, states, and counts from bundle alone")
    replay.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    output = check_bundle(args.bundle) if args.command == "check" else finalize_bundle(args.bundle) if args.command == "finalize" else structural_replay(args.bundle)
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    accepted = output.get("ok") if args.command == "replay" else output.get("evidence_ok")
    return 0 if accepted else 1


def _run(args: argparse.Namespace) -> int:
    from safety_kernel.telemetry.adapter import TelemetryRun
    from safety_kernel.telemetry.bundle import RunBundle

    manifest = _read_object(args.manifest, "manifest")
    contract = _read_object(args.contract, "contract")
    environment = _read_object(args.environment, "environment")
    total = manifest.get("total_episodes")
    if not isinstance(total, int) or total < 0:
        raise SystemExit("manifest.total_episodes must be a non-negative integer")
    telemetry = TelemetryRun(RunBundle(args.out, manifest=manifest, contract=contract, environment=environment), total_episodes=total)
    if not args.adapter:
        raise SystemExit("--adapter module:callable is required; runtime ownership stays outside telemetry")
    module_name, separator, attribute = args.adapter.partition(":")
    if not separator or not module_name or not attribute:
        raise SystemExit("--adapter must be module:callable")
    callback = getattr(importlib.import_module(module_name), attribute)
    callback(telemetry)
    output = telemetry.complete(timestamp_ns=int(manifest.get("completion_timestamp_ns", 0)))
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return parsed
