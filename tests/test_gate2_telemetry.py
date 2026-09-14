"""Gate 2 acceptance vectors for the accepted decision-evidence producer ABI."""

from __future__ import annotations

import binascii
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from safety_kernel.telemetry.adapter import TelemetryRun
from safety_kernel.telemetry.bundle import RunBundle
from safety_kernel.telemetry.canonical import event_bytes, event_hash, prepare_event
from safety_kernel.telemetry.checker import (
    _h3_flapping,
    _h3_sla,
    check_bundle,
    finalize_bundle,
    structural_replay,
)
from safety_kernel.telemetry.independent import validate_event as validate_independent_event
from safety_kernel.telemetry.errors import BundleImmutableError
from safety_kernel.telemetry.errors import TelemetryError
from safety_kernel.telemetry.schema import EVENT_TYPES, validate_event as validate_producer_event
from safety_kernel.env.cell_arm import CellArmEnv
from safety_kernel.contract.shift_injector import ShiftSpec


def _contract_envelope() -> dict:
    return {
        "contract_envelope_version": "ffsk.contract-envelope.v1",
        "original_yaml": Path("contracts/cell_arm_track_a.yaml").read_text(),
    }


def _environment() -> dict:
    model_xml = Path("models/cell_arm.xml").read_text()
    return {
        "environment": "CellArmEnv", "model": "models/cell_arm.xml",
        "model_xml": model_xml,
        "model_sha256": hashlib.sha256(model_xml.encode()).hexdigest(), "control_timestep_seconds": 0.02,
        "l0r_parameters": {
            "n_joints": 5, "dt": 0.02, "qdot_max_eroded": [1.0] * 5,
            "q_min_eroded": [-2.0] * 5, "q_max_eroded": [2.0] * 5,
            "e_max_safe": 5.0, "d_safe": 0.02,
            "collision_pair_names": ["a-b"],
        },
    }


def _environment_for(env: CellArmEnv) -> dict:
    params = env.l0r_params
    return {
        "environment": "CellArmEnv", "model": "models/cell_arm.xml",
        "model_xml": env._model_path.read_text(),
        "model_sha256": hashlib.sha256(env._model_path.read_bytes()).hexdigest(),
        "control_timestep_seconds": float(env.model.opt.timestep),
        "l0r_parameters": {
            "n_joints": 5, "dt": float(params.dt),
            "qdot_max_eroded": params.qdot_max_eroded.tolist(),
            "q_min_eroded": params.q_min_eroded.tolist(),
            "q_max_eroded": params.q_max_eroded.tolist(),
            "e_max_safe": float(params.e_max_safe), "d_safe": float(params.d_safe),
            "collision_pair_names": list(params.collision_pair_names),
        },
    }


def _manifest(total: int = 1, steps: int = 1, *, bundle_kind: str = "diagnostic", seeds: list[int] | None = None) -> dict:
    return {"bundle_kind": bundle_kind, "total_episodes": total, "steps_per_episode": steps,
            "seed_schedule": list(range(total)) if seeds is None else seeds, "flap_limit": 4, "loop_rate_hz": 50,
            "command_authority": "software_command_suppression_only", "monitor_slas": {},
            "completion_timestamp_ns": total * steps * 20_000_000}


def _producer(path: Path, *, steps: int = 1, rta: bool = False, l0r_evidence: dict | None = None) -> Path:
    run = TelemetryRun(RunBundle(path, manifest=_manifest(1, steps), contract=_contract_envelope(), environment=_environment()), total_episodes=1)
    run.start_episode("episode-0", seed=0, planned_steps=steps)
    for step in range(steps): run.record_step(_info(step, rta=rta), shifts=[{"shift_type": "anchor_drift", "params": {"anchor": "fiducial_fixture", "drift_xyz": [0.0, 0.0, 0.0]}, "affected_channels": ["anchor_poses"]}] if step == 0 else [], l0r_evidence=l0r_evidence)
    run.end_episode(); run.complete(timestamp_ns=steps * 20_000_000)
    return path


def _info(step: int, *, rta: bool = False) -> dict:
    proposal, l0c, l0r, final = [1.0] * 5, [0.5] * 5, [0.25] * 5, ([0.0] * 5 if rta else [0.25] * 5)
    monitor_inputs = _monitor_inputs()
    if rta:
        monitor_inputs["safety_io"]["estop_ok"] = False
    return {"step": step, "simulated_pre_step_timestamp_ns": step * 20_000_000, "truth": {"q": [0.0] * 5, "qdot": [0.0] * 5, "tau": [0.0] * 5, "tcp_pos": [0.0] * 3, "tcp_to_keepout_distances": {"fixture": 1.0}, "self_collision_surface_distances": {"a-b": 1.0}},
            "observed": {"q": [0.0] * 5, "qdot": [0.0] * 5, "tau": [0.0] * 5},
            "believed": {"q_interval": [[-0.001, 0.001]] * 5, "qdot_interval": [[-0.01, 0.01]] * 5},
            "intervention": {"qdot_cmd": l0c, "qdot_exec": l0r, "constraints_active": ["velocity_clamp"], "margin_to_L0R": {"velocity": 1.0}, "energy_ratio": 0.1, "energy_scaling_factor": 1.0, "collision_pairs": [{"pair_id": "a-b", "distance": 1.0}], "rta_nearmiss": False},
            "l0c": {"l0c_active": True, "scaling_factor": 0.5, "tcp_speed_before": 2.0, "tcp_speed_after": 1.0, "max_tcp_speed_mps": 1.0, "tcp_jacobian": [[0.4] * 5, [0.0] * 5, [0.0] * 5]},
            "authority_info": {"proposal": proposal, "l0c_command": l0c, "l0r_command": l0r, "final_command": final, "state_pre": "ACTIVE", "state_post": "RTA_STOP" if rta else "ACTIVE", "authorized": not rta, "reason": "estop_chain_ok" if rta else "allowed", "reset_input": None, "monitor_inputs": monitor_inputs, "monitor_results": _monitor_results(rta)},
            "contract_state": "RTA_STOP" if rta else "ACTIVE", "contract_events": [{"from_state": "ACTIVE", "to_state": "RTA_STOP", "trigger_monitor": "estop_chain_ok", "trigger_severity": "rta_stop", "monitors_status": {name: {"triggered": result["triggered"], "violated": result["predicate_violated"]} for name, result in _monitor_results(rta).items()}}] if rta else []}


def _monitor_inputs() -> dict:
    return {
        "anchor_poses": {
            "fiducial_fixture": {"xyz": [0.6, 0.25, 0.6], "rpy": [0.0, 0.0, 0.0]},
            "fiducial_wall": {"xyz": [-0.25, 0.0, 0.8], "rpy": [0.0, 0.0, 0.0]},
        },
        "safety_io": {"gate_closed": True, "estop_ok": True},
        "sensor_health": {"frame_age_ms": 5.0, "timestamp": 0.0},
        "base_pose_estimate": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
    }


def _monitor_results(rta: bool) -> dict:
    specs = {
        "anchors_consistent": "revoke", "gate_interlock_ok": "rta_stop",
        "estop_chain_ok": "rta_stop", "front_cam_health": "degrade",
        "robot_mount_integrity": "revoke",
    }
    return {
        name: {
            "triggered": rta and name == "estop_chain_ok",
            "should_trigger": rta and name == "estop_chain_ok",
            # A monitor that was not previously triggered cannot clear.
            "should_clear": False,
            "severity": severity,
            "predicate_violated": rta and name == "estop_chain_ok",
        }
        for name, severity in specs.items()
    }


def test_nine_event_abi_has_decision_evidence_and_no_producer_verdict(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle")
    events = _events(bundle)
    assert EVENT_TYPES == ("run_start", "episode_start", "shift_injected", "decision_evidence", "safety_intervention", "contract_state_change", "kernel_step_summary", "episode_summary", "run_end")
    assert [event["event_type"] for event in events] == ["run_start", "episode_start", "shift_injected", "decision_evidence", "safety_intervention", "safety_intervention", "kernel_step_summary", "episode_summary", "run_end"]
    assert not (bundle / "checker.json").exists()
    assert events[-1]["actual_event_count"] == len(events)
    assert events[-1]["last_event_hash"] == event_hash(events[-2])


def test_rta_decision_and_causal_order_are_exact(tmp_path: Path):
    events = _events(_producer(tmp_path / "bundle", rta=True))
    assert [event["event_type"] + (":" + event["layer"] if event["event_type"] == "safety_intervention" else "") for event in events[2:-2]] == ["shift_injected", "decision_evidence", "safety_intervention:L0C", "safety_intervention:L0R", "safety_intervention:RTA", "contract_state_change", "kernel_step_summary"]
    decision = next(event for event in events if event["event_type"] == "decision_evidence")
    rta = next(event for event in events if event.get("layer") == "RTA")
    assert decision["authorized"] is False and rta["qdot_exec"] == decision["final_command"]


def test_denial_uses_authoritative_positive_zero_not_display_rounding(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle", rta=True)
    assert finalize_bundle(bundle)["evidence_ok"]
    events = _events(bundle)
    decision = next(event for event in events if event["event_type"] == "decision_evidence")
    # Both render as 0.000000, but only the first is an authority zero.
    decision["float_hex"]["/final_command/0"] = (0.0000004).hex()
    _rehash(events); _write_rehashed(bundle, events)
    assert any("authoritative values must be positive zero" in error for error in check_bundle(bundle)["errors"])


def test_checker_import_isolation_does_not_load_producer_modules():
    script = "import sys; import safety_kernel.telemetry.checker; print(','.join(sorted(k for k in sys.modules if k.startswith('safety_kernel.telemetry.'))))"
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    loaded = set(completed.stdout.strip().split(","))
    assert "safety_kernel.telemetry.checker" in loaded
    assert not loaded & {"safety_kernel.telemetry.adapter", "safety_kernel.telemetry.bundle", "safety_kernel.telemetry.schema", "safety_kernel.telemetry.canonical"}


def test_crc_is_exact_newline_bytes_from_episode_start_and_zero_steps_work(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle", steps=0)
    lines = (bundle / "events.ndjson").read_bytes().splitlines(keepends=True)
    events = [json.loads(line) for line in lines]
    start_index = next(index for index, event in enumerate(events) if event["event_type"] == "episode_start")
    summary_index = next(index for index, event in enumerate(events) if event["event_type"] == "episode_summary")
    crc = 0
    for line in lines[start_index:summary_index]: crc = binascii.crc32(line, crc)
    assert events[summary_index]["episode_crc32"] == f"{crc & 0xffffffff:08x}"
    assert structural_replay(bundle)["ok"]


def test_finalize_is_separate_and_receipt_binds_all_components(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle")
    assert not check_bundle(bundle)["ok"]
    final = finalize_bundle(bundle)
    assert final["ok"] and (bundle / "checker.json").is_file() and (bundle / "completion_receipt.json").is_file()
    with pytest.raises(BundleImmutableError): finalize_bundle(bundle)


def test_checker_rejects_missing_or_orphan_float_hex_and_semantic_rehash(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle"); assert finalize_bundle(bundle)["ok"]
    events = _events(bundle); decision = next(event for event in events if event["event_type"] == "decision_evidence")
    decision["float_hex"].pop("/proposal/0")
    _rehash(events); _write_rehashed(bundle, events)
    assert any("missing wire float" in error for error in check_bundle(bundle)["errors"])
    bundle = _producer(tmp_path / "bundle2"); assert finalize_bundle(bundle)["ok"]
    events = _events(bundle); summary = next(event for event in events if event["event_type"] == "kernel_step_summary")
    summary["l0c_intervened"] = False
    _rehash(events); _write_rehashed(bundle, events)
    assert any("L0C intervention summary mismatch" in error for error in check_bundle(bundle)["errors"])
    bundle = _producer(tmp_path / "bundle3"); assert finalize_bundle(bundle)["ok"]
    events = _events(bundle); decision = next(event for event in events if event["event_type"] == "decision_evidence")
    decision["float_hex"]["/final_command/0"] = (0.25000004).hex()
    _rehash(events); _write_rehashed(bundle, events)
    assert any("authorized final command mismatch" in error for error in check_bundle(bundle)["errors"])


def test_exact_boundary_and_collision_do_not_use_rounded_or_zero_command_exemptions(tmp_path: Path):
    evidence = {"q": [0.0] * 5, "qdot_exec": [1.00000004, 0.0, 0.0, 0.0, 0.0], "qdot_max_eroded": [1.0] * 5, "q_min_eroded": [-2.0] * 5, "q_max_eroded": [2.0] * 5, "dt": 0.02, "mass_matrix": [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)], "e_max_safe": 5.0, "d_safe": 0.02, "collision_distances": {"a-b": 0.0}}
    bundle = _producer(tmp_path / "bundle", l0r_evidence=evidence); assert finalize_bundle(bundle)["h3"]["predicates"]["l0r"]["status"] == "fail"
    output = check_bundle(bundle)
    assert any("velocity" in failure for failure in output["h3"]["predicates"]["l0r"]["failures"])
    assert any("collision" in failure for failure in output["h3"]["predicates"]["l0r"]["failures"])


def test_determinism_truncation_unknown_and_duplicate_rejected(tmp_path: Path):
    first, second = _producer(tmp_path / "first"), _producer(tmp_path / "second")
    assert (first / "events.ndjson").read_bytes() == (second / "events.ndjson").read_bytes()
    raw = (first / "events.ndjson").read_bytes().rstrip(b"\n"); (first / "events.ndjson").write_bytes(raw)
    assert not check_bundle(first, require_complete=False)["ok"]
    bundle = _producer(tmp_path / "third")
    line = (bundle / "events.ndjson").read_bytes().splitlines()[0]
    (bundle / "events.ndjson").write_bytes(line[:-1] + b',"timestamp_ns":0}\n')
    assert any("duplicate JSON key" in error for error in check_bundle(bundle, require_complete=False)["errors"])


def test_run_identity_time_and_closed_manifest_are_bound(tmp_path: Path):
    bundle = _producer(tmp_path / "identity")
    events = _events(bundle)
    for event in events:
        if event["event_type"] in {"episode_start", "episode_summary"}:
            event["contract_hash"] = "1" * 64
    _rehash(events)
    (bundle / "events.ndjson").write_bytes(
        b"".join(event_bytes(event) + b"\n" for event in events)
    )
    assert any("RFC-0001 contract identity mismatch" in error for error in check_bundle(bundle, require_complete=False)["errors"])

    bundle = _producer(tmp_path / "time")
    events = _events(bundle)
    decision = next(event for event in events if event["event_type"] == "decision_evidence")
    decision["timestamp_ns"] = 1
    _rehash(events)
    (bundle / "events.ndjson").write_bytes(
        b"".join(event_bytes(event) + b"\n" for event in events)
    )
    assert any("timestamp does not match simulated step time" in error for error in check_bundle(bundle, require_complete=False)["errors"])

    bundle = _producer(tmp_path / "manifest")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["unknown"] = True
    (bundle / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    assert "manifest.json must be a closed Gate 2 record" in check_bundle(bundle, require_complete=False)["errors"]


def test_episode_level_shift_step_minus_one_is_part_of_the_frozen_abi():
    event = prepare_event({
        "event_type": "shift_injected", "timestamp_ns": 0,
        "episode_id": "episode-0", "step": -1,
        "shift_type": "safety_io_flip", "shift_params": {"signal": "estop_ok"},
        "affected_channels": ["safety_io"], "prev_hash": "0" * 64,
    })
    assert validate_producer_event(event) == []
    assert validate_independent_event(event) == []


def test_producer_and_independent_validators_close_nested_shift_and_boolean_scalars(tmp_path: Path):
    """Frozen 0.2.0 rejects recursive mutation before checker replay."""
    events = _events(_producer(tmp_path / "recursive-wire"))
    shift = next(event for event in events if event["event_type"] == "shift_injected")
    shift["shift_type"] = "sensor_stale"
    shift["shift_params"] = {"frame_age_ms": {"unknown_nested_field": True}}
    shift["affected_channels"] = ["sensor_health"]
    decision = next(event for event in events if event["event_type"] == "decision_evidence")
    decision["rta_nearmiss"] = 0
    episode = next(event for event in events if event["event_type"] == "episode_summary")
    episode["state_transitions"] = [{"step": False, "from": "ACTIVE", "to": "DEGRADED", "trigger": "front_cam_health"}]
    episode["state_transition_count"] = 1
    for event in (shift, decision, episode):
        assert validate_producer_event(event)
        assert validate_independent_event(event)


def test_acceptance_requires_sla_coverage_even_when_h3_has_another_failure(tmp_path: Path):
    shift = ShiftSpec(
        start_step=0, end_step=None, shift_type="sensor_stale",
        params={"frame_age_ms": 999.0},
    )
    env = CellArmEnv(
        contract_path="contracts/cell_arm_track_a.yaml", shift_schedule=[shift],
        max_steps=27, seed=13,
    )
    env.reset(seed=13)
    manifest = _manifest(steps=27, bundle_kind="acceptance", seeds=[13])
    manifest["monitor_slas"] = {
        "sensor_stale": {
            "monitor": "front_cam_health", "max_steps": 26, "to_state": "DEGRADED",
        }
    }
    bundle = tmp_path / "sensor-sla"
    run = TelemetryRun(
        RunBundle(bundle, manifest=manifest, contract=_contract_envelope(), environment=_environment_for(env)),
        total_episodes=1,
    )
    run.start_episode("episode-0", 13, planned_steps=27)
    for step in range(27):
        info = env.step([0.1] * 5)[-1]
        shifts = [{"shift_type": "sensor_stale", "params": {"frame_age_ms": 999.0}, "affected_channels": ["sensor_health"]}] if step == 0 else []
        run.record_step(info, shifts=shifts)
    run.end_episode(); run.complete(timestamp_ns=540_000_000)
    # Preserve complete Gate 2 evidence while making strict H3 independently
    # fail; absent SLA coverage must still make acceptance non-evidentiary.
    events = _events(bundle)
    summary = next(event for event in events if event["event_type"] == "kernel_step_summary")
    pair = next(iter(summary["l0r_evidence"]["collision_distances"]))
    pointer = f"/l0r_evidence/collision_distances/{pair}"
    summary["l0r_evidence"]["collision_distances"][pair] = "-1.000000"
    summary["float_hex"][pointer] = (-1.0).hex()
    _rehash(events)
    start = next(index for index, event in enumerate(events) if event["event_type"] == "episode_start")
    end = next(index for index, event in enumerate(events) if event["event_type"] == "episode_summary")
    crc = 0
    for event in events[start:end]:
        crc = binascii.crc32(event_bytes(event) + b"\n", crc)
    events[end]["episode_crc32"] = f"{crc & 0xffffffff:08x}"
    _rehash(events)
    (bundle / "events.ndjson").write_bytes(b"".join(event_bytes(event) + b"\n" for event in events))
    baseline = finalize_bundle(bundle)
    assert baseline["evidence_ok"] is True and baseline["h3"]["overall"] == "fail", baseline["errors"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["monitor_slas"] = {}
    (bundle / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    result = check_bundle(bundle)
    assert result["evidence_ok"] is False
    assert "acceptance bundle lacks required detection_sla evidence" in result["errors"]


def test_l0c_intervention_layer_info_is_bound_to_decision_and_cap_evidence(tmp_path: Path):
    bundle = _producer(tmp_path / "l0c-layer")
    assert finalize_bundle(bundle)["evidence_ok"] is True
    events = _events(bundle)
    intervention = next(event for event in events if event.get("layer") == "L0C")
    intervention["layer_info"]["tcp_speed_after"] = "123.000000"
    intervention["float_hex"]["/layer_info/tcp_speed_after"] = (123.0).hex()
    _rehash(events); _write_rehashed(bundle, events)
    result = check_bundle(bundle)
    assert result["evidence_ok"] is False
    assert any("tcp_speed_after" in error for error in result["errors"])


def test_runtime_authority_and_pre_step_evidence_reach_adapter(tmp_path: Path):
    bundle = tmp_path / "runtime"
    contract = _contract_envelope()
    env = CellArmEnv(contract_path="contracts/cell_arm_track_a.yaml", max_steps=1, seed=7)
    env.reset(seed=7)
    run = TelemetryRun(
            RunBundle(bundle, manifest=_manifest(seeds=[7]), contract=contract, environment=_environment_for(env)),
        total_episodes=1,
    )
    proposal = [0.01, 0.02, 0.03, 0.04, 0.05]
    _, _, _, _, info = env.step(proposal)
    run.start_episode("episode-000", 7, planned_steps=1)
    run.record_step(info)
    run.end_episode(); run.complete(timestamp_ns=20_000_000)
    assert finalize_bundle(bundle)["ok"]
    decision = next(event for event in _events(bundle) if event["event_type"] == "decision_evidence")
    assert decision["proposal"] == [f"{value:.6f}" for value in proposal]
    assert decision["authority_state_pre"] == "ACTIVE"
    assert decision["monitor_inputs"] and decision["monitor_results"]
    for field in ("q", "qdot", "tau"):
        assert decision["truth"][field] == decision["observed"][field]


def test_replay_episode_boundaries_are_not_lexically_ordered(tmp_path: Path):
    bundle = tmp_path / "many"
    run = TelemetryRun(RunBundle(bundle, manifest=_manifest(12), contract=_contract_envelope(), environment=_environment()), total_episodes=12)
    for episode in range(12):
        run.start_episode(f"episode-{episode}", episode, planned_steps=1)
        run.record_step(_info(0))
        run.end_episode()
    run.complete(timestamp_ns=12 * 20_000_000)
    assert finalize_bundle(bundle)["ok"]


def test_checker_artifact_cannot_self_attest(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle"); assert finalize_bundle(bundle)["ok"]
    checker = json.loads((bundle / "checker.json").read_text())
    checker["ok"] = False
    (bundle / "checker.json").write_text(json.dumps(checker, sort_keys=True, separators=(",", ":")))
    receipt = json.loads((bundle / "completion_receipt.json").read_text())
    receipt["checker_sha256"] = hashlib.sha256((bundle / "checker.json").read_bytes()).hexdigest()
    (bundle / "completion_receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    assert "checker artifact does not match independent recomputation" in check_bundle(bundle)["errors"]


def test_contract_is_source_envelope_and_nested_l0r_unknowns_fail_closed(tmp_path: Path):
    with pytest.raises(TelemetryError, match="verifiable RFC-0001 envelope"):
        TelemetryRun(RunBundle(tmp_path / "hash-only", manifest=_manifest(0, 0), contract={"contract_hash": "1" * 64}, environment=_environment()), total_episodes=0)
    evidence = {"q": [0.0] * 5, "qdot_exec": [0.0] * 5, "qdot_max_eroded": [1.0] * 5, "q_min_eroded": [-2.0] * 5, "q_max_eroded": [2.0] * 5, "dt": 0.02, "mass_matrix": [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)], "e_max_safe": 5.0, "d_safe": 0.02, "collision_distances": {"a-b": 1.0}, "unknown": 1}
    with pytest.raises(TelemetryError, match="l0r evidence fields"):
        _producer(tmp_path / "nested", l0r_evidence=evidence)


def test_replay_rejects_missing_mandatory_state_change(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle", rta=True); assert finalize_bundle(bundle)["ok"]
    events = [event for event in _events(bundle) if event["event_type"] != "contract_state_change"]
    _rehash(events); _write_rehashed(bundle, events)
    assert any("mandatory state change" in error for error in check_bundle(bundle)["errors"])


def test_runtime_missing_monitor_evidence_serializes_typed_denial_and_replays(tmp_path: Path):
    """The adapter's last-resort denial must be checker-visible, not a sentinel."""
    bundle = tmp_path / "unavailable"
    run = TelemetryRun(RunBundle(bundle, manifest=_manifest(), contract=_contract_envelope(), environment=_environment()), total_episodes=1)
    info = _info(0, rta=True)
    authority = info["authority_info"]
    authority.pop("monitor_inputs"); authority.pop("monitor_results")
    authority["reason"] = "monitor_evidence_missing"
    info["contract_events"][0]["trigger_monitor"] = "monitor_evidence_missing"
    info["contract_events"][0]["monitors_status"] = {
        "monitor_evidence_missing": {"triggered": True, "violated": True}
    }
    run.start_episode("episode-0", 0, planned_steps=1)
    run.record_step(info); run.end_episode(); run.complete(timestamp_ns=20_000_000)
    assert finalize_bundle(bundle)["evidence_ok"]
    decision = next(event for event in _events(bundle) if event["event_type"] == "decision_evidence")
    assert decision["monitor_inputs"] == decision["monitor_results"] == {"availability": "unavailable", "reason": "monitor evidence unavailable"}


def test_cell_arm_empty_monitor_inputs_are_closed_denials_on_each_tick(tmp_path: Path, monkeypatch):
    """The actual runtime seam must not turn {} into synthetic monitor evidence."""
    env = CellArmEnv(contract_path="contracts/cell_arm_track_a.yaml", max_steps=2, seed=4)
    env.reset(seed=4)
    monkeypatch.setattr(env._shift_injector, "get_monitor_inputs", lambda step: {})
    run = TelemetryRun(RunBundle(tmp_path / "missing", manifest=_manifest(steps=2, seeds=[4]), contract=_contract_envelope(), environment=_environment_for(env)), total_episodes=1)
    run.start_episode("episode-0", 4, planned_steps=2)
    for _ in range(2):
        info = env.step([0.1] * 5)[-1]
        assert info["authority_info"]["monitor_inputs"] is None
        assert info["authority_info"]["final_command"] == [0.0] * 5
        run.record_step(info)
    run.end_episode(); run.complete(timestamp_ns=40_000_000)
    result = finalize_bundle(tmp_path / "missing")
    assert result["evidence_ok"], result["errors"]
    decisions = [event for event in _events(tmp_path / "missing") if event["event_type"] == "decision_evidence"]
    assert len(decisions) == 2 and all(event["monitor_inputs"]["availability"] == "unavailable" for event in decisions)


def test_acceptance_completeness_is_per_summary_record(tmp_path: Path):
    bundle = _producer(tmp_path / "record-completeness")
    manifest = json.loads((bundle / "manifest.json").read_text()); manifest["bundle_kind"] = "acceptance"
    (bundle / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    events = _events(bundle)
    summary = next(event for event in events if event["event_type"] == "kernel_step_summary")
    summary["l0r_evidence"] = None
    _rehash(events)
    (bundle / "events.ndjson").write_bytes(b"".join(event_bytes(event) + b"\n" for event in events))
    result = check_bundle(bundle, require_complete=False)
    assert any("acceptance record lacks L0-R evidence" in error for error in result["errors"])


def test_closed_episode_and_shift_scalars_reject_bool_and_inconsistent_revocation(tmp_path: Path):
    bundle = _producer(tmp_path / "closed")
    events = _events(bundle)
    episode = next(event for event in events if event["event_type"] == "episode_summary")
    episode["total_interventions"]["RTA"] = True
    episode["synthetic_revocation"] = False
    episode["synthetic_revocation_params"] = {"trigger_step": 0, "duration_steps": 1, "injected_monitor": "x"}
    episode["l1_arm"] = "L1"
    shift = next(event for event in events if event["event_type"] == "shift_injected")
    shift["affected_channels"] = ["bogus"]
    _rehash(events)
    (bundle / "events.ndjson").write_bytes(b"".join(event_bytes(event) + b"\n" for event in events))
    errors = check_bundle(bundle, require_complete=False)["errors"]
    assert any("invalid intervention totals" in error for error in errors)
    assert any("affected_channels" in error for error in errors)


def test_checker_rejects_invented_transition_snapshot_and_nearmiss_summary(tmp_path: Path):
    bundle = _producer(tmp_path / "causal", rta=True)
    events = _events(bundle)
    change = next(event for event in events if event["event_type"] == "contract_state_change")
    change["monitors_status"]["estop_chain_ok"]["triggered"] = False
    summary = next(event for event in events if event["event_type"] == "kernel_step_summary")
    summary["rta_nearmiss"] = True
    _rehash(events)
    (bundle / "events.ndjson").write_bytes(b"".join(event_bytes(event) + b"\n" for event in events))
    errors = check_bundle(bundle, require_complete=False)["errors"]
    assert any("snapshot" in error for error in errors)
    assert any("near-miss summary" in error for error in errors)


def test_acceptance_rejects_typed_unavailable_monitor_denial(tmp_path: Path):
    bundle = tmp_path / "acceptance-unavailable"
    run = TelemetryRun(RunBundle(bundle, manifest=_manifest(bundle_kind="acceptance"), contract=_contract_envelope(), environment=_environment()), total_episodes=1)
    info = _info(0, rta=True)
    authority = info["authority_info"]
    authority.pop("monitor_inputs"); authority.pop("monitor_results")
    authority["reason"] = "monitor_evidence_missing"
    info["contract_events"][0]["trigger_monitor"] = "monitor_evidence_missing"
    run.start_episode("episode-0", 0, planned_steps=1)
    run.record_step(info); run.end_episode(); run.complete(timestamp_ns=20_000_000)
    result = finalize_bundle(bundle)
    assert result["evidence_ok"] is False
    assert "acceptance bundle contains unavailable monitor evidence" in result["errors"]


def test_independent_replay_accepts_runtime_operator_reset(tmp_path: Path):
    shift = ShiftSpec(
        start_step=0, end_step=2, shift_type="safety_io_flip",
        params={"signal": "estop_ok"},
    )
    env = CellArmEnv(
        contract_path="contracts/cell_arm_track_a.yaml",
        shift_schedule=[shift], max_steps=3, seed=0,
    )
    env.reset(seed=0)
    bundle = tmp_path / "reset"
    run = TelemetryRun(
        RunBundle(
            bundle, manifest=_manifest(steps=3),
            contract=_contract_envelope(), environment=_environment_for(env),
        ),
        total_episodes=1,
    )
    run.start_episode("episode-0", 0, planned_steps=3)
    run.record_step(env.step([0.1] * 5)[-1])
    run.record_step(env.step([0.1] * 5)[-1])
    # Reset is a request; its sole monitor evaluation and native transition are
    # emitted inside the following decision tick.
    assert env.operator_reset() == []
    run.record_step(env.step([0.1] * 5)[-1])
    run.end_episode()
    run.complete(timestamp_ns=60_000_000)
    result = finalize_bundle(bundle)
    assert result["evidence_ok"] is True, result["errors"]
    decision = [
        event for event in _events(bundle)
        if event["event_type"] == "decision_evidence"
    ][-1]
    assert decision["reset_input"] is True
    assert decision["authority_state_pre"] == "RTA_STOP"
    assert decision["authority_state_post"] == "ACTIVE"


@pytest.mark.parametrize("monitor", [
    "anchors_consistent", "gate_interlock_ok", "estop_chain_ok",
    "front_cam_health", "robot_mount_integrity",
])
def test_independent_replay_rejects_each_monitor_family_counterexample(tmp_path: Path, monitor: str):
    bundle = _producer(tmp_path / monitor)
    events = _events(bundle)
    decision = next(event for event in events if event["event_type"] == "decision_evidence")
    # The verified inputs are healthy; one invented predicate result must not
    # become acceptable merely because the producer says it was observed.
    decision["monitor_results"][monitor]["predicate_violated"] = True
    _rehash(events)
    (bundle / "events.ndjson").write_bytes(b"".join(event_bytes(event) + b"\n" for event in events))
    assert any(f"independent monitor replay contradicts {monitor}" in error for error in check_bundle(bundle, require_complete=False)["errors"])


def test_independent_replay_rejects_debounce_counterexample(tmp_path: Path):
    bundle = _producer(tmp_path / "debounce")
    events = _events(bundle)
    decision = next(event for event in events if event["event_type"] == "decision_evidence")
    decision["monitor_results"]["front_cam_health"].update({"triggered": True, "should_trigger": True})
    _rehash(events)
    (bundle / "events.ndjson").write_bytes(b"".join(event_bytes(event) + b"\n" for event in events))
    assert any("independent monitor replay contradicts front_cam_health" in error for error in check_bundle(bundle, require_complete=False)["errors"])


def test_independent_replay_rejects_priority_counterexample(tmp_path: Path):
    shifts = [
        ShiftSpec(start_step=0, end_step=None, shift_type="safety_io_flip", params={"signal": "gate_closed"}),
        ShiftSpec(start_step=0, end_step=None, shift_type="safety_io_flip", params={"signal": "estop_ok"}),
    ]
    env = CellArmEnv(contract_path="contracts/cell_arm_track_a.yaml", shift_schedule=shifts, max_steps=1, seed=2)
    env.reset(seed=2)
    bundle = tmp_path / "priority"
    run = TelemetryRun(RunBundle(bundle, manifest=_manifest(seeds=[2]), contract=_contract_envelope(), environment=_environment_for(env)), total_episodes=1)
    run.start_episode("episode-0", 2, planned_steps=1); run.record_step(env.step([0.1] * 5)[-1]); run.end_episode(); run.complete(timestamp_ns=20_000_000)
    assert finalize_bundle(bundle)["evidence_ok"]
    events = _events(bundle)
    change = next(event for event in events if event["event_type"] == "contract_state_change")
    assert change["trigger_monitor"] == "estop_chain_ok"  # severity tie -> lexical name.
    change["trigger_monitor"] = "gate_interlock_ok"
    _rehash(events); _write_rehashed(bundle, events)
    assert any("independent state replay contradicts emitted transitions" in error for error in check_bundle(bundle)["errors"])


def test_repeated_rta_veto_is_zero_and_counted_for_each_nonzero_proposal(tmp_path: Path):
    shift = ShiftSpec(start_step=0, end_step=None, shift_type="safety_io_flip", params={"signal": "estop_ok"})
    env = CellArmEnv(contract_path="contracts/cell_arm_track_a.yaml", shift_schedule=[shift], max_steps=2, seed=3)
    env.reset(seed=3)
    run = TelemetryRun(RunBundle(tmp_path / "rta", manifest=_manifest(), contract=_contract_envelope(), environment=_environment_for(env)), total_episodes=1)
    run.start_episode("episode-0", 3, planned_steps=2)
    for _ in range(2):
        run.record_step(env.step([0.5] * 5)[-1])
    run.end_episode(); run.complete(timestamp_ns=40_000_000)
    events = _events(tmp_path / "rta")
    rta = [event for event in events if event.get("layer") == "RTA"]
    assert len(rta) == 2 and all(event["qdot_exec"] == ["0.000000"] * 5 for event in rta)
    summary = next(event for event in events if event["event_type"] == "episode_summary")
    assert summary["total_interventions"]["RTA"] == 2


def test_nested_canonicalization_is_insertion_order_invariant():
    left = prepare_event({"event_type": "shift_injected", "timestamp_ns": 0, "episode_id": "e", "step": 0, "shift_type": "anchor_drift", "shift_params": {"z": 1, "a": {"y": 2, "b": 3}}, "affected_channels": ["anchor_poses"], "prev_hash": "0" * 64})
    right = prepare_event({"event_type": "shift_injected", "timestamp_ns": 0, "episode_id": "e", "step": 0, "shift_type": "anchor_drift", "shift_params": {"a": {"b": 3, "y": 2}, "z": 1}, "affected_channels": ["anchor_poses"], "prev_hash": "0" * 64})
    assert event_bytes(left) == event_bytes(right)


def test_checker_recomputes_contract_and_model_identity(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle")
    contract = json.loads((bundle / "contract.json").read_text())
    contract["original_yaml"] = contract["original_yaml"].replace(
        "max_tcp_speed_mps: 0.20", "max_tcp_speed_mps: 999.0"
    )
    (bundle / "contract.json").write_text(json.dumps(contract, sort_keys=True, separators=(",", ":")))
    assert any("RFC-0001 verification failed" in error for error in check_bundle(bundle, require_complete=False)["errors"])

    bundle = _producer(tmp_path / "model")
    environment = json.loads((bundle / "environment.json").read_text())
    environment["model_xml"] += "\n<!-- tampered -->"
    (bundle / "environment.json").write_text(json.dumps(environment, sort_keys=True, separators=(",", ":")))
    assert any("embedded model" in error for error in check_bundle(bundle, require_complete=False)["errors"])


def test_acceptance_environment_cannot_self_declare_weaker_l0r_limits(tmp_path: Path):
    bundle = _producer(tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["bundle_kind"] = "acceptance"
    (bundle / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    environment = json.loads((bundle / "environment.json").read_text())
    environment["l0r_parameters"]["d_safe"] = -1.0
    (bundle / "environment.json").write_text(json.dumps(environment, sort_keys=True, separators=(",", ":")))
    assert any("d_safe does not match normative Track A parameters" in error for error in check_bundle(bundle, require_complete=False)["errors"])


def test_checker_binds_h3_state_to_decision_truth_and_final_command(tmp_path: Path):
    evidence = {"q": [0.0] * 5, "qdot_exec": [0.25] * 5, "qdot_max_eroded": [1.0] * 5, "q_min_eroded": [-2.0] * 5, "q_max_eroded": [2.0] * 5, "dt": 0.02, "mass_matrix": [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)], "e_max_safe": 5.0, "d_safe": 0.02, "collision_distances": {"a-b": 1.0}}
    bundle = _producer(tmp_path / "bundle", l0r_evidence=evidence); finalize_bundle(bundle)
    events = _events(bundle)
    summary = next(event for event in events if event["event_type"] == "kernel_step_summary")
    summary["l0r_evidence"]["qdot_exec"][0] = "0.200000"
    summary["float_hex"]["/l0r_evidence/qdot_exec/0"] = (0.2).hex()
    _rehash(events); _write_rehashed(bundle, events)
    assert any("disconnected from decision" in error for error in check_bundle(bundle)["errors"])


def test_sla_requires_named_monitor_and_flap_limiter_boundary():
    shift = {"event_type": "shift_injected", "episode_id": "e", "step": 0, "shift_type": "anchor_drift", "shift_params": {"anchor": "a"}}
    unrelated = {"event_type": "contract_state_change", "episode_id": "e", "step": 1, "to_state": "REVOKED", "trigger_monitor": "other"}
    decision = {"event_type": "decision_evidence", "episode_id": "e", "step": 1, "monitor_results": {"anchors_consistent": {"predicate_violated": True, "triggered": True}}}
    manifest = {"monitor_slas": {"anchor_drift": {"max_steps": 2, "to_state": "REVOKED", "monitor": "anchors_consistent"}}}
    assert _h3_sla([shift, unrelated, decision], manifest)["status"] == "fail"

    transitions = []
    for index in range(4):
        transitions.extend([
            {"event_type": "contract_state_change", "episode_id": "e", "step": index * 2, "from_state": "ACTIVE", "to_state": "DEGRADED", "trigger_monitor": "m"},
            {"event_type": "contract_state_change", "episode_id": "e", "step": index * 2 + 1, "from_state": "DEGRADED", "to_state": "ACTIVE", "trigger_monitor": "m"},
        ])
    transitions.append({"event_type": "contract_state_change", "episode_id": "e", "step": 99, "from_state": "DEGRADED", "to_state": "RTA_STOP", "trigger_monitor": "flap_limiter"})
    # Native recovery guards begin only after a completed DEGRADED→ACTIVE
    # recovery, so four raw cycles contain three eligible re-degradations.
    assert _h3_flapping(transitions, {"flap_limit": 4, "flap_hysteresis_ticks": 0})["status"] == "pass"


def test_checker_flap_vectors_mirror_native_recovery_guard_only():
    recovery_then_redegrade = [
        {"event_type": "contract_state_change", "episode_id": "e", "step": 0, "from_state": "ACTIVE", "to_state": "DEGRADED"},
        {"event_type": "contract_state_change", "episode_id": "e", "step": 1, "from_state": "DEGRADED", "to_state": "ACTIVE"},
        {"event_type": "contract_state_change", "episode_id": "e", "step": 2, "from_state": "ACTIVE", "to_state": "DEGRADED"},
    ]
    assert _h3_flapping(recovery_then_redegrade, {"flap_limit": 2, "flap_hysteresis_ticks": 2})["flaps"] == {"e": 1}
    late = [dict(item) for item in recovery_then_redegrade]
    late[2]["step"] = 3
    assert _h3_flapping(late, {"flap_limit": 2, "flap_hysteresis_ticks": 2})["flaps"] == {"e": 0}


def test_checker_flap_exact_threshold_late_limiter_and_outside_window():
    exact = [
        {"event_type": "contract_state_change", "episode_id": "e", "step": 0, "from_state": "ACTIVE", "to_state": "DEGRADED", "trigger_monitor": "m"},
        {"event_type": "contract_state_change", "episode_id": "e", "step": 1, "from_state": "DEGRADED", "to_state": "ACTIVE", "trigger_monitor": "m"},
        {"event_type": "contract_state_change", "episode_id": "e", "step": 2, "from_state": "ACTIVE", "to_state": "DEGRADED", "trigger_monitor": "m"},
        {"event_type": "contract_state_change", "episode_id": "e", "step": 3, "from_state": "DEGRADED", "to_state": "ACTIVE", "trigger_monitor": "m"},
        {"event_type": "contract_state_change", "episode_id": "e", "step": 4, "from_state": "ACTIVE", "to_state": "DEGRADED", "trigger_monitor": "m"},
        {"event_type": "contract_state_change", "episode_id": "e", "step": 4, "from_state": "DEGRADED", "to_state": "RTA_STOP", "trigger_monitor": "flap_limiter"},
    ]
    manifest = {"flap_limit": 2, "flap_hysteresis_ticks": 2}
    result = _h3_flapping(exact, manifest)
    assert result["status"] == "pass" and result["flaps"] == {"e": 2}

    late_limiter = [dict(event) for event in exact]
    late_limiter[-1]["step"] = 5
    assert _h3_flapping(late_limiter, manifest)["status"] == "fail"

    outside = [dict(event) for event in exact[:-1]]
    outside[2]["step"] = 3  # guard end is exclusive: 1 + 2 == 3.
    outside[3]["step"] = 4
    outside[4]["step"] = 6  # second guard also expires at 4 + 2 == 6.
    result = _h3_flapping(outside, manifest)
    assert result["status"] == "pass" and result["flaps"] == {"e": 0}


def test_check_cli_accepts_gate2_evidence_while_reporting_h3_fail(tmp_path: Path):
    evidence = {"q": [0.0] * 5, "qdot_exec": [0.25] * 5, "qdot_max_eroded": [1.0] * 5, "q_min_eroded": [-2.0] * 5, "q_max_eroded": [2.0] * 5, "dt": 0.02, "mass_matrix": [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)], "e_max_safe": 5.0, "d_safe": 0.02, "collision_distances": {"a-b": 0.0}}
    bundle = _producer(tmp_path / "bundle", l0r_evidence=evidence); finalize_bundle(bundle)
    completed = subprocess.run([sys.executable, "-m", "safety_kernel", "check", str(bundle)], capture_output=True, text=True, check=False)
    output = json.loads(completed.stdout)
    assert completed.returncode == 0 and output["evidence_ok"] is True and output["h3"]["overall"] == "fail"


def _events(bundle: Path) -> list[dict]: return [json.loads(line) for line in (bundle / "events.ndjson").read_text().splitlines()]

def _rehash(events: list[dict]) -> None:
    previous = "0" * 64
    for event in events:
        event["prev_hash"] = previous
        if event["event_type"] == "run_end": event["last_event_hash"] = previous
        previous = event_hash(event)

def _write_rehashed(bundle: Path, events: list[dict]) -> None:
    contents = b"".join(event_bytes(event) + b"\n" for event in events)
    (bundle / "events.ndjson").write_bytes(contents)
    receipt = json.loads((bundle / "completion_receipt.json").read_text())
    receipt["events_sha256"], receipt["final_event_hash"] = hashlib.sha256(contents).hexdigest(), event_hash(events[-1])
    (bundle / "completion_receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
