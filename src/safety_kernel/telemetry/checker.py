"""Independent fail-closed checking, finalization, and structural replay."""

from __future__ import annotations

import binascii
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from safety_kernel.telemetry.independent import (
    canonical_json_bytes, event_bytes, event_hash, exact as independent_exact,
    validate_event,
)
from safety_kernel.telemetry.errors import BundleImmutableError, TelemetryError
from safety_kernel.contract.loader import load_contract_text

COMMAND_AUTHORITY = "software_command_suppression_only"
SCHEMA_VERSION = "0.2.0"
_HASH = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED = {"manifest.json", "contract.json", "environment.json", "events.ndjson", "checker.json", "completion_receipt.json"}
_STATES = {"ACTIVE", "DEGRADED", "REVOKED", "RTA_STOP"}
_LEGAL = {("ACTIVE", "degrade", "DEGRADED"), ("ACTIVE", "revoke", "REVOKED"), ("ACTIVE", "rta_stop", "RTA_STOP"),
          ("DEGRADED", "revoke", "REVOKED"), ("DEGRADED", "clear", "ACTIVE"), ("DEGRADED", "rta_stop", "RTA_STOP"),
          ("REVOKED", "reset", "ACTIVE"), ("REVOKED", "rta_stop", "RTA_STOP"), ("RTA_STOP", "reset", "ACTIVE")}
_TRACK_A_COLLISIONS = ["geom_link_0-geom_link_3", "geom_link_0-geom_link_4", "geom_link_0-geom_link_5", "geom_link_1-geom_link_4", "geom_link_1-geom_link_5"]


def check_bundle(path: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    root, errors = Path(path), []
    if not root.is_dir():
        return _result(False, [f"bundle directory missing: {root}"], {}, {})
    if require_complete:
        missing = sorted(name for name in _REQUIRED if not (root / name).is_file())
        if missing:
            errors.append(f"bundle incomplete; missing {missing!r}")
    components = _components(root, errors)
    _manifest_semantics(components.get("manifest"), errors)
    _component_semantics(components, errors)
    _manifest_environment_binding(components.get("manifest"), components.get("environment"), errors)
    events, raw_lines = _events(root / "events.ndjson", errors)
    _contract_evidence_binding(events, components.get("contract"), errors)
    flapping = _contract_temporal_replay(
        events, components.get("contract"), components.get("manifest", {}), errors
    )
    integrity = _integrity(events, raw_lines, components, errors)
    _l0r_binding(events, components.get("environment"), errors)
    replay = structural_replay_events(events, components.get("manifest", {}))
    errors.extend(replay["errors"])
    h3 = _h3(events, components.get("manifest", {}), flapping=flapping)
    _acceptance_evidence(events, components.get("manifest", {}), h3, errors)
    checker_expected = _result(
        not errors and h3["overall"] != "fail",
        list(errors),
        integrity,
        {"h3": h3, "replay": replay["summary"]},
    )
    if "checker" in components and components["checker"] != checker_expected:
        errors.append("checker artifact does not match independent recomputation")
    if "receipt" in components:
        errors.extend(_receipt_errors(root, components["receipt"], events))
    return _result(not errors and h3["overall"] != "fail", errors, integrity, {"h3": h3, "replay": replay["summary"]})


def finalize_bundle(path: str | Path) -> dict[str, Any]:
    """Run the checker before writing immutable checker and receipt artifacts."""
    preliminary = check_bundle(path, require_complete=False)
    if preliminary["errors"]:
        return preliminary
    _write_finalization(path, preliminary)
    return check_bundle(path)


def structural_replay(path: str | Path) -> dict[str, Any]:
    root, errors = Path(path), []
    components = _components(root, errors)
    events, _ = _events(root / "events.ndjson", errors)
    result = structural_replay_events(events, components.get("manifest", {}))
    result["errors"] = errors + result["errors"]
    result["ok"] = not result["errors"]
    return result


def structural_replay_events(events: list[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    active: str | None = None
    episodes: dict[str, dict[str, Any]] = {}
    rank = {"shift_injected": 0, "decision_evidence": 1, "safety_intervention:L0C": 2,
            "safety_intervention:L0R": 3, "safety_intervention:RTA": 4, "contract_state_change": 5,
            "kernel_step_summary": 6}
    last_key: tuple[str, int, int] | None = None
    step_records: dict[tuple[str, int], list[dict[str, Any]]] = {}
    last_timestamp = -1
    for index, event in enumerate(events):
        kind = event.get("event_type"); counts[str(kind)] += 1
        if "timestamp_ns" in event:
            if kind == "episode_start":
                last_timestamp = -1  # simulator episode clocks intentionally reset
            timestamp = event.get("timestamp_ns")
            if not isinstance(timestamp, int) or timestamp < last_timestamp:
                errors.append(f"non-monotonic timestamp at index {index}")
            elif isinstance(timestamp, int):
                last_timestamp = timestamp
        if index == 0 and kind != "run_start": errors.append("first event must be run_start")
        if kind == "episode_start":
            episode_id = event.get("episode_id")
            if active is not None or not isinstance(episode_id, str) or episode_id in episodes:
                errors.append("invalid or overlapping episode_start")
            else:
                active = episode_id
                episodes[episode_id] = {"planned": event.get("planned_steps"), "seed": event.get("seed"), "contract_hash": event.get("contract_hash"), "steps": [], "summarized": False, "state": "ACTIVE", "transitions": [], "interventions": Counter({"L0C": 0, "L0R": 0, "L1": 0, "RTA": 0})}
                last_key = None
        elif kind in {"shift_injected", "decision_evidence", "safety_intervention", "contract_state_change", "kernel_step_summary"}:
            episode_id, step = event.get("episode_id"), event.get("step")
            minimum_step = -1 if kind == "shift_injected" else 0
            if active != episode_id or not isinstance(step, int) or step < minimum_step:
                errors.append(f"step event outside active episode at index {index}"); continue
            key_name = kind if kind != "safety_intervention" else f"{kind}:{event.get('layer')}"
            if key_name not in rank:
                errors.append(f"invalid causal event at index {index}"); continue
            key = (episode_id, step, rank[key_name])
            if last_key is not None and key < last_key: errors.append(f"non-deterministic event ordering at index {index}")
            last_key = key; step_records.setdefault((episode_id, step), []).append(event)
            if kind == "kernel_step_summary":
                episodes[episode_id]["steps"].append(step)
                _step_semantics(event, step_records[(episode_id, step)], episodes[episode_id], errors)
        elif kind == "episode_summary":
            episode_id = event.get("episode_id")
            if active != episode_id: errors.append("episode_summary without active matching episode")
            elif event.get("total_steps") != len(episodes[episode_id]["steps"]): errors.append(f"episode total_steps mismatch for {episode_id}")
            elif event.get("seed") != episodes[episode_id]["seed"] or event.get("contract_hash") != episodes[episode_id]["contract_hash"]: errors.append(f"episode identity mismatch for {episode_id}")
            elif event.get("final_state") != episodes[episode_id]["state"]: errors.append(f"episode final_state mismatch for {episode_id}")
            elif event.get("state_transition_count") != len(episodes[episode_id]["transitions"]): errors.append(f"episode transition count mismatch for {episode_id}")
            elif event.get("state_transitions") != episodes[episode_id]["transitions"]: errors.append(f"episode transition summary mismatch for {episode_id}")
            elif event.get("total_interventions") != dict(episodes[episode_id]["interventions"]): errors.append(f"episode intervention count mismatch for {episode_id}")
            else: episodes[episode_id]["summarized"] = True; active = None
        elif kind == "run_end":
            if active is not None: errors.append("run_end precedes episode completion")
            if index != len(events) - 1: errors.append("run_end must be last")
    if active is not None: errors.append("truncated active episode")
    for episode_id, data in episodes.items():
        if data["steps"] != list(range(len(data["steps"]))) or data["planned"] != len(data["steps"]):
            errors.append(f"zero-based contiguous step evidence mismatch for {episode_id}")
    if counts["run_start"] != 1 or counts["run_end"] != 1: errors.append("bundle requires exactly one run_start and run_end")
    if events and events[-1].get("event_type") != "run_end": errors.append("truncated bundle lacks final run_end")
    if isinstance(manifest.get("total_episodes"), int) and manifest["total_episodes"] != len(episodes): errors.append("episode count does not match manifest")
    if isinstance(manifest.get("seed_schedule"), list):
        observed = [episodes[episode]["seed"] for episode in episodes]
        if observed != manifest["seed_schedule"]: errors.append("episode seeds do not match manifest seed_schedule")
    if isinstance(manifest.get("steps_per_episode"), int) and any(data["planned"] != manifest["steps_per_episode"] for data in episodes.values()):
        errors.append("episode planned steps do not match manifest steps_per_episode")
    if events:
        start, end = events[0], events[-1]
        if start.get("total_episodes") != manifest.get("total_episodes"):
            errors.append("run_start total_episodes does not match manifest")
        if end.get("event_type") == "run_end" and end.get("timestamp_ns") != manifest.get("completion_timestamp_ns"):
            errors.append("run_end timestamp does not match manifest completion_timestamp_ns")
    return {"errors": errors, "summary": {"event_counts": dict(counts), "episodes": sorted(episodes), "final_states": {key: value["state"] for key, value in episodes.items()}}}


def _step_semantics(summary: Mapping[str, Any], records: list[Mapping[str, Any]], episode: dict[str, Any], errors: list[str]) -> None:
    location = f"{summary.get('episode_id')}:{summary.get('step')}"
    decision = [event for event in records if event.get("event_type") == "decision_evidence"]
    if len(decision) != 1:
        errors.append(f"missing or duplicate decision evidence at {location}"); return
    decision = decision[0]
    if decision.get("authority_state_pre") != episode["state"]:
        errors.append(f"decision pre-state discontinuity at {location}")
    l0c = _layer(records, "L0C"); l0r = _layer(records, "L0R"); rta = _layer(records, "RTA")
    if any(len(layer) > 1 for layer in (l0c, l0r, rta)):
        errors.append(f"duplicate safety intervention layer at {location}")
    if bool(summary.get("l0c_intervened")) != bool(l0c): errors.append(f"L0C intervention summary mismatch at {location}")
    if bool(summary.get("l0r_intervened")) != bool(l0r): errors.append(f"L0R intervention summary mismatch at {location}")
    try:
        if bool(summary.get("l0c_intervened")) != (_exact(summary, "/l0c_scaling_factor") < 1.0): errors.append(f"L0C scaling semantics mismatch at {location}")
    except TelemetryError: errors.append(f"missing exact L0C scaling evidence at {location}")
    try:
        l0c_matches = not l0c or (_same_vector(l0c[0], "/qdot_cmd", decision, "/proposal") and _same_vector(l0c[0], "/qdot_exec", decision, "/l0c_command"))
        l0r_matches = not l0r or (_same_vector(l0r[0], "/qdot_cmd", decision, "/l0c_command") and _same_vector(l0r[0], "/qdot_exec", decision, "/l0r_command"))
        final_matches_l0r = _same_vector(decision, "/l0r_command", decision, "/final_command")
        veto_changed = decision.get("authorized") is False and not _same_vector(decision, "/l0r_command", decision, "/final_command")
        rta_matches = not rta or (_same_vector(rta[0], "/qdot_cmd", decision, "/l0r_command") and _same_vector(rta[0], "/qdot_exec", decision, "/final_command"))
        l0r_evidence = summary.get("l0r_evidence")
        evidence_matches = l0r_evidence is None or (
            _same_vector(summary, "/l0r_evidence/q", decision, "/truth/q")
            and _same_vector(summary, "/l0r_evidence/qdot_exec", decision, "/final_command")
        )
    except TelemetryError:
        errors.append(f"missing exact command evidence at {location}")
        return
    if not l0c_matches: errors.append(f"L0C causal command mismatch at {location}")
    if not l0r_matches: errors.append(f"L0R causal command mismatch at {location}")
    if bool(l0c) != (not _same_vector(decision, "/proposal", decision, "/l0c_command")):
        errors.append(f"L0C intervention omission/orphan at {location}")
    if bool(l0r) != (not _same_vector(decision, "/l0c_command", decision, "/l0r_command")):
        errors.append(f"L0R intervention omission/orphan at {location}")
    if bool(rta) != (not _same_vector(decision, "/l0r_command", decision, "/final_command")):
        errors.append(f"RTA intervention omission/orphan at {location}")
    if decision.get("authorized") is True and not final_matches_l0r: errors.append(f"authorized final command mismatch at {location}")
    if bool(rta) != veto_changed: errors.append(f"RTA veto/intervention mismatch at {location}")
    if not rta_matches: errors.append(f"RTA causal command mismatch at {location}")
    if not evidence_matches: errors.append(f"L0-R evidence is disconnected from decision at {location}")
    if l0c:
        _l0c_command_binding(l0c[0], decision, summary, location, errors)
    if summary.get("rta_nearmiss") != decision.get("rta_nearmiss"):
        errors.append(f"RTA near-miss summary is disconnected from decision at {location}")
    for intervention in l0c + l0r + rta:
        for channel in ("truth", "observed", "believed"):
            if not _same_tree(intervention, f"/{channel}", decision, f"/{channel}"):
                errors.append(f"{intervention.get('layer')} {channel} evidence is disconnected from decision at {location}")
    changes = [event for event in records if event.get("event_type") == "contract_state_change"]
    unavailable = _typed_unavailable(decision)
    if unavailable:
        required = [
            change for change in changes
            if change.get("trigger_monitor") == "monitor_evidence_missing"
            and change.get("trigger_severity") == "rta_stop"
            and change.get("from_state") == decision.get("authority_state_pre")
            and change.get("to_state") == "RTA_STOP"
        ]
        expected_count = 0 if decision.get("authority_state_pre") == "RTA_STOP" else 1
        if len(required) != expected_count:
            errors.append(f"typed unavailable denial requires immediate monitor_evidence_missing RTA_STOP at {location}")
    current = episode["state"]
    for change in changes:
        if change.get("from_state") != current:
            errors.append(f"transition from-state discontinuity at {location}")
        current = change.get("to_state")
        episode["transitions"].append({"step": summary.get("step"), "from": change.get("from_state"), "to": change.get("to_state"), "trigger": change.get("trigger_monitor")})
    if decision.get("authority_state_pre") != decision.get("authority_state_post") and not changes:
        errors.append(f"mandatory state change missing at {location}")
    expected = decision.get("authority_state_post")
    if current != expected:
        errors.append(f"decision post-state transition mismatch at {location}")
    if summary.get("contract_state") != expected: errors.append(f"contract state summary mismatch at {location}")
    episode["state"] = summary.get("contract_state")
    for layer in ("L0C", "L0R", "RTA"):
        episode["interventions"][layer] += len(_layer(records, layer))


def _l0c_command_binding(
    intervention: Mapping[str, Any], decision: Mapping[str, Any],
    summary: Mapping[str, Any], location: str, errors: list[str],
) -> None:
    """Bind one L0-C intervention to this tick's commands and cap Jacobian."""
    cap, info = summary.get("l0c_cap_evidence"), intervention.get("layer_info")
    if not isinstance(cap, dict) or not isinstance(info, dict):
        errors.append(f"L0C intervention lacks command-stage cap evidence at {location}")
        return
    try:
        proposal = [_exact(decision, f"/proposal/{index}") for index in range(5)]
        command = [_exact(decision, f"/l0c_command/{index}") for index in range(5)]
        jacobian = [
            [_exact(summary, f"/l0c_cap_evidence/tcp_jacobian/{row}/{column}") for column in range(5)]
            for row in range(3)
        ]
        before = math.sqrt(sum(sum(row[column] * proposal[column] for column in range(5)) ** 2 for row in jacobian))
        after = math.sqrt(sum(sum(row[column] * command[column] for column in range(5)) ** 2 for row in jacobian))
        logged_before = _exact(intervention, "/layer_info/tcp_speed_before")
        logged_after = _exact(intervention, "/layer_info/tcp_speed_after")
        factor = _exact(intervention, "/layer_info/scaling_factor")
        summary_factor = _exact(summary, "/l0c_scaling_factor")
        cap_after = _exact(summary, "/l0c_cap_evidence/tcp_speed_after")
        cap_limit = _exact(summary, "/l0c_cap_evidence/max_tcp_speed_mps")
    except (KeyError, TypeError, ValueError, TelemetryError):
        errors.append(f"L0C intervention lacks exact command-stage evidence at {location}")
        return
    if not math.isclose(logged_before, before, rel_tol=0.0, abs_tol=1e-8):
        errors.append(f"L0C tcp_speed_before is disconnected from decision command at {location}")
    if not math.isclose(logged_after, after, rel_tol=0.0, abs_tol=1e-8):
        errors.append(f"L0C tcp_speed_after is disconnected from decision command at {location}")
    if not math.isclose(logged_after, cap_after, rel_tol=0.0, abs_tol=1e-8):
        errors.append(f"L0C intervention tcp_speed_after is disconnected from cap evidence at {location}")
    if not math.isclose(factor, summary_factor, rel_tol=0.0, abs_tol=1e-8):
        errors.append(f"L0C intervention scaling_factor is disconnected from summary at {location}")
    if before <= 0.0 or factor >= 1.0 or not math.isclose(factor, after / before, rel_tol=0.0, abs_tol=1e-8):
        errors.append(f"L0C intervention scaling_factor is not derived from command speeds at {location}")
    if any(not math.isclose(command[index], proposal[index] * factor, rel_tol=0.0, abs_tol=1e-8) for index in range(5)):
        errors.append(f"L0C intervention scaling_factor is disconnected from command vector at {location}")
    if after > cap_limit + 1e-8:
        errors.append(f"L0C intervention exceeds configured TCP cap at {location}")


def _typed_unavailable(decision: Mapping[str, Any]) -> bool:
    expected = {"availability", "reason"}
    inputs, results = decision.get("monitor_inputs"), decision.get("monitor_results")
    return (
        isinstance(inputs, dict) and isinstance(results, dict)
        and set(inputs) == expected and set(results) == expected
        and inputs.get("availability") == results.get("availability") == "unavailable"
    )


def _layer(records: list[Mapping[str, Any]], layer: str) -> list[Mapping[str, Any]]:
    return [event for event in records if event.get("event_type") == "safety_intervention" and event.get("layer") == layer]


def _integrity(events: list[dict[str, Any]], raw_lines: list[bytes], components: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    previous: str | None = None
    for index, event in enumerate(events):
        errors.extend(f"event[{index}]: {error}" for error in validate_event(event, first=index == 0))
        if previous is not None and event.get("prev_hash") != previous: errors.append(f"event[{index}] hash chain is discontinuous")
        previous = event_hash(event)
    if events and isinstance(components.get("manifest"), dict):
        start = events[0]
        for component, key in (("manifest", "manifest_hash"), ("environment", "environment_hash")):
            value = components.get(component)
            if isinstance(value, dict) and start.get(key) != hashlib.sha256(canonical_json_bytes(value)).hexdigest(): errors.append(f"run_start {component} hash mismatch")
        contract_identity = _contract_identity(components.get("contract"), errors)
        if start.get("contract_hash") != contract_identity:
            errors.append("run_start RFC-0001 contract identity mismatch")
        for event in events:
            if event.get("event_type") in {"episode_start", "episode_summary"} and event.get("contract_hash") != contract_identity:
                errors.append(f"{event.get('event_type')} RFC-0001 contract identity mismatch")
        environment = components.get("environment")
        timestep = environment.get("control_timestep_seconds") if isinstance(environment, dict) else None
        if isinstance(timestep, (int, float)) and not isinstance(timestep, bool):
            for event in events:
                kind = event.get("event_type")
                if kind in {"run_start", "episode_start"} and event.get("timestamp_ns") != 0:
                    errors.append(f"{kind} timestamp must be zero")
                elif kind in {"shift_injected", "decision_evidence", "safety_intervention", "contract_state_change", "kernel_step_summary"}:
                    step = event.get("step")
                    if isinstance(step, int) and event.get("timestamp_ns") != _simulated_timestamp_ns(max(step, 0), float(timestep)):
                        errors.append(f"{kind} timestamp does not match simulated step time")
    _crcs(events, raw_lines, errors)
    if events and events[-1].get("event_type") == "run_end":
        end = events[-1]
        if end.get("actual_event_count") != len(events): errors.append("run_end actual event count mismatch")
        if end.get("actual_episode_count") != sum(event.get("event_type") == "episode_summary" for event in events): errors.append("run_end actual episode count mismatch")
        if len(events) < 2 or end.get("last_event_hash") != event_hash(events[-2]): errors.append("run_end last event hash mismatch")
    return {"events": len(events), "final_event_hash": previous, "continuous": not any("hash" in error for error in errors)}


def _crcs(events: list[dict[str, Any]], lines: list[bytes], errors: list[str]) -> None:
    crcs: dict[str, int] = {}
    for event, line in zip(events, lines):
        episode_id, kind = event.get("episode_id"), event.get("event_type")
        if kind == "episode_start" and isinstance(episode_id, str): crcs[episode_id] = binascii.crc32(line)
        elif isinstance(episode_id, str) and kind == "episode_summary":
            if event.get("episode_crc32") != f"{crcs.get(episode_id, 0) & 0xffffffff:08x}": errors.append(f"episode CRC mismatch for {episode_id}")
        elif isinstance(episode_id, str) and episode_id in crcs: crcs[episode_id] = binascii.crc32(line, crcs[episode_id])


def _h3(
    events: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    flapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    predicates = {"l0r": _h3_l0r(events), "detection_sla": _h3_sla(events, manifest), "transitions": _h3_transitions(events), "flapping": dict(flapping) if flapping is not None else _h3_flapping(events, manifest), "degraded_cap": _h3_cap(events)}
    statuses = {value["status"] for value in predicates.values()}
    return {"overall": "fail" if "fail" in statuses else "unavailable" if "unavailable" in statuses else "pass", "predicates": predicates}


def _h3_l0r(events: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [event for event in events if event.get("event_type") == "kernel_step_summary"]
    if not summaries or any(event.get("l0r_evidence") is None for event in summaries): return {"status": "unavailable", "reason": "l0r evidence unavailable"}
    failures: list[str] = []
    for event in summaries:
        evidence = event["l0r_evidence"]
        try:
            q = _vector(event, evidence, "q"); qdot = _vector(event, evidence, "qdot_exec"); vmax = _vector(event, evidence, "qdot_max_eroded")
            qmin, qmax = _vector(event, evidence, "q_min_eroded"), _vector(event, evidence, "q_max_eroded")
            dt, emax, d_safe = _exact(event, "/l0r_evidence/dt"), _exact(event, "/l0r_evidence/e_max_safe"), _exact(event, "/l0r_evidence/d_safe")
            matrix = [[_exact(event, f"/l0r_evidence/mass_matrix/{i}/{j}") for j in range(len(qdot))] for i in range(len(qdot))]
            distances = {key: _exact(event, f"/l0r_evidence/collision_distances/{_escape(key)}") for key in evidence["collision_distances"]}
        except (KeyError, TelemetryError, TypeError): return {"status": "unavailable", "reason": "malformed exact l0r evidence"}
        where = f"{event['episode_id']}:{event['step']}"; eps = 1e-8
        if any(abs(value) > limit + eps for value, limit in zip(qdot, vmax)): failures.append(f"velocity at {where}")
        if any(next_q < lo - eps or next_q > hi + eps for next_q, lo, hi in zip((q[i] + qdot[i] * dt for i in range(len(q))), qmin, qmax)): failures.append(f"position at {where}")
        if 0.5 * sum(qdot[i] * matrix[i][j] * qdot[j] for i in range(len(qdot)) for j in range(len(qdot))) > emax + eps: failures.append(f"energy at {where}")
        if any(distance < d_safe - eps for distance in distances.values()): failures.append(f"collision at {where}")
    return {"status": "fail" if failures else "pass", "failures": failures}


def _h3_sla(events: list[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    config, shifts = manifest.get("monitor_slas"), [event for event in events if event.get("event_type") == "shift_injected"]
    if not isinstance(config, dict) or not shifts: return {"status": "unavailable", "reason": "no monitor SLA configuration or injected shifts"}
    transitions, decisions, failures, matched = [event for event in events if event.get("event_type") == "contract_state_change"], [event for event in events if event.get("event_type") == "decision_evidence"], [], 0
    coverage_gaps: list[str] = []
    used: set[str] = set()
    for shift in shifts:
        target = config.get(_shift_key(shift)) or config.get(shift.get("shift_type"))
        if not isinstance(target, dict):
            coverage_gaps.append(f"unconfigured injected shift {_shift_key(shift)}")
            continue
        used.add(_shift_key(shift) if _shift_key(shift) in config else str(shift.get("shift_type")))
        matched += 1; maximum, target_state = target.get("max_steps"), target.get("to_state")
        monitor = target.get("monitor")
        if not isinstance(maximum, int) or target_state not in _STATES or not isinstance(monitor, str): failures.append("malformed monitor_slas entry"); continue
        hits = [item for item in transitions if item.get("episode_id") == shift.get("episode_id") and item.get("to_state") == target_state and item.get("trigger_monitor") == monitor and isinstance(item.get("step"), int) and shift["step"] <= item["step"] <= shift["step"] + maximum]
        if not hits:
            failures.append(f"SLA miss for {_shift_key(shift)}")
            continue
        hit = hits[0]
        decision = next((item for item in decisions if item.get("episode_id") == hit.get("episode_id") and item.get("step") == hit.get("step")), None)
        monitor_result = decision.get("monitor_results", {}).get(monitor) if isinstance(decision, dict) and isinstance(decision.get("monitor_results"), dict) else None
        if not isinstance(monitor_result, dict) or monitor_result.get("predicate_violated") is not True or monitor_result.get("triggered") is not True:
            failures.append(f"SLA trigger evidence missing for {_shift_key(shift)}")
    unused = set(config) - used
    if unused:
        coverage_gaps.append(f"declared SLA coverage has no corresponding shift: {sorted(unused)!r}")
    if coverage_gaps:
        return {"status": "unavailable", "reason": "; ".join(coverage_gaps)}
    return {"status": "unavailable", "reason": "no configured injected shift"} if not matched else {"status": "fail" if failures else "pass", "failures": failures}


def _h3_transitions(events: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = [event for event in events if event.get("event_type") == "contract_state_change"]
    if not transitions: return {"status": "unavailable", "reason": "no transition evidence"}
    failures = [f"illegal transition {event.get('from_state')}->{event.get('to_state')}" for event in transitions if (event.get("from_state"), event.get("trigger_severity"), event.get("to_state")) not in _LEGAL]
    return {"status": "fail" if failures else "pass", "failures": failures}


def _h3_flapping(events: list[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    limit, transitions = manifest.get("flap_limit"), [event for event in events if event.get("event_type") == "contract_state_change"]
    if not isinstance(limit, int) or not transitions: return {"status": "unavailable", "reason": "no flap limit or transition evidence"}
    # Closed 0.2.0 manifests intentionally do not carry one global hysteresis
    # value.  The contract replay above is authoritative because a recovery's
    # guard belongs to the monitor that performed that recovery.  This legacy
    # helper remains only for isolated vector tests that explicitly supply a
    # guard; it must never invent one for a real bundle.
    if "flap_hysteresis_ticks" not in manifest:
        return {"status": "unavailable", "reason": "per-monitor flap guards are validated by contract replay"}
    guard = manifest.get("flap_hysteresis_ticks", float("inf"))
    if isinstance(guard, bool) or not isinstance(guard, (int, float)) or guard < 0:
        return {"status": "unavailable", "reason": "invalid flap hysteresis window"}
    failures, flaps = [], Counter()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in transitions: grouped.setdefault(str(event.get("episode_id")), []).append(event)
    for episode, items in grouped.items():
        count, boundary_ok, guard_end = 0, True, None
        for index, item in enumerate(items):
            if item.get("from_state") == "DEGRADED" and item.get("to_state") == "ACTIVE" and isinstance(item.get("step"), int):
                guard_end = item["step"] + guard
                continue
            if item.get("from_state") == "ACTIVE" and item.get("to_state") == "DEGRADED" and isinstance(item.get("step"), int) and guard_end is not None and item["step"] < guard_end:
                count += 1
                if count == limit:
                    following = items[index + 1] if index + 1 < len(items) else None
                    boundary_ok = bool(following and following.get("trigger_monitor") == "flap_limiter" and following.get("from_state") == "DEGRADED" and following.get("to_state") == "RTA_STOP" and following.get("step") == item["step"])
        flaps[episode] = count
        if count >= limit and (count != limit or not boundary_ok):
            failures.append(f"flap limiter must fire exactly at limit for {episode}")
    return {"status": "fail" if failures else "pass", "flaps": dict(flaps), "failures": failures}


def _acceptance_evidence(events: list[dict[str, Any]], manifest: Mapping[str, Any], h3: Mapping[str, Any], errors: list[str]) -> None:
    """Only an explicit acceptance claim requires positive coverage evidence."""
    if manifest.get("bundle_kind") != "acceptance":
        return
    if any(event.get("event_type") == "decision_evidence" and isinstance(event.get("monitor_inputs"), dict) and event["monitor_inputs"].get("availability") == "unavailable" for event in events):
        errors.append("acceptance bundle contains unavailable monitor evidence")
    for summary in (event for event in events if event.get("event_type") == "kernel_step_summary"):
        location = f"{summary.get('episode_id')}:{summary.get('step')}"
        if summary.get("l0r_evidence") is None:
            errors.append(f"acceptance record lacks L0-R evidence at {location}")
        if summary.get("contract_state") == "DEGRADED" and summary.get("l0c_cap_evidence") is None:
            errors.append(f"acceptance DEGRADED record lacks L0-C cap evidence at {location}")
    predicates = h3.get("predicates", {}) if isinstance(h3, Mapping) else {}
    required = ("l0r", "detection_sla", "transitions", "flapping", "degraded_cap")
    for name in required:
        if not isinstance(predicates.get(name), Mapping) or predicates[name].get("status") == "unavailable":
            errors.append(f"acceptance bundle lacks required {name} evidence")


def _h3_cap(events: list[dict[str, Any]]) -> dict[str, Any]:
    degraded = [event for event in events if event.get("event_type") == "kernel_step_summary" and event.get("contract_state") == "DEGRADED"]
    if not degraded: return {"status": "unavailable", "reason": "no DEGRADED evidence"}
    failures, unavailable = [], False
    for event in degraded:
        if event.get("l0c_active") is not True: failures.append("DEGRADED state lacks L0-C cap enforcement"); continue
        if not isinstance(event.get("l0c_cap_evidence"), dict): unavailable = True; continue
        try:
            if _exact(event, "/l0c_cap_evidence/tcp_speed_after") > _exact(event, "/l0c_cap_evidence/max_tcp_speed_mps") + 1e-8: failures.append("DEGRADED TCP cap exceeded")
        except TelemetryError: unavailable = True
    return {"status": "fail" if failures else "unavailable" if unavailable else "pass", "failures": failures}


def _components(root: Path, errors: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for filename, key in (("manifest.json", "manifest"), ("contract.json", "contract"), ("environment.json", "environment"), ("checker.json", "checker"), ("completion_receipt.json", "receipt")):
        path = root / filename
        if path.is_file():
            try: result[key] = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_dupes)
            except (OSError, json.JSONDecodeError, ValueError) as exc: errors.append(f"cannot read {filename}: {exc}")
    return result


def _component_semantics(components: Mapping[str, Any], errors: list[str]) -> None:
    """Verify source identity and the fixed model/L0-R binding before replay."""
    _contract_identity(components.get("contract"), errors)
    environment = components.get("environment")
    expected = {
        "environment", "model", "model_xml", "model_sha256", "control_timestep_seconds",
        "l0r_parameters",
    }
    if not isinstance(environment, dict) or set(environment) != expected:
        errors.append("environment.json must be a closed model/L0-R binding")
        return
    if not isinstance(environment.get("model_sha256"), str) or not _HASH.fullmatch(environment["model_sha256"]):
        errors.append("environment model_sha256 is invalid")
    model_xml = environment.get("model_xml")
    if not isinstance(model_xml, str) or hashlib.sha256(model_xml.encode("utf-8")).hexdigest() != environment.get("model_sha256"):
        errors.append("environment embedded model does not match model_sha256")
    params = environment.get("l0r_parameters")
    required = {
        "n_joints", "dt", "qdot_max_eroded", "q_min_eroded",
        "q_max_eroded", "e_max_safe", "d_safe", "collision_pair_names",
    }
    if not isinstance(params, dict) or set(params) != required:
        errors.append("environment l0r_parameters must be closed")
        return
    if params.get("n_joints") != 5 or not isinstance(params.get("collision_pair_names"), list) or not params["collision_pair_names"] or len(set(params["collision_pair_names"])) != len(params["collision_pair_names"]) or not all(isinstance(name, str) for name in params["collision_pair_names"]):
        errors.append("environment L0-R configured collision set is invalid")
    for key in ("qdot_max_eroded", "q_min_eroded", "q_max_eroded"):
        if not isinstance(params.get(key), list) or len(params[key]) != 5 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in params[key]):
            errors.append(f"environment {key} must be length five")
    for key in ("dt", "e_max_safe", "d_safe"):
        value = params.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"environment {key} must be numeric")
    if environment.get("control_timestep_seconds") != params.get("dt"):
        errors.append("environment control timestep does not match L0-R dt")
    # Track A is frozen by Appendix D and the checked-in MJCF, not by values an
    # evidence bundle happens to declare. This prevents a tampered d_safe or
    # erosion margin from making H3 vacuously pass.
    normative_model = Path(__file__).resolve().parents[3] / "models" / "cell_arm.xml"
    if environment.get("model") != "models/cell_arm.xml" or not normative_model.is_file() or environment.get("model_sha256") != _hash(normative_model):
        errors.append("environment model is not the normative Track A MJCF")
    expected = {
        "dt": 0.02, "e_max_safe": 4.9, "d_safe": 0.02,
        "qdot_max_eroded": [math.pi - 0.01] * 5,
        "q_min_eroded": [-math.pi + 0.001 + 0.01 * 0.02] * 5,
        "q_max_eroded": [math.pi - 0.001 - 0.01 * 0.02] * 5,
        "collision_pair_names": _TRACK_A_COLLISIONS,
    }
    if components.get("manifest", {}).get("bundle_kind") == "acceptance":
        for key, value in expected.items():
            observed = params.get(key)
            if isinstance(value, list) and value and isinstance(value[0], float):
                if not isinstance(observed, list) or len(observed) != len(value) or any(abs(float(left) - right) > 1e-7 for left, right in zip(observed, value)):
                    errors.append(f"environment {key} does not match normative Track A parameters")
            elif observed != value:
                errors.append(f"environment {key} does not match normative Track A parameters")


def _manifest_semantics(manifest: Any, errors: list[str]) -> None:
    """The manifest is evidence, not an extensible diagnostic side channel."""
    fields = {"bundle_kind", "total_episodes", "steps_per_episode", "seed_schedule",
              "flap_limit", "loop_rate_hz", "command_authority", "monitor_slas",
              "completion_timestamp_ns"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        errors.append("manifest.json must be a closed Gate 2 record")
        return
    if manifest.get("bundle_kind") not in {"diagnostic", "acceptance"}:
        errors.append("manifest bundle_kind is invalid")
    if manifest.get("command_authority") != COMMAND_AUTHORITY:
        errors.append("manifest command_authority is invalid")
    for key in ("total_episodes", "steps_per_episode", "flap_limit", "completion_timestamp_ns"):
        if isinstance(manifest.get(key), bool) or not isinstance(manifest.get(key), int) or manifest[key] < 0:
            errors.append(f"manifest {key} must be a non-negative integer")
    if manifest.get("flap_limit", 0) < 1:
        errors.append("manifest flap_limit must be positive")
    rate = manifest.get("loop_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        errors.append("manifest loop_rate_hz must be positive")
    seeds = manifest.get("seed_schedule")
    if not isinstance(seeds, list) or len(seeds) != manifest.get("total_episodes") or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        errors.append("manifest seed_schedule must exactly bind every episode")
    slas = manifest.get("monitor_slas")
    if not isinstance(slas, dict):
        errors.append("manifest monitor_slas must be a closed object")
    elif any(not isinstance(name, str) or not name or not isinstance(value, dict) or set(value) != {"monitor", "max_steps", "to_state"} or not isinstance(value.get("monitor"), str) or not value["monitor"] or isinstance(value.get("max_steps"), bool) or not isinstance(value.get("max_steps"), int) or value["max_steps"] < 0 or value.get("to_state") not in _STATES for name, value in slas.items()):
        errors.append("manifest monitor_slas entries must be closed typed records")


def _manifest_environment_binding(manifest: Any, environment: Any, errors: list[str]) -> None:
    """Bind manifest time semantics to the embedded deterministic environment."""
    if not isinstance(manifest, dict) or not isinstance(environment, dict):
        return
    dt, rate = environment.get("control_timestep_seconds"), manifest.get("loop_rate_hz")
    if (isinstance(dt, bool) or not isinstance(dt, (int, float)) or dt <= 0
            or isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isclose(float(rate), 1.0 / float(dt), rel_tol=0.0, abs_tol=1e-12)):
        errors.append("manifest loop_rate_hz must equal inverse environment control timestep")


def _contract_identity(component: Any, errors: list[str]) -> str | None:
    if not isinstance(component, dict) or set(component) != {"contract_envelope_version", "original_yaml"}:
        errors.append("contract.json must be a full RFC-0001 source envelope")
        return None
    if component.get("contract_envelope_version") != "ffsk.contract-envelope.v1" or not isinstance(component.get("original_yaml"), str):
        errors.append("contract.json envelope is malformed")
        return None
    try:
        return load_contract_text(component["original_yaml"]).body_sha256
    except Exception as exc:  # Contract loader supplies the authority boundary.
        errors.append(f"contract.json RFC-0001 verification failed: {exc}")
        return None


def _contract_evidence_binding(
    events: list[dict[str, Any]], component: Any, errors: list[str]
) -> None:
    """Bind monitor and shift evidence to the verified contract vocabulary."""
    if not isinstance(component, dict) or not isinstance(component.get("original_yaml"), str):
        return
    try:
        raw = load_contract_text(component["original_yaml"]).raw_contract
        monitor_specs = {
            item["name"]: item["severity"] for item in raw["assumption_monitors"]
        }
        anchors = {item["name"] for item in raw["anchors"]}
        signals = {
            item["params"]["signal"]
            for item in raw["assumption_monitors"]
            if item["type"] == "safety_io_health"
        }
        degraded_cap = raw["revocation_policy"]["on_degrade"]["caps"]["max_tcp_speed_mps"]
    except Exception:
        return
    result_fields = {
        "triggered", "should_trigger", "should_clear", "severity",
        "predicate_violated",
    }
    input_fields = {
        "anchor_poses", "safety_io", "sensor_health", "base_pose_estimate",
    }
    reset_keys = {
        (item.get("episode_id"), item.get("step"))
        for item in events
        if item.get("event_type") == "contract_state_change"
        and item.get("trigger_severity") == "reset"
    }
    decisions = {
        (item.get("episode_id"), item.get("step")): item
        for item in events if item.get("event_type") == "decision_evidence"
    }
    for event in events:
        kind = event.get("event_type")
        if kind == "decision_evidence":
            results = event.get("monitor_results")
            inputs = event.get("monitor_inputs")
            if _typed_unavailable(event):
                # The only normative unavailable representation is the closed
                # typed denial accepted by both producer and independent wire
                # validator.  It carries no invented monitor values.
                if event.get("authorized") is not False or event.get("authority_state_post") != "RTA_STOP":
                    errors.append("typed unavailable monitor evidence is not fail-closed")
            else:
                if not isinstance(results, dict) or set(results) != set(monitor_specs):
                    errors.append("decision monitor result set does not match contract")
                else:
                    for name, result in results.items():
                        if not isinstance(result, dict) or set(result) != result_fields:
                            errors.append(f"monitor result fields are not closed for {name}")
                        elif result.get("severity") != monitor_specs[name] or any(
                            not isinstance(result.get(key), bool)
                            for key in ("triggered", "should_trigger", "should_clear", "predicate_violated")
                        ):
                            errors.append(f"monitor result semantics do not match contract for {name}")
                if not isinstance(inputs, dict) or set(inputs) != input_fields:
                    errors.append("decision monitor input fields do not match contract")
                else:
                    anchor_poses = inputs.get("anchor_poses")
                    safety_io = inputs.get("safety_io")
                    if not isinstance(anchor_poses, dict) or set(anchor_poses) != anchors:
                        errors.append("anchor monitor input set does not match contract")
                    if not isinstance(safety_io, dict) or set(safety_io) != signals or any(not isinstance(value, bool) for value in safety_io.values()):
                        errors.append("safety I/O input set does not match contract")
                    # Immediate safety-I/O predicates are independently replayable:
                    # an ACTIVE/authorized result cannot contradict an unsafe E-stop.
                    if isinstance(safety_io, dict):
                        for spec in raw["assumption_monitors"]:
                            if spec["type"] != "safety_io_health":
                                continue
                            name, params = spec["name"], spec["params"]
                            expected = safety_io.get(params["signal"]) != params["expected"]
                            result = results.get(name) if isinstance(results, dict) else None
                            if not isinstance(result, dict) or result.get("predicate_violated") is not expected:
                                errors.append(f"safety I/O predicate contradicts input for {name}")
                            if expected and params.get("expected") is True and params["signal"] == "estop_ok":
                                if event.get("authority_state_post") == "ACTIVE" or event.get("authorized") is True:
                                    errors.append("unsafe E-stop input contradicts ACTIVE/nonzero authority")
            reset_present = (event.get("episode_id"), event.get("step")) in reset_keys
            # A reset request may correctly be denied.  A reset transition,
            # however, can only occur on the same fully logged request tick.
            if reset_present and event.get("reset_input") is not True:
                errors.append("reset transition lacks its decision reset request")
        elif kind == "shift_injected":
            shift_type, params = event.get("shift_type"), event.get("shift_params")
            if shift_type == "anchor_drift" and isinstance(params, dict) and params.get("anchor") not in anchors:
                errors.append("anchor shift target does not exist in contract")
            if shift_type == "safety_io_flip" and isinstance(params, dict) and params.get("signal") not in signals:
                errors.append("safety I/O shift target does not exist in contract")
            expected_channels = {
                "anchor_drift": {"anchor_poses"},
                "safety_io_flip": {"safety_io"},
                "sensor_stale": {"sensor_health"},
                "base_drift": {"base_pose_estimate"},
            }.get(shift_type)
            if expected_channels is None or event.get("affected_channels") != sorted(expected_channels):
                errors.append("shift affected_channels do not match normative shift channels")
        elif kind == "contract_state_change":
            trigger = event.get("trigger_monitor")
            if trigger not in set(monitor_specs) | {"l0r_nearmiss", "operator_reset", "flap_limiter", "monitor_evidence_missing"}:
                errors.append("transition trigger is not contract-bound")
        elif kind == "kernel_step_summary" and event.get("contract_state") == "DEGRADED":
            cap = event.get("l0c_cap_evidence")
            decision = decisions.get((event.get("episode_id"), event.get("step")))
            if not isinstance(cap, dict) or not isinstance(decision, dict):
                continue
            try:
                contract_cap = float(degraded_cap)
                recorded_cap = _exact(event, "/l0c_cap_evidence/max_tcp_speed_mps")
                if recorded_cap != contract_cap:
                    errors.append("L0-C cap does not match verified RFC-0001 contract")
                command = [_exact(decision, f"/l0c_command/{i}") for i in range(5)]
                jacobian = [[_exact(event, f"/l0c_cap_evidence/tcp_jacobian/{row}/{col}") for col in range(5)] for row in range(3)]
                derived_speed = math.sqrt(sum(sum(jacobian[row][col] * command[col] for col in range(5)) ** 2 for row in range(3)))
                if not math.isclose(_exact(event, "/l0c_cap_evidence/tcp_speed_after"), derived_speed, rel_tol=0.0, abs_tol=1e-8):
                    errors.append("L0-C TCP speed is not derived from command-stage Jacobian evidence")
            except (KeyError, TypeError, ValueError, TelemetryError):
                errors.append("L0-C cap evidence lacks independent command-stage derivation")


def _contract_temporal_replay(events: list[dict[str, Any]], component: Any,
                              manifest: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    """Replay L0-C from the signed RFC-0001 source, never producer monitors.

    This deliberately duplicates the small, deterministic monitor/state rules.
    It consumes only decision inputs and checked contract data, so matching a
    producer implementation cannot make contradictory evidence acceptable.
    """
    if not isinstance(component, dict) or not isinstance(component.get("original_yaml"), str):
        return {"status": "unavailable", "reason": "verified contract unavailable"}
    try:
        raw = load_contract_text(component["original_yaml"]).raw_contract
        rate = manifest.get("loop_rate_hz", 50)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
            errors.append("manifest loop_rate_hz is invalid for contract replay")
            return {"status": "unavailable", "reason": "invalid loop rate"}
        specs = {item["name"]: item for item in raw["assumption_monitors"]}
        anchors = {item["name"]: item for item in raw["anchors"]}
        base = raw["frames"]["robot_base_in_cell"]
    except Exception as exc:
        errors.append(f"cannot initialize independent contract replay: {exc}")
        return {"status": "unavailable", "reason": "contract replay initialization failed"}
    changes: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event_type") == "contract_state_change":
            changes.setdefault((str(event.get("episode_id")), event.get("step")), []).append(event)
    state = "ACTIVE"
    runtime: dict[str, dict[str, Any]] = {}
    guard_end: int | None = None
    flaps = 0
    flap_counts: dict[str, int] = {}
    flap_failures: list[str] = []
    for event in events:
        if event.get("event_type") == "episode_start":
            if runtime:
                flap_counts[episode] = flaps
            state, runtime, guard_end, flaps = "ACTIVE", _monitor_runtime(specs, rate), None, 0
            episode = str(event.get("episode_id"))
            continue
        if event.get("event_type") != "decision_evidence":
            continue
        episode, step = str(event.get("episode_id")), event.get("step")
        if not isinstance(step, int):
            continue
        emitted = changes.get((episode, step), [])
        if _typed_unavailable(event):
            expected = [(state, "RTA_STOP", "monitor_evidence_missing", "rta_stop")] if state != "RTA_STOP" else []
            _expect_transitions(emitted, expected, event, errors)
            for change in emitted:
                if change.get("monitors_status") != {"monitor_evidence_missing": {"triggered": True, "violated": True}}:
                    errors.append(f"unavailable transition monitor snapshot is not closed at {episode}:{step}")
            state = "RTA_STOP" if state != "RTA_STOP" else state
            continue
        inputs, reported = event.get("monitor_inputs"), event.get("monitor_results")
        if not isinstance(inputs, dict) or not isinstance(reported, dict):
            continue  # wire/binding errors provide the primary diagnostic.
        computed: dict[str, dict[str, Any]] = {}
        for name in sorted(specs):
            violated = _monitor_predicate(name, specs[name], inputs, event, anchors, base, runtime[name])
            row = runtime[name]
            row["fail"], row["ok"] = (row["fail"] + 1, 0) if violated else (0, row["ok"] + 1)
            trigger = not row["triggered"] and row["fail"] > row["debounce"]
            if trigger:
                row["triggered"] = True
            clear = row["triggered"] and row["ok"] > row["hysteresis"]
            if clear:
                row["triggered"] = False
            computed[name] = {"predicate_violated": violated, "triggered": row["triggered"],
                              "should_trigger": trigger, "should_clear": clear,
                              "severity": specs[name]["severity"]}
            if reported.get(name) != computed[name]:
                errors.append(f"independent monitor replay contradicts {name} at {episode}:{step}")
        expected_snapshot = {
            name: {"triggered": value["triggered"], "violated": value["predicate_violated"]}
            for name, value in computed.items()
        }
        for change in emitted:
            if change.get("monitors_status") != expected_snapshot:
                errors.append(f"transition monitor snapshot contradicts replay at {episode}:{step}")
        expected: list[tuple[str, str, str, str]] = []
        candidates = [name for name, value in computed.items() if value["should_trigger"]]
        selected = min(candidates, key=lambda name: ({"rta_stop": 0, "revoke": 1, "degrade": 2}.get(computed[name]["severity"], 99), name), default=None)
        if selected is not None:
            severity = computed[selected]["severity"]
            target = {"rta_stop": "RTA_STOP", "revoke": "REVOKED", "degrade": "DEGRADED"}.get(severity)
            legal = {("ACTIVE", "RTA_STOP"), ("ACTIVE", "REVOKED"), ("ACTIVE", "DEGRADED"),
                     ("DEGRADED", "RTA_STOP"), ("DEGRADED", "REVOKED"), ("REVOKED", "RTA_STOP")}
            if target and (state, target) in legal:
                expected.append((state, target, selected, severity)); state = target
                if target == "DEGRADED" and guard_end is not None and step < guard_end:
                    flaps += 1
                    if flaps == manifest.get("flap_limit", 4):
                        expected.append(("DEGRADED", "RTA_STOP", "flap_limiter", "rta_stop")); state = "RTA_STOP"
                    elif flaps > manifest.get("flap_limit", 4):
                        flap_failures.append(f"flap limit exceeded for {episode}")
        clears = [name for name, value in computed.items() if value["should_clear"] and value["severity"] == "degrade"]
        if state == "DEGRADED" and clears and all(not value["triggered"] for value in computed.values() if value["severity"] == "degrade"):
            name = min(clears)
            expected.append(("DEGRADED", "ACTIVE", name, "clear")); state = "ACTIVE"
            guard_end = step + runtime[name]["hysteresis"]
        if event.get("reset_input") is True and state in {"REVOKED", "RTA_STOP"} and all(not item["predicate_violated"] and not item["triggered"] for item in computed.values()):
            expected.append((state, "ACTIVE", "operator_reset", "reset")); state = "ACTIVE"
        if event.get("rta_nearmiss") is True and state != "RTA_STOP":
            expected.append((state, "RTA_STOP", "l0r_nearmiss", "rta_stop")); state = "RTA_STOP"
        if any(item[2] == "flap_limiter" for item in expected):
            observed = [
                (item.get("from_state"), item.get("to_state"), item.get("trigger_monitor"), item.get("trigger_severity"))
                for item in emitted
            ]
            if observed != expected:
                flap_failures.append(f"flap limiter evidence mismatch for {episode}:{step}")
        _expect_transitions(emitted, expected, event, errors)
    if runtime:
        flap_counts[episode] = flaps
    return {
        "status": "fail" if flap_failures else "pass",
        "flaps": flap_counts,
        "failures": flap_failures,
    }


def _monitor_runtime(specs: Mapping[str, Mapping[str, Any]], rate: float) -> dict[str, dict[str, Any]]:
    return {name: {"fail": 0, "ok": 0, "triggered": False,
                   "debounce": math.ceil(float(spec["debounce_ms"]) * rate / 1000.0),
                   "hysteresis": math.ceil(float(spec["hysteresis_ms"]) * rate / 1000.0),
                   "timestamp": None, "pos": [], "rot": []} for name, spec in specs.items()}


def _monitor_predicate(name: str, spec: Mapping[str, Any], inputs: Mapping[str, Any], event: Mapping[str, Any],
                       anchors: Mapping[str, Mapping[str, Any]], base: Mapping[str, Any], row: dict[str, Any]) -> bool:
    kind, params = spec["type"], spec["params"]
    if kind == "safety_io_health":
        return inputs.get("safety_io", {}).get(params["signal"]) != params["expected"]
    if kind == "anchor_pose_consistency":
        for anchor_name in params.get("anchors", []):
            anchor, observed = anchors.get(anchor_name), inputs.get("anchor_poses", {}).get(anchor_name)
            if not anchor or anchor.get("required") and not observed:
                return True
            if anchor and anchor.get("required"):
                if _distance(event, observed["xyz"], anchor["expected_pose_in_cell"]["xyz"], f"/monitor_inputs/anchor_poses/{_escape(anchor_name)}/xyz") > float(anchor["tolerance"]["pos_m"]): return True
                if _distance(event, observed["rpy"], anchor["expected_pose_in_cell"]["rpy"], f"/monitor_inputs/anchor_poses/{_escape(anchor_name)}/rpy") > float(anchor["tolerance"]["rot_rad"]): return True
        return False
    if kind == "sensor_health":
        sensor = inputs.get("sensor_health", {})
        age = _number(event, "/monitor_inputs/sensor_health/frame_age_ms")
        stamp = _number(event, "/monitor_inputs/sensor_health/timestamp")
        violated = age > float(params["max_frame_age_ms"])
        if params.get("monotonic_timestamps", True) and row["timestamp"] is not None and stamp <= row["timestamp"]:
            violated = True
        # Match the native monitor: rejected stale/nonmonotonic samples do not
        # replace the last known-good timestamp.
        if not violated:
            row["timestamp"] = stamp
        return violated
    if kind == "robot_mount_integrity":
        pose = inputs.get("base_pose_estimate", {})
        pos = _distance(event, pose["xyz"], base["xyz"], "/monitor_inputs/base_pose_estimate/xyz")
        rot = _distance(event, pose["rpy"], base["rpy"], "/monitor_inputs/base_pose_estimate/rpy")
        row["pos"].append(pos); row["rot"].append(rot); del row["pos"][:-50]; del row["rot"][:-50]
        return max(row["pos"]) > float(params["base_pose_drift_pos_m"]) or max(row["rot"]) > float(params["base_pose_drift_rot_rad"])
    return True


def _number(event: Mapping[str, Any], pointer: str) -> float:
    return _exact(event, pointer)


def _distance(event: Mapping[str, Any], observed: list[Any], expected: list[Any], pointer: str) -> float:
    return math.sqrt(sum((_number(event, f"{pointer}/{index}") - float(expected[index])) ** 2 for index in range(3)))


def _expect_transitions(emitted: list[dict[str, Any]], expected: list[tuple[str, str, str, str]], event: Mapping[str, Any], errors: list[str]) -> None:
    observed = [(item.get("from_state"), item.get("to_state"), item.get("trigger_monitor"), item.get("trigger_severity")) for item in emitted]
    if observed != expected:
        errors.append(f"independent state replay contradicts emitted transitions at {event.get('episode_id')}:{event.get('step')}")
def _l0r_binding(events: list[dict[str, Any]], environment: Any, errors: list[str]) -> None:
    if not isinstance(environment, dict) or not isinstance(environment.get("l0r_parameters"), dict):
        return
    params = environment["l0r_parameters"]
    names = params.get("collision_pair_names")
    for event in events:
        if event.get("event_type") != "kernel_step_summary":
            continue
        evidence = event.get("l0r_evidence")
        if evidence is None:
            continue
        if not isinstance(evidence, dict):
            errors.append("malformed per-step L0-R evidence")
            continue
        if list(evidence.get("collision_distances", {})) != names:
            errors.append("per-step L0-R collision pair set does not match environment")
        for key in ("dt", "e_max_safe", "d_safe"):
            try:
                if _exact(event, f"/l0r_evidence/{key}") != float(params[key]):
                    errors.append(f"per-step L0-R {key} does not match environment")
            except (KeyError, TypeError, ValueError, TelemetryError):
                errors.append(f"per-step L0-R {key} evidence is unavailable")
        for key in ("qdot_max_eroded", "q_min_eroded", "q_max_eroded"):
            try:
                observed = [_exact(event, f"/l0r_evidence/{key}/{index}") for index in range(5)]
                if observed != [float(value) for value in params[key]]:
                    errors.append(f"per-step L0-R {key} does not match environment")
            except (KeyError, TypeError, ValueError, TelemetryError):
                errors.append(f"per-step L0-R {key} evidence is unavailable")


def _events(path: Path, errors: list[str]) -> tuple[list[dict[str, Any]], list[bytes]]:
    if not path.is_file(): return [], []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"): errors.append("NDJSON is truncated: missing final newline")
    events, lines = [], raw.splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        raw_line = line[:-1] if line.endswith(b"\n") else line
        try: event = json.loads(raw_line.decode("utf-8"), object_pairs_hook=_no_dupes)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc: errors.append(f"invalid NDJSON at line {number}: {exc}"); continue
        if not isinstance(event, dict): errors.append(f"NDJSON line {number} is not an object"); continue
        if not raw_line.startswith(b'{"event_type"'): errors.append(f"event_type is not first on disk at line {number}")
        if event_bytes(event) != raw_line: errors.append(f"non-canonical event bytes at line {number}")
        events.append(event)
    return events, lines


def _receipt_errors(root: Path, receipt: Any, events: list[dict[str, Any]]) -> list[str]:
    if not isinstance(receipt, dict): return ["completion receipt must be an object"]
    expected = {"schema_version": SCHEMA_VERSION, "command_authority": COMMAND_AUTHORITY, "events_sha256": _hash(root / "events.ndjson"), "manifest_sha256": _hash(root / "manifest.json"), "contract_sha256": _hash(root / "contract.json"), "environment_sha256": _hash(root / "environment.json"), "checker_sha256": _hash(root / "checker.json"), "final_event_hash": event_hash(events[-1]) if events else None}
    errors = []
    if set(receipt) != set(expected):
        errors.append("completion receipt fields do not exactly match schema")
    errors.extend(f"completion receipt mismatch for {key}" for key, value in expected.items() if receipt.get(key) != value)
    return errors


def _write_finalization(root_value: str | Path, checker_output: Mapping[str, Any]) -> None:
    """Independently write checker and non-circular receipt artifacts once."""
    root = Path(root_value)
    checker_path = root / "checker.json"
    receipt_path = root / "completion_receipt.json"
    if checker_path.exists() or receipt_path.exists():
        raise BundleImmutableError("checker or completion receipt already exists")
    parse_errors: list[str] = []
    events, _ = _events(root / "events.ndjson", parse_errors)
    if parse_errors or not events:
        raise TelemetryError("cannot finalize invalid producer events")
    with checker_path.open("xb") as handle:
        handle.write(canonical_json_bytes(dict(checker_output)))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "command_authority": COMMAND_AUTHORITY,
        "events_sha256": _hash(root / "events.ndjson"),
        "manifest_sha256": _hash(root / "manifest.json"),
        "contract_sha256": _hash(root / "contract.json"),
        "environment_sha256": _hash(root / "environment.json"),
        "checker_sha256": _hash(checker_path),
        "final_event_hash": event_hash(events[-1]),
    }
    with receipt_path.open("xb") as handle:
        handle.write(canonical_json_bytes(receipt))


def _exact(event: Mapping[str, Any], pointer: str) -> float:
    try:
        return independent_exact(event, pointer)
    except ValueError as exc:
        raise TelemetryError(str(exc)) from exc
def _vector(event: Mapping[str, Any], evidence: Mapping[str, Any], key: str) -> list[float]: return [_exact(event, f"/l0r_evidence/{key}/{index}") for index in range(len(evidence[key]))]
def _same_vector(left: Mapping[str, Any], left_pointer: str, right: Mapping[str, Any], right_pointer: str) -> bool:
    left_values = _pointer_list(left, left_pointer); right_values = _pointer_list(right, right_pointer)
    return len(left_values) == len(right_values) and all(_exact(left, f"{left_pointer}/{index}") == _exact(right, f"{right_pointer}/{index}") for index in range(len(left_values)))
def _same_tree(left: Mapping[str, Any], left_pointer: str, right: Mapping[str, Any], right_pointer: str) -> bool:
    def value(event: Mapping[str, Any], pointer: str) -> Any:
        current: Any = event
        for part in pointer.lstrip("/").split("/"):
            current = current[part.replace("~1", "/").replace("~0", "~")]
        return current
    def walk(a: Any, ap: str, b: Any, bp: str) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(walk(a[key], f"{ap}/{_escape(key)}", b[key], f"{bp}/{_escape(key)}") for key in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(walk(item, f"{ap}/{i}", b[i], f"{bp}/{i}") for i, item in enumerate(a))
        if isinstance(a, str) and isinstance(b, str):
            try: return _exact(left, ap) == _exact(right, bp)
            except TelemetryError: return a == b
        return a == b
    try: return walk(value(left, left_pointer), left_pointer, value(right, right_pointer), right_pointer)
    except (KeyError, TypeError): return False
def _pointer_list(event: Mapping[str, Any], pointer: str) -> list[Any]:
    value: Any = event
    for part in pointer.lstrip("/").split("/"):
        value = value[part]
    if not isinstance(value, list): raise TelemetryError(f"expected vector at {pointer}")
    return value
def _escape(value: str) -> str: return value.replace("~", "~0").replace("/", "~1")
def _shift_key(event: Mapping[str, Any]) -> str:
    params = event.get("shift_params"); signal = params.get("signal") if isinstance(params, dict) else None
    return f"{event.get('shift_type')}:{signal}" if isinstance(signal, str) else str(event.get("shift_type"))
def _hash(path: Path) -> str | None: return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def _simulated_timestamp_ns(step: int, timestep: float) -> int:
    # MuJoCo advances its binary64 clock by repeated addition. Multiplication
    # would erase the deterministic one-nanosecond truncation visible at some
    # ticks and would reject faithful simulator evidence.
    value = 0.0
    for _ in range(step):
        value += timestep
    return int(value * 1_000_000_000)
def _result(ok: bool, errors: list[str], integrity: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    return {"ok": ok, "evidence_ok": not errors, "errors": errors, "integrity": integrity, **details}
def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result
