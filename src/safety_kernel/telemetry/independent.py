"""Independent Gate 2 wire validator and canonical/hash implementation.

This deliberately has no dependency on producer telemetry canonicalization or
schema modules.  It is intentionally small, explicit, and fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

ZERO_HASH = "0" * 64
FLOAT_HEX_FIELD = "float_hex"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_WIRE = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}")
_STATES = {"ACTIVE", "DEGRADED", "REVOKED", "RTA_STOP"}
_COMMON = {"event_type", "prev_hash", FLOAT_HEX_FIELD}
_FIELDS = {
    "run_start": _COMMON | {"timestamp_ns", "manifest_hash", "contract_hash", "environment_hash", "total_episodes", "schema_version", "command_authority"},
    "episode_start": _COMMON | {"timestamp_ns", "episode_id", "seed", "contract_hash", "planned_steps"},
    "shift_injected": _COMMON | {"timestamp_ns", "episode_id", "step", "shift_type", "shift_params", "affected_channels"},
    "decision_evidence": _COMMON | {"timestamp_ns", "episode_id", "step", "proposal", "l0c_command", "l0r_command", "final_command", "truth", "observed", "believed", "monitor_inputs", "monitor_results", "authority_state_pre", "authority_state_post", "authorized", "authority_reason", "reset_input", "rta_nearmiss"},
    "safety_intervention": _COMMON | {"timestamp_ns", "episode_id", "step", "layer", "qdot_cmd", "qdot_exec", "constraints_active", "margin_to_L0R", "energy_ratio", "energy_scaling_factor", "collision_pairs", "rta_nearmiss", "layer_info", "truth", "observed", "believed"},
    "contract_state_change": _COMMON | {"timestamp_ns", "episode_id", "step", "from_state", "to_state", "trigger_monitor", "trigger_severity", "monitors_status"},
    "kernel_step_summary": _COMMON | {"timestamp_ns", "episode_id", "step", "contract_state", "l0r_intervened", "l0c_active", "l0c_intervened", "l0c_scaling_factor", "l0c_cap_evidence", "energy_ratio", "min_margin_to_L0R", "rta_nearmiss", "min_collision_distance", "min_keepout_distance", "l0r_evidence"},
    "episode_summary": _COMMON | {"episode_id", "seed", "contract_hash", "total_steps", "total_interventions", "state_transitions", "state_transition_count", "final_state", "synthetic_revocation", "synthetic_revocation_params", "l1_arm", "full_head_staleness_episodes", "min_margin_to_L0R", "max_energy_ratio", "fn_count", "fp_count", "episode_crc32"},
    "run_end": _COMMON | {"timestamp_ns", "status", "command_authority", "actual_event_count", "actual_episode_count", "last_event_hash"},
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def event_bytes(event: Mapping[str, Any]) -> bytes:
    if "event_type" not in event:
        raise ValueError("event_type is required")
    ordered = {"event_type": _sort_nested(event["event_type"])}
    ordered.update((key, _sort_nested(event[key])) for key in sorted(event) if key != "event_type")
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def event_hash(event: Mapping[str, Any]) -> str:
    return hashlib.sha256(event_bytes(event)).hexdigest()


def exact(event: Mapping[str, Any], pointer: str) -> float:
    evidence = event.get(FLOAT_HEX_FIELD)
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get(pointer), str):
        raise ValueError(f"missing exact evidence at {pointer}")
    value = float.fromhex(evidence[pointer])
    if not math.isfinite(value) or value.hex() != evidence[pointer]:
        raise ValueError(f"invalid exact evidence at {pointer}")
    return value


def validate_event(event: Mapping[str, Any], *, first: bool = False) -> list[str]:
    errors: list[str] = []
    kind = event.get("event_type")
    if kind not in _FIELDS:
        return [f"unknown event_type: {kind!r}"]
    if set(event) != _FIELDS[kind]:
        errors.append(f"{kind} fields must exactly match schema")
    if not isinstance(event.get("prev_hash"), str) or not _HASH.fullmatch(event["prev_hash"]):
        errors.append("prev_hash must be lowercase SHA256 hex")
    if first and (kind != "run_start" or event.get("prev_hash") != ZERO_HASH):
        errors.append("first event must be run_start with zero prev_hash")
    if kind == "run_start" and (event.get("schema_version") != "0.2.0" or event.get("command_authority") != "software_command_suppression_only"):
        errors.append("run_start schema or authority is invalid")
    if kind == "run_end" and (event.get("status") != "completed" or event.get("command_authority") != "software_command_suppression_only"):
        errors.append("run_end status or authority is invalid")
    _generic(event, errors)
    if kind == "decision_evidence":
        _decision(event, errors)
    elif kind == "safety_intervention":
        _intervention(event, errors)
    elif kind == "kernel_step_summary":
        _summary(event, errors)
    elif kind == "shift_injected":
        _shift(event, errors)
    elif kind == "contract_state_change":
        status = event.get("monitors_status")
        if event.get("from_state") not in _STATES or event.get("to_state") not in _STATES or not isinstance(status, dict) or not status or any(not isinstance(name, str) or not isinstance(value, dict) or set(value) != {"triggered", "violated"} or not isinstance(value.get("triggered"), bool) or not isinstance(value.get("violated"), bool) for name, value in status.items()):
            errors.append("malformed contract state change")
    elif kind == "episode_summary":
        _episode(event, errors)
    _float_evidence(event, errors)
    return errors


def _decision(event: Mapping[str, Any], errors: list[str]) -> None:
    for key in ("proposal", "l0c_command", "l0r_command", "final_command"):
        _vector(event.get(key), key, errors)
        _require_exact_values(event, f"/{key}", 5, errors)
    _channels(event, errors)
    if event.get("authority_state_pre") not in _STATES or event.get("authority_state_post") not in _STATES:
        errors.append("invalid decision authority state")
    if not isinstance(event.get("authorized"), bool) or event["authorized"] != (event.get("authority_state_post") not in {"REVOKED", "RTA_STOP"}):
        errors.append("decision authorization contradicts post-state authority")
    if event.get("authorized") is False:
        for index in range(5):
            try:
                if exact(event, f"/final_command/{index}").hex() != (0.0).hex():
                    errors.append("unauthorized final command authoritative values must be positive zero")
                    break
            except (ValueError, TypeError):
                errors.append("unauthorized final command lacks exact zero evidence")
                break
    if event.get("reset_input") is not None and not isinstance(event.get("reset_input"), bool):
        errors.append("reset_input must be boolean or null")
    if type(event.get("rta_nearmiss")) is not bool:
        errors.append("decision rta_nearmiss must be boolean")
    _monitor_evidence(event, errors)


def _intervention(event: Mapping[str, Any], errors: list[str]) -> None:
    layer = event.get("layer")
    # L1 is reserved-and-rejected by the frozen 0.2.0 ABI.
    if layer not in {"L0C", "L0R", "RTA"}:
        errors.append("invalid intervention layer")
    _vector(event.get("qdot_cmd"), "qdot_cmd", errors)
    _vector(event.get("qdot_exec"), "qdot_exec", errors)
    _channels(event, errors)
    if type(event.get("rta_nearmiss")) is not bool:
        errors.append("intervention rta_nearmiss must be boolean")
    pairs = event.get("collision_pairs")
    if not isinstance(pairs, list) or any(not isinstance(pair, dict) or set(pair) != {"pair_id", "distance"} or not isinstance(pair.get("pair_id"), str) for pair in pairs):
        errors.append("collision_pairs must be closed pair evidence")
    else:
        for index in range(len(pairs)):
            _require_exact_pointer(event, f"/collision_pairs/{index}/distance", errors)
    for key in ("energy_ratio", "energy_scaling_factor"):
        _require_exact_pointer(event, f"/{key}", errors)
    constraints, margins, info = event.get("constraints_active"), event.get("margin_to_L0R"), event.get("layer_info")
    if not isinstance(constraints, list) or any(not isinstance(value, str) or not value for value in constraints):
        errors.append("intervention constraints must be a typed string list")
    if layer == "L0R":
        if not constraints or not isinstance(margins, dict) or not margins or not all(isinstance(name, str) and name for name in margins): errors.append("L0R intervention requires closed constraints and margins")
        if info != {}: errors.append("L0R layer_info must be empty")
        if isinstance(margins, dict):
            for name in margins: _require_exact_pointer(event, f"/margin_to_L0R/{_escape(name)}", errors)
    elif layer == "L0C":
        if margins != {} or not isinstance(info, dict) or set(info) != {"tcp_speed_before", "tcp_speed_after", "scaling_factor"}: errors.append("L0C layer_info must be closed")
        elif any(_require_exact_pointer(event, f"/layer_info/{key}", errors) for key in info): pass
    elif layer == "RTA":
        if margins != {} or not isinstance(info, dict) or set(info) != {"authority_reason"} or not isinstance(info.get("authority_reason"), str) or not info["authority_reason"]: errors.append("RTA layer_info must be a closed authority reason")


def _summary(event: Mapping[str, Any], errors: list[str]) -> None:
    if event.get("contract_state") not in _STATES:
        errors.append("invalid summary contract state")
    if any(type(event.get(key)) is not bool for key in ("l0r_intervened", "l0c_active", "l0c_intervened", "rta_nearmiss")):
        errors.append("summary booleans must be actual booleans")
    for key in (
        "l0c_scaling_factor", "energy_ratio", "min_margin_to_L0R",
        "min_collision_distance", "min_keepout_distance",
    ):
        _require_exact_pointer(event, f"/{key}", errors)
    cap = event.get("l0c_cap_evidence")
    if cap is not None:
        if not isinstance(cap, dict) or set(cap) != {"tcp_speed_after", "max_tcp_speed_mps", "tcp_jacobian"}:
            errors.append("l0c_cap_evidence fields must exactly match schema")
        else:
            for key in ("tcp_speed_after", "max_tcp_speed_mps"):
                _require_exact_pointer(event, f"/l0c_cap_evidence/{key}", errors)
            jacobian = cap.get("tcp_jacobian")
            if not isinstance(jacobian, list) or len(jacobian) != 3 or any(not isinstance(row, list) or len(row) != 5 for row in jacobian):
                errors.append("l0c cap Jacobian must be 3x5")
            else:
                for row in range(3): _require_exact_values(event, f"/l0c_cap_evidence/tcp_jacobian/{row}", 5, errors)
    evidence = event.get("l0r_evidence")
    if evidence is None:
        return
    required = {"q", "qdot_exec", "qdot_max_eroded", "q_min_eroded", "q_max_eroded", "dt", "mass_matrix", "e_max_safe", "d_safe", "collision_distances"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        errors.append("l0r evidence fields must exactly match schema")
        return
    for key in ("q", "qdot_exec", "qdot_max_eroded", "q_min_eroded", "q_max_eroded"):
        _vector(evidence.get(key), f"l0r_evidence.{key}", errors)
        _require_exact_values(event, f"/l0r_evidence/{key}", 5, errors)
    matrix = evidence.get("mass_matrix")
    if not isinstance(matrix, list) or len(matrix) != 5 or any(not isinstance(row, list) or len(row) != 5 for row in matrix):
        errors.append("l0r mass_matrix must be 5x5")
    else:
        for index in range(5):
            _require_exact_values(event, f"/l0r_evidence/mass_matrix/{index}", 5, errors)
    if not isinstance(evidence.get("collision_distances"), dict) or not evidence["collision_distances"] or not all(isinstance(name, str) for name in evidence["collision_distances"]):
        errors.append("l0r collision distances must be a nonempty named map")
    else:
        for name in evidence["collision_distances"]:
            _require_exact_pointer(event, f"/l0r_evidence/collision_distances/{_escape(name)}", errors)
    for key in ("dt", "e_max_safe", "d_safe"):
        _require_exact_pointer(event, f"/l0r_evidence/{key}", errors)


def _shift(event: Mapping[str, Any], errors: list[str]) -> None:
    kind, params = event.get("shift_type"), event.get("shift_params")
    if not isinstance(params, dict):
        errors.append("shift_params must be a closed typed record")
    elif kind == "anchor_drift":
        if set(params) != {"anchor", "drift_xyz"} or not isinstance(params.get("anchor"), str) or not params["anchor"] or not _wire_vector(params.get("drift_xyz"), 3):
            errors.append("anchor_drift shift_params must be closed typed evidence")
    elif kind == "safety_io_flip":
        if set(params) != {"signal"} or not isinstance(params.get("signal"), str) or not params["signal"]:
            errors.append("safety_io_flip shift_params must be closed typed evidence")
    elif kind == "sensor_stale":
        if not params or not set(params) <= {"frame_age_ms", "timestamp_override"} or any(not _wire_scalar(value) for value in params.values()):
            errors.append("sensor_stale shift_params must be closed typed evidence")
    elif kind == "base_drift":
        if not params or not set(params) <= {"drift_xyz", "drift_rpy"} or any(not _wire_vector(value, 3) for value in params.values()):
            errors.append("base_drift shift_params must be closed typed evidence")
    else:
        errors.append("unknown shift type")
    if not isinstance(event.get("affected_channels"), list) or not all(isinstance(value, str) for value in event["affected_channels"]):
        errors.append("affected_channels must be a string list")
    channels = {"anchor_drift": ["anchor_poses"], "safety_io_flip": ["safety_io"], "sensor_stale": ["sensor_health"], "base_drift": ["base_pose_estimate"]}
    if event.get("affected_channels") != channels.get(event.get("shift_type")):
        errors.append("affected_channels must match the normative shift channel")


def _episode(event: Mapping[str, Any], errors: list[str]) -> None:
    totals, transitions = event.get("total_interventions"), event.get("state_transitions")
    if not isinstance(totals, dict) or set(totals) != {"L0C", "L0R", "L1", "RTA"} or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in totals.values()):
        errors.append("invalid intervention totals")
    if not isinstance(transitions, list) or event.get("state_transition_count") != len(transitions):
        errors.append("invalid transition summary/count")
    elif any(not isinstance(item, dict) or set(item) != {"step", "from", "to", "trigger"} or isinstance(item.get("step"), bool) or not isinstance(item.get("step"), int) or item["step"] < 0 or item.get("from") not in _STATES or item.get("to") not in _STATES or not isinstance(item.get("trigger"), str) or not item["trigger"] for item in transitions):
        errors.append("episode transitions must be closed typed records")
    synthetic, params = event.get("synthetic_revocation"), event.get("synthetic_revocation_params")
    if not isinstance(synthetic, bool) or (synthetic and (not isinstance(params, dict) or set(params) != {"trigger_step", "duration_steps", "injected_monitor"})) or (not synthetic and params is not None):
        errors.append("synthetic revocation fields are inconsistent")
    elif synthetic and (any(isinstance(params[key], bool) or not isinstance(params[key], int) or params[key] < 0 for key in ("trigger_step", "duration_steps")) or not isinstance(params["injected_monitor"], str) or not params["injected_monitor"]):
        errors.append("synthetic revocation params are malformed")
    if event.get("l1_arm") != "none": errors.append("l1_arm is reserved and must be none")


def _monitor_evidence(event: Mapping[str, Any], errors: list[str]) -> None:
    inputs, results = event.get("monitor_inputs"), event.get("monitor_results")
    unavailable = {"availability", "reason"}
    if isinstance(inputs, dict) and isinstance(results, dict) and set(inputs) == unavailable and set(results) == unavailable:
        if inputs.get("availability") == results.get("availability") == "unavailable" and all(isinstance(item.get("reason"), str) and item["reason"] for item in (inputs, results)) and event.get("authorized") is False and event.get("authority_state_post") == "RTA_STOP":
            return
        errors.append("unavailable monitor evidence must be typed and denied")
        return
    expected = {"anchor_poses", "safety_io", "sensor_health", "base_pose_estimate"}
    if not isinstance(inputs, dict) or set(inputs) != expected:
        errors.append("monitor inputs must be closed")
    else:
        anchors, base, sensor, io = inputs["anchor_poses"], inputs["base_pose_estimate"], inputs["sensor_health"], inputs["safety_io"]
        if not isinstance(anchors, dict) or not anchors or any(not isinstance(name, str) or not isinstance(value, dict) or set(value) != {"xyz", "rpy"} or any(not isinstance(value[key], list) or len(value[key]) != 3 for key in value) for name, value in anchors.items()): errors.append("anchor poses must be closed")
        if not isinstance(io, dict) or not io or not all(isinstance(name, str) and type(value) is bool for name, value in io.items()): errors.append("safety I/O must be typed")
        if not isinstance(sensor, dict) or set(sensor) != {"frame_age_ms", "timestamp"}: errors.append("sensor health must be closed")
        if not isinstance(base, dict) or set(base) != {"xyz", "rpy"} or any(not isinstance(base[key], list) or len(base[key]) != 3 for key in base): errors.append("base pose must be closed")
    fields = {"triggered", "should_trigger", "should_clear", "severity", "predicate_violated"}
    if not isinstance(results, dict) or not results or any(not isinstance(name, str) or not isinstance(value, dict) or set(value) != fields or not all(isinstance(value.get(key), bool) for key in ("triggered", "should_trigger", "should_clear", "predicate_violated")) or not isinstance(value.get("severity"), str) for name, value in results.items()): errors.append("monitor results must be closed typed records")


def _generic(event: Mapping[str, Any], errors: list[str]) -> None:
    for key in ("timestamp_ns", "step", "seed", "planned_steps", "total_steps", "total_episodes", "actual_event_count", "actual_episode_count", "state_transition_count", "full_head_staleness_episodes", "fn_count", "fp_count"):
        minimum = -1 if key == "step" and event.get("event_type") == "shift_injected" else 0
        if key in event and (isinstance(event[key], bool) or not isinstance(event[key], int) or event[key] < minimum): errors.append(f"{key} must be an integer >= {minimum}")
    for key in ("manifest_hash", "contract_hash", "environment_hash", "last_event_hash"):
        if key in event and (not isinstance(event[key], str) or not _HASH.fullmatch(event[key])): errors.append(f"{key} must be lowercase SHA256 hex")
    for key in ("episode_id", "authority_reason", "shift_type", "trigger_monitor"):
        if key in event and (not isinstance(event[key], str) or not event[key]): errors.append(f"{key} must be a non-empty string")


def _channels(event: Mapping[str, Any], errors: list[str]) -> None:
    truth, observed, believed = event.get("truth"), event.get("observed"), event.get("believed")
    truth_fields = {"q", "qdot", "tau", "tcp_pos", "tcp_to_keepout_distances", "self_collision_surface_distances"}
    if not isinstance(truth, dict) or set(truth) != truth_fields:
        errors.append("truth fields must exactly match schema")
    else:
        for key in ("q", "qdot", "tau"):
            _vector(truth.get(key), f"truth.{key}", errors)
            _require_exact_values(event, f"/truth/{key}", 5, errors)
        if not isinstance(truth["tcp_pos"], list) or len(truth["tcp_pos"]) != 3:
            errors.append("truth.tcp_pos must be length 3")
        else:
            _require_exact_values(event, "/truth/tcp_pos", 3, errors)
        for key in ("tcp_to_keepout_distances", "self_collision_surface_distances"):
            if not isinstance(truth[key], dict) or not truth[key] or not all(isinstance(name, str) for name in truth[key]):
                errors.append(f"truth.{key} must be a nonempty named map")
            else:
                for name in truth[key]:
                    _require_exact_pointer(event, f"/truth/{key}/{_escape(name)}", errors)
    if not isinstance(observed, dict) or set(observed) != {"q", "qdot", "tau"}:
        errors.append("observed fields must exactly match schema")
    else:
        for key in observed:
            _vector(observed[key], f"observed.{key}", errors)
            _require_exact_values(event, f"/observed/{key}", 5, errors)
    if not isinstance(believed, dict) or set(believed) != {"q_interval", "qdot_interval"}:
        errors.append("believed fields must exactly match schema")
    else:
        for key in believed:
            value = believed[key]
            if not isinstance(value, list) or len(value) != 5 or any(not isinstance(row, list) or len(row) != 2 for row in value):
                errors.append(f"believed.{key} must be 5x2")
            else:
                for index in range(5):
                    _require_exact_values(event, f"/believed/{key}/{index}", 2, errors)


def _vector(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 5:
        errors.append(f"{name} must be a length-5 vector")


def _wire_scalar(value: Any) -> bool:
    return isinstance(value, str) and _WIRE.fullmatch(value) is not None


def _wire_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_wire_scalar(item) for item in value)


def _require_exact_values(
    event: Mapping[str, Any], prefix: str, count: int, errors: list[str]
) -> None:
    for index in range(count):
        _require_exact_pointer(event, f"{prefix}/{index}", errors)


def _require_exact_pointer(
    event: Mapping[str, Any], pointer: str, errors: list[str]
) -> None:
    evidence = event.get(FLOAT_HEX_FIELD)
    if not isinstance(evidence, Mapping) or pointer not in evidence:
        errors.append(f"missing required exact numeric evidence at {pointer}")


def _float_evidence(event: Mapping[str, Any], errors: list[str]) -> None:
    evidence = event.get(FLOAT_HEX_FIELD)
    if not isinstance(evidence, dict):
        errors.append("float_hex must be an object")
        return
    wire = _pointers(event)
    if set(evidence) != set(wire):
        missing = sorted(set(wire) - set(evidence))
        orphaned = sorted(set(evidence) - set(wire))
        if missing:
            errors.append(f"float_hex missing wire float entries: {missing!r}")
        if orphaned:
            errors.append(f"float_hex has orphan entries: {orphaned!r}")
    for pointer, text in wire.items():
        try:
            value = exact(event, pointer)
            encoded = format(Decimal.from_float(value).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), ".6f")
            if encoded == "-0.000000": encoded = "0.000000"
            if text != encoded: errors.append(f"float evidence mismatch at {pointer}")
        except (ValueError, TypeError):
            errors.append(f"invalid float evidence at {pointer}")


def _pointers(value: Any, pointer: str = "") -> dict[str, str]:
    if isinstance(value, str) and _WIRE.fullmatch(value): return {pointer or "/": value}
    if isinstance(value, list):
        return {key: item for index, child in enumerate(value) for key, item in _pointers(child, f"{pointer}/{index}").items()}
    if isinstance(value, dict):
        return {key: item for name, child in value.items() if name != FLOAT_HEX_FIELD for key, item in _pointers(child, f"{pointer}/{_escape(name)}").items()}
    return {}


def _escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _sort_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_sort_nested(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _sort_nested(value[key]) for key in sorted(value)}
    return value
