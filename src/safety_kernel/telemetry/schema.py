"""Closed, fail-closed event ABI for accepted Gate 2 telemetry."""

from __future__ import annotations

import re
from typing import Any, Mapping

from safety_kernel.telemetry.canonical import FLOAT_HEX_FIELD, ZERO_HASH, authoritative_float, validate_float_evidence
from safety_kernel.telemetry.errors import TelemetryError

SCHEMA_VERSION = "0.2.0"
COMMAND_AUTHORITY = "software_command_suppression_only"
EVENT_TYPES = (
    "run_start", "episode_start", "shift_injected", "decision_evidence",
    "safety_intervention", "contract_state_change", "kernel_step_summary",
    "episode_summary", "run_end",
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_WIRE = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}$")
_STATES = {"ACTIVE", "DEGRADED", "REVOKED", "RTA_STOP"}
_COMMON = {"event_type", "prev_hash", FLOAT_HEX_FIELD}
_FIELDS: dict[str, set[str]] = {
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


def validate_event(event: Mapping[str, Any], *, first: bool = False) -> list[str]:
    event_type = event.get("event_type")
    if event_type not in _FIELDS:
        return [f"unknown event_type: {event_type!r}"]
    errors: list[str] = []
    actual = set(event)
    if actual != _FIELDS[event_type]:
        errors.append(f"{event_type} fields must exactly match schema; missing={sorted(_FIELDS[event_type] - actual)!r} unknown={sorted(actual - _FIELDS[event_type])!r}")
    prev = event.get("prev_hash")
    if not isinstance(prev, str) or not _HASH.fullmatch(prev):
        errors.append("prev_hash must be lowercase SHA256 hex")
    if first and (event_type != "run_start" or prev != ZERO_HASH):
        errors.append("first event must be run_start with zero prev_hash")
    if event_type == "run_start" and (event.get("schema_version") != SCHEMA_VERSION or event.get("command_authority") != COMMAND_AUTHORITY):
        errors.append("run_start schema or command authority is invalid")
    if event_type == "run_end" and event.get("command_authority") != COMMAND_AUTHORITY:
        errors.append("run_end command authority claim is invalid")
    if event_type == "run_end" and event.get("status") != "completed":
        errors.append("run_end status must be completed")
    _validate_generic_fields(event, event_type, errors)
    if event_type == "episode_start" and (not isinstance(event.get("planned_steps"), int) or event["planned_steps"] < 0):
        errors.append("episode_start planned_steps must be a non-negative integer")
    if event_type == "decision_evidence":
        if event.get("authority_state_pre") not in _STATES or event.get("authority_state_post") not in _STATES:
            errors.append("invalid decision authority state")
        for field in ("proposal", "l0c_command", "l0r_command", "final_command"):
            _require_exact_vector(event, field, 5, errors)
        _require_channel(event, "truth", {"q", "qdot", "tau", "tcp_pos", "tcp_to_keepout_distances", "self_collision_surface_distances"}, errors)
        _require_channel(event, "observed", {"q", "qdot", "tau"}, errors)
        _require_channel(event, "believed", {"q_interval", "qdot_interval"}, errors)
        _require_monitor_evidence(event, errors)
        if not isinstance(event.get("authorized"), bool):
            errors.append("decision authorized must be boolean")
        elif event.get("authorized") != (event.get("authority_state_post") not in {"REVOKED", "RTA_STOP"}):
            errors.append("decision authorization contradicts post-state authority")
        if not isinstance(event.get("authority_reason"), str) or not event["authority_reason"]:
            errors.append("decision authority_reason must be a non-empty string")
        if event.get("reset_input") is not None and not isinstance(event.get("reset_input"), bool):
            errors.append("decision reset_input must be boolean or null")
        if type(event.get("rta_nearmiss")) is not bool:
            errors.append("decision rta_nearmiss must be boolean")
        if event.get("authorized") is False:
            for index in range(5):
                try:
                    if authoritative_float(event, f"/final_command/{index}").hex() != (0.0).hex():
                        errors.append("unauthorized final_command authoritative values must be positive zero")
                        break
                except TelemetryError:
                    errors.append("unauthorized final_command lacks authoritative zero evidence")
                    break
    # L1 is reserved in 0.2.0: its event semantics and causal ordering are not
    # defined, so a producer must not smuggle it into an acceptance chain.
    if event_type == "safety_intervention":
        _require_intervention(event, errors)
    if event_type == "kernel_step_summary" and event.get("contract_state") not in _STATES:
        errors.append("invalid contract state")
    if event_type == "kernel_step_summary":
        for key in ("l0r_intervened", "l0c_active", "l0c_intervened", "rta_nearmiss"):
            if not isinstance(event.get(key), bool):
                errors.append(f"{key} must be boolean")
        cap = event.get("l0c_cap_evidence")
        if cap is not None:
            if not isinstance(cap, dict) or set(cap) != {"tcp_speed_after", "max_tcp_speed_mps", "tcp_jacobian"}:
                errors.append("l0c cap evidence must include closed command-stage Jacobian evidence")
            elif (not isinstance(cap["tcp_jacobian"], list) or len(cap["tcp_jacobian"]) != 3
                  or any(not isinstance(row, list) or len(row) != 5 for row in cap["tcp_jacobian"])):
                errors.append("l0c cap Jacobian must be 3x5")
    if event_type == "kernel_step_summary" and event.get("l0r_evidence") is not None:
        evidence = event.get("l0r_evidence")
        required = {"q", "qdot_exec", "qdot_max_eroded", "q_min_eroded", "q_max_eroded", "dt", "mass_matrix", "e_max_safe", "d_safe", "collision_distances"}
        if not isinstance(evidence, dict) or set(evidence) != required:
            errors.append("l0r evidence fields must exactly match schema")
        else:
            for key in ("q", "qdot_exec", "qdot_max_eroded", "q_min_eroded", "q_max_eroded"):
                value = evidence.get(key)
                if not isinstance(value, list) or len(value) != 5:
                    errors.append(f"l0r_evidence.{key} must be a length-5 vector")
            matrix = evidence.get("mass_matrix")
            if not isinstance(matrix, list) or len(matrix) != 5 or any(not isinstance(row, list) or len(row) != 5 for row in matrix):
                errors.append("l0r_evidence.mass_matrix must be 5x5")
            distances = evidence.get("collision_distances")
            if not isinstance(distances, dict) or not distances:
                errors.append("l0r_evidence.collision_distances must be non-empty")
    if event_type == "episode_summary":
        totals = event.get("total_interventions")
        transitions = event.get("state_transitions")
        if not isinstance(totals, dict) or set(totals) != {"L0C", "L0R", "L1", "RTA"} or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in totals.values()):
            errors.append("episode total_interventions must contain exact non-negative layer counts")
        if not isinstance(transitions, list) or not isinstance(event.get("state_transition_count"), int) or event["state_transition_count"] != len(transitions):
            errors.append("episode transition count must equal transition summary length")
        for transition in transitions if isinstance(transitions, list) else ():
            if not isinstance(transition, dict) or set(transition) != {"step", "from", "to", "trigger"} or isinstance(transition.get("step"), bool) or not isinstance(transition.get("step"), int) or transition["step"] < 0 or transition.get("from") not in _STATES or transition.get("to") not in _STATES or not isinstance(transition.get("trigger"), str) or not transition["trigger"]:
                errors.append("episode transition summaries must be closed typed records")
                break
        synthetic, params = event.get("synthetic_revocation"), event.get("synthetic_revocation_params")
        if not isinstance(synthetic, bool) or (synthetic and (not isinstance(params, dict) or set(params) != {"trigger_step", "duration_steps", "injected_monitor"})) or (not synthetic and params is not None):
            errors.append("synthetic revocation and closed params must be consistent")
        elif synthetic and (any(isinstance(params[key], bool) or not isinstance(params[key], int) or params[key] < 0 for key in ("trigger_step", "duration_steps")) or not isinstance(params["injected_monitor"], str) or not params["injected_monitor"]):
            errors.append("synthetic revocation params must be closed typed values")
        if event.get("l1_arm") != "none":
            errors.append("l1_arm is reserved and must be none in telemetry 0.2.0")
    if event_type == "contract_state_change":
        _require_transition_status(event, errors)
    if event_type == "shift_injected":
        _require_shift_params(event, errors)
        expected_channels = {
            "anchor_drift": ["anchor_poses"], "safety_io_flip": ["safety_io"],
            "sensor_stale": ["sensor_health"], "base_drift": ["base_pose_estimate"],
        }
        if event.get("affected_channels") != expected_channels.get(event.get("shift_type")):
            errors.append("affected_channels must match the normative shift channel")
    _reject_raw_float(event, errors)
    errors.extend(validate_float_evidence(event))
    return errors


def _require_intervention(event: Mapping[str, Any], errors: list[str]) -> None:
    layer = event.get("layer")
    if layer not in {"L0C", "L0R", "RTA"}:
        errors.append("invalid intervention layer")
    for field in ("qdot_cmd", "qdot_exec"):
        _require_exact_vector(event, field, 5, errors)
    if type(event.get("rta_nearmiss")) is not bool:
        errors.append("intervention rta_nearmiss must be boolean")
    constraints = event.get("constraints_active")
    if not isinstance(constraints, list) or any(not isinstance(item, str) or not item for item in constraints):
        errors.append("intervention constraints must be a typed string list")
    pairs = event.get("collision_pairs")
    if not isinstance(pairs, list) or any(not isinstance(pair, dict) or set(pair) != {"pair_id", "distance"} or not isinstance(pair.get("pair_id"), str) or not pair["pair_id"] for pair in pairs):
        errors.append("collision_pairs must be closed pair evidence")
    if layer == "L0R":
        margins = event.get("margin_to_L0R")
        if not constraints or not isinstance(margins, dict) or not margins or not all(isinstance(name, str) and name for name in margins) or event.get("layer_info") != {}:
            errors.append("L0R intervention evidence must be closed")
    elif layer == "L0C":
        info = event.get("layer_info")
        if event.get("margin_to_L0R") != {} or not isinstance(info, dict) or set(info) != {"tcp_speed_before", "tcp_speed_after", "scaling_factor"}:
            errors.append("L0C layer_info must be closed")
    elif layer == "RTA":
        info = event.get("layer_info")
        if event.get("margin_to_L0R") != {} or not isinstance(info, dict) or set(info) != {"authority_reason"} or not isinstance(info.get("authority_reason"), str) or not info["authority_reason"]:
            errors.append("RTA layer_info must be a closed authority reason")


def _require_shift_params(event: Mapping[str, Any], errors: list[str]) -> None:
    kind, params = event.get("shift_type"), event.get("shift_params")
    if not isinstance(params, dict):
        errors.append("shift_params must be a closed typed record")
        return
    if kind == "anchor_drift":
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


def _wire_scalar(value: Any) -> bool:
    return isinstance(value, str) and _WIRE.fullmatch(value) is not None


def _wire_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_wire_scalar(item) for item in value)


def _require_exact_vector(
    event: Mapping[str, Any], field: str, length: int, errors: list[str]
) -> None:
    value = event.get(field)
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"{field} must be a length-{length} vector")
        return
    evidence = event.get(FLOAT_HEX_FIELD)
    if not isinstance(evidence, dict) or any(
        f"/{field}/{index}" not in evidence for index in range(length)
    ):
        errors.append(f"{field} lacks exact float evidence")


def _require_channel(
    event: Mapping[str, Any], field: str, keys: set[str], errors: list[str]
) -> None:
    value = event.get(field)
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        errors.append(
            f"{field} fields must exactly match schema; "
            f"missing={sorted(keys - actual)!r} unknown={sorted(actual - keys)!r}"
        )
        return
    if field in {"truth", "observed"}:
        for key in ("q", "qdot", "tau"):
            vector = value.get(key)
            if not isinstance(vector, list) or len(vector) != 5:
                errors.append(f"{field}.{key} must be a length-5 vector")
    elif field == "believed":
        for key in ("q_interval", "qdot_interval"):
            intervals = value.get(key)
            if not isinstance(intervals, list) or len(intervals) != 5 or any(
                not isinstance(interval, list) or len(interval) != 2
                for interval in intervals
            ):
                errors.append(f"believed.{key} must be a 5x2 interval matrix")


def _require_monitor_evidence(event: Mapping[str, Any], errors: list[str]) -> None:
    inputs, results = event.get("monitor_inputs"), event.get("monitor_results")
    unavailable = {"availability", "reason"}
    if isinstance(inputs, dict) and isinstance(results, dict) and set(inputs) == unavailable and set(results) == unavailable:
        if inputs.get("availability") == results.get("availability") == "unavailable" and all(isinstance(item.get("reason"), str) and item["reason"] for item in (inputs, results)) and event.get("authorized") is False and event.get("authority_state_post") == "RTA_STOP":
            return
        errors.append("unavailable monitor evidence must be typed and denied")
        return
    keys = {"anchor_poses", "safety_io", "sensor_health", "base_pose_estimate"}
    if not isinstance(inputs, dict) or set(inputs) != keys:
        errors.append("decision monitor_inputs must be a closed full or unavailable record")
    else:
        anchors = inputs["anchor_poses"]
        if not isinstance(anchors, dict) or not anchors or any(not isinstance(name, str) or not isinstance(pose, dict) or set(pose) != {"xyz", "rpy"} or any(not isinstance(pose[key], list) or len(pose[key]) != 3 for key in pose) for name, pose in anchors.items()): errors.append("monitor anchor inputs must be closed poses")
        if not isinstance(inputs["safety_io"], dict) or not inputs["safety_io"] or not all(isinstance(name, str) and type(value) is bool for name, value in inputs["safety_io"].items()): errors.append("monitor safety I/O inputs must be typed")
        sensor = inputs["sensor_health"]
        if not isinstance(sensor, dict) or set(sensor) != {"frame_age_ms", "timestamp"}: errors.append("monitor sensor inputs must be closed")
        base = inputs["base_pose_estimate"]
        if not isinstance(base, dict) or set(base) != {"xyz", "rpy"} or any(not isinstance(base[key], list) or len(base[key]) != 3 for key in base): errors.append("monitor base inputs must be closed")
    fields = {"triggered", "should_trigger", "should_clear", "severity", "predicate_violated"}
    if not isinstance(results, dict) or not results or any(not isinstance(name, str) or not isinstance(value, dict) or set(value) != fields or not all(type(value.get(key)) is bool for key in ("triggered", "should_trigger", "should_clear", "predicate_violated")) or not isinstance(value.get("severity"), str) for name, value in results.items()): errors.append("monitor results must be closed typed records")


def _require_transition_status(event: Mapping[str, Any], errors: list[str]) -> None:
    status = event.get("monitors_status")
    if not isinstance(status, dict) or not status or any(not isinstance(name, str) or not isinstance(value, dict) or set(value) != {"triggered", "violated"} or not all(type(value.get(key)) is bool for key in ("triggered", "violated")) for name, value in status.items()):
        errors.append("transition monitor status must be closed typed records")


def _validate_generic_fields(event: Mapping[str, Any], event_type: str, errors: list[str]) -> None:
    for key in ("timestamp_ns", "step", "seed", "planned_steps", "total_steps", "total_episodes", "actual_event_count", "actual_episode_count", "state_transition_count", "full_head_staleness_episodes", "fn_count", "fp_count"):
        minimum = -1 if key == "step" and event_type == "shift_injected" else 0
        if key in event and (isinstance(event[key], bool) or not isinstance(event[key], int) or event[key] < minimum): errors.append(f"{key} must be an integer >= {minimum}")
    for key in ("manifest_hash", "contract_hash", "environment_hash", "last_event_hash"):
        if key in event and (not isinstance(event[key], str) or not _HASH.fullmatch(event[key])): errors.append(f"{key} must be lowercase SHA256 hex")
    for key in ("episode_id", "authority_reason", "shift_type", "trigger_monitor"):
        if key in event and (not isinstance(event[key], str) or not event[key]): errors.append(f"{key} must be a non-empty string")


def _reject_raw_float(value: Any, errors: list[str], path: str = "") -> None:
    if isinstance(value, float):
        errors.append(f"raw float is forbidden at {path or '/'}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_raw_float(item, errors, f"{path}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            if key != FLOAT_HEX_FIELD:
                _reject_raw_float(item, errors, f"{path}/{key}")
