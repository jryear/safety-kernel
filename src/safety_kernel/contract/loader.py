from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from safety_kernel.contract.canonical import canonicalize_contract_body
from safety_kernel.contract.errors import (
    ContractHashMismatchError,
    ContractValidationError,
)
from safety_kernel.contract.schema import Contract
from safety_kernel.contract.strict_yaml import load_yaml_strict


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_INT_DECIMAL_KEYS = {"debounce_ms", "hysteresis_ms", "max_frame_age_ms"}


@dataclass(frozen=True)
class LoadedContract:
    schema_version: str
    body_sha256: str
    canonical_body_bytes: bytes
    raw_contract: Mapping[str, Any]
    canonical_body_obj: Any
    runtime_contract: Any
    contract: Contract  # Typed contract dataclass


def load_contract_file(path: str | Path) -> LoadedContract:
    path = Path(path)
    return load_contract_text(path.read_text(encoding="utf-8"))


def load_contract_text(yaml_text: str) -> LoadedContract:
    raw = load_yaml_strict(yaml_text)
    if not isinstance(raw, Mapping):
        raise ContractValidationError("Contract document must be a mapping at top-level")

    _validate_schema_version(raw)
    _validate_unknown_fields(raw)
    _validate_integrity_profile(raw)

    canon = canonicalize_contract_body(raw)
    declared_hash = _get_declared_body_sha256(raw)
    if canon.body_sha256 != declared_hash:
        raise ContractHashMismatchError(
            f"Contract body_sha256 mismatch: declared={declared_hash} computed={canon.body_sha256}"
        )

    schema_version = raw["schema_version"]
    runtime = _to_runtime_types(raw)
    return LoadedContract(
        schema_version=schema_version,
        body_sha256=canon.body_sha256,
        canonical_body_bytes=canon.canonical_bytes,
        raw_contract=raw,
        canonical_body_obj=canon.canonical_obj,
        runtime_contract=runtime,
        contract=Contract.from_dict(runtime),
    )


def _to_runtime_types(obj: Any, *, parent_key: str | None = None) -> Any:
    """Convert Decimal scalars into runtime numeric types (int/float).

    This is *not* used for hashing/canonicalization; it's for monitors and the state machine.
    """
    if isinstance(obj, Decimal):
        if parent_key in _INT_DECIMAL_KEYS:
            return int(obj)
        return float(obj)
    if isinstance(obj, list):
        return [_to_runtime_types(v, parent_key=parent_key) for v in obj]
    if isinstance(obj, Mapping):
        return {k: _to_runtime_types(v, parent_key=k) for k, v in obj.items()}
    return obj


def _validate_schema_version(contract: Mapping[str, Any]) -> None:
    schema_version = contract.get("schema_version")
    if not isinstance(schema_version, str):
        raise ContractValidationError("schema_version must be a string")

    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", schema_version)
    if not m:
        raise ContractValidationError("schema_version must be MAJOR.MINOR.PATCH (e.g. '0.1.0')")

    major, minor, _patch = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if major != 0 or minor != 1:
        raise ContractValidationError(f"Unsupported schema_version {schema_version!r}; expected 0.1.*")


def _get_declared_body_sha256(contract: Mapping[str, Any]) -> str:
    integrity = contract.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ContractValidationError("integrity must be a mapping")
    body_sha256 = integrity.get("body_sha256")
    if not isinstance(body_sha256, str):
        raise ContractValidationError("integrity.body_sha256 must be a string")
    if not _SHA256_HEX_RE.match(body_sha256):
        raise ContractValidationError("integrity.body_sha256 must be lowercase hex sha256")
    return body_sha256


def _validate_integrity_profile(contract: Mapping[str, Any]) -> None:
    integrity = contract.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ContractValidationError("integrity must be a mapping")

    canonicalization = integrity.get("canonicalization")
    if not isinstance(canonicalization, Mapping):
        raise ContractValidationError("integrity.canonicalization must be a mapping")

    expected_keys = {"sort_keys", "float_quantization", "normalize_newlines", "unit_normalization"}
    if set(canonicalization.keys()) != expected_keys:
        raise ContractValidationError("integrity.canonicalization keys must match the v0.1 profile exactly")

    if canonicalization.get("sort_keys") is not True:
        raise ContractValidationError("integrity.canonicalization.sort_keys must be true")
    if canonicalization.get("normalize_newlines") is not True:
        raise ContractValidationError("integrity.canonicalization.normalize_newlines must be true")
    if canonicalization.get("unit_normalization") is not True:
        raise ContractValidationError("integrity.canonicalization.unit_normalization must be true")

    float_quant = canonicalization.get("float_quantization")
    if not isinstance(float_quant, Decimal) or float_quant != Decimal("0.000001"):
        raise ContractValidationError("integrity.canonicalization.float_quantization must equal 1e-6")


def _validate_unknown_fields(contract: Mapping[str, Any]) -> None:
    # v0.1 strict allow-list. Unknown keys are rejected (fail-closed), except metadata.extensions.
    allowed_top = {
        "schema_version",
        "metadata",
        "frames",
        "geometry",
        "anchors",
        "assumption_monitors",
        "revocation_policy",
        "integrity",
    }
    unknown = set(contract.keys()) - allowed_top
    if unknown:
        raise ContractValidationError(f"Unknown top-level keys not allowed: {sorted(unknown)!r}")

    metadata = contract.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContractValidationError("metadata must be a mapping")

    allowed_metadata = {
        "contract_id",
        "cell_id",
        "created_at",
        "created_by",
        "description",
        "robot_compat",
        "validity",
        "extensions",
    }
    unknown_metadata = set(metadata.keys()) - allowed_metadata
    if unknown_metadata:
        raise ContractValidationError(f"Unknown metadata keys not allowed: {sorted(unknown_metadata)!r}")

    # extensions is the only permissive escape hatch; if present it must be a mapping.
    extensions = metadata.get("extensions")
    if extensions is not None and not isinstance(extensions, Mapping):
        raise ContractValidationError("metadata.extensions must be a mapping when present")

    # Metadata required + type constraints (minimal but strict enough to fail closed).
    _require_str(metadata, "contract_id")
    _require_str(metadata, "cell_id")
    _require_str(metadata, "created_at")
    _require_str(metadata, "created_by")
    _require_str(metadata, "description")

    robot_compat = _require_mapping(metadata, "robot_compat")
    allowed_robot_compat = {"robot_model", "tool_flange", "payload_class_kg"}
    _reject_unknown_keys(robot_compat, allowed_robot_compat, "metadata.robot_compat")
    _require_list_of_str(robot_compat, "robot_model")
    if "tool_flange" in robot_compat:
        _require_str(robot_compat, "tool_flange")
    if "payload_class_kg" in robot_compat:
        payload = robot_compat["payload_class_kg"]
        if not (isinstance(payload, list) and len(payload) == 2):
            raise ContractValidationError("metadata.robot_compat.payload_class_kg must be a 2-element list")
        _require_decimal(payload[0], "metadata.robot_compat.payload_class_kg[0]")
        _require_decimal(payload[1], "metadata.robot_compat.payload_class_kg[1]")

    validity = _require_mapping(metadata, "validity")
    _reject_unknown_keys(validity, {"valid_from", "expires_at"}, "metadata.validity")
    _require_str(validity, "valid_from")
    expires_at = validity.get("expires_at")
    if expires_at is not None and not isinstance(expires_at, str):
        raise ContractValidationError("metadata.validity.expires_at must be a string or null")

    frames = _require_mapping(contract, "frames")
    _reject_unknown_keys(frames, {"units", "cell_frame", "robot_base_in_cell", "tool_frame"}, "frames")

    units = _require_mapping(frames, "units")
    _reject_unknown_keys(units, {"length", "angle"}, "frames.units")
    _require_str(units, "length")
    _require_str(units, "angle")

    cell_frame = _require_mapping(frames, "cell_frame")
    _reject_unknown_keys(cell_frame, {"name", "convention"}, "frames.cell_frame")
    _require_str(cell_frame, "name")
    _require_str(cell_frame, "convention")

    robot_base = _require_mapping(frames, "robot_base_in_cell")
    _reject_unknown_keys(robot_base, {"xyz", "rpy", "tolerance"}, "frames.robot_base_in_cell")
    _require_vec_decimal(robot_base, "xyz", 3)
    _require_vec_decimal(robot_base, "rpy", 3)
    tolerance = _require_mapping(robot_base, "tolerance")
    _reject_unknown_keys(tolerance, {"pos_m", "rot_rad"}, "frames.robot_base_in_cell.tolerance")
    _require_decimal(tolerance.get("pos_m"), "frames.robot_base_in_cell.tolerance.pos_m")
    _require_decimal(tolerance.get("rot_rad"), "frames.robot_base_in_cell.tolerance.rot_rad")

    if "tool_frame" in frames and frames["tool_frame"] is not None:
        tool_frame = _require_mapping(frames, "tool_frame")
        _reject_unknown_keys(tool_frame, {"name", "xyz", "rpy"}, "frames.tool_frame")
        _require_str(tool_frame, "name")
        _require_vec_decimal(tool_frame, "xyz", 3)
        _require_vec_decimal(tool_frame, "rpy", 3)

    geometry = _require_mapping(contract, "geometry")
    _reject_unknown_keys(geometry, {"keepouts", "allowed_regions"}, "geometry")
    _require_list_of_mappings(geometry, "keepouts")
    if "allowed_regions" in geometry:
        _require_list_of_mappings(geometry, "allowed_regions")

    for region in geometry.get("keepouts", []):
        _validate_region(region, "geometry.keepouts[]", is_keepout=True)
    for region in geometry.get("allowed_regions", []):
        _validate_region(region, "geometry.allowed_regions[]", is_keepout=False)

    _require_list_of_mappings(contract, "anchors")
    for anchor in contract.get("anchors", []):
        _validate_anchor(anchor, "anchors[]")

    _require_list_of_mappings(contract, "assumption_monitors")
    for monitor in contract.get("assumption_monitors", []):
        _validate_monitor(monitor, "assumption_monitors[]")

    revocation = _require_mapping(contract, "revocation_policy")
    _reject_unknown_keys(revocation, {"on_degrade", "on_revoke", "on_rta_stop"}, "revocation_policy")
    on_degrade = _require_mapping(revocation, "on_degrade")
    _reject_unknown_keys(on_degrade, {"l0c_still_enforced", "caps"}, "revocation_policy.on_degrade")
    if on_degrade.get("l0c_still_enforced") is not True and on_degrade.get("l0c_still_enforced") is not False:
        raise ContractValidationError("revocation_policy.on_degrade.l0c_still_enforced must be boolean")
    caps = _require_mapping(on_degrade, "caps")
    _reject_unknown_keys(caps, {"max_tcp_speed_mps", "restrict_to_allowed_regions"}, "revocation_policy.on_degrade.caps")
    _require_decimal(caps.get("max_tcp_speed_mps"), "revocation_policy.on_degrade.caps.max_tcp_speed_mps")
    if caps.get("restrict_to_allowed_regions") is not True and caps.get("restrict_to_allowed_regions") is not False:
        raise ContractValidationError("revocation_policy.on_degrade.caps.restrict_to_allowed_regions must be boolean")

    on_revoke = _require_mapping(revocation, "on_revoke")
    _reject_unknown_keys(on_revoke, {"l0c_enabled", "l1_mode"}, "revocation_policy.on_revoke")
    if on_revoke.get("l0c_enabled") is not True and on_revoke.get("l0c_enabled") is not False:
        raise ContractValidationError("revocation_policy.on_revoke.l0c_enabled must be boolean")
    _require_str(on_revoke, "l1_mode")

    on_rta_stop = _require_mapping(revocation, "on_rta_stop")
    _reject_unknown_keys(on_rta_stop, {"sequence"}, "revocation_policy.on_rta_stop")
    seq = on_rta_stop.get("sequence")
    if not isinstance(seq, list) or not all(isinstance(x, str) for x in seq):
        raise ContractValidationError("revocation_policy.on_rta_stop.sequence must be a list of strings")

    integrity = _require_mapping(contract, "integrity")
    _reject_unknown_keys(
        integrity,
        {"canonicalization", "body_sha256", "signature", "signer_id", "signed_at"},
        "integrity",
    )
    _require_str(integrity, "signer_id")
    signed_at = integrity.get("signed_at")
    if signed_at is not None and not isinstance(signed_at, str):
        raise ContractValidationError("integrity.signed_at must be a string or null")
    signature = integrity.get("signature")
    if signature is not None and not isinstance(signature, str):
        raise ContractValidationError("integrity.signature must be a string or null")


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{key} must be a mapping")
    return value


def _require_str(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str):
        raise ContractValidationError(f"{key} must be a string")
    return value


def _require_decimal(value: Any, path: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise ContractValidationError(f"{path} must be a number")
    return value


def _reject_unknown_keys(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(mapping.keys()) - allowed
    if unknown:
        raise ContractValidationError(f"Unknown keys not allowed at {path}: {sorted(unknown)!r}")


def _require_vec_decimal(parent: Mapping[str, Any], key: str, n: int) -> None:
    value = parent.get(key)
    if not isinstance(value, list) or len(value) != n:
        raise ContractValidationError(f"{key} must be a {n}-vector list")
    for i, v in enumerate(value):
        _require_decimal(v, f"{key}[{i}]")


def _require_list_of_str(parent: Mapping[str, Any], key: str) -> None:
    value = parent.get(key)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ContractValidationError(f"{key} must be a list of strings")


def _require_list_of_mappings(parent: Mapping[str, Any], key: str) -> None:
    value = parent.get(key)
    if not isinstance(value, list) or not all(isinstance(x, Mapping) for x in value):
        raise ContractValidationError(f"{key} must be a list of mappings")


def _validate_region(region: Mapping[str, Any], path: str, *, is_keepout: bool) -> None:
    allowed = {"name", "type", "pose_in_cell", "dims", "tolerance", "enforcement"}
    _reject_unknown_keys(region, allowed, path)
    if not isinstance(region.get("name"), str):
        raise ContractValidationError(f"{path}.name must be a string")
    if region.get("type") not in {"box", "cylinder", "sphere"}:
        raise ContractValidationError(f"{path}.type must be one of box/cylinder/sphere")

    pose = region.get("pose_in_cell")
    if not isinstance(pose, Mapping):
        raise ContractValidationError(f"{path}.pose_in_cell must be a mapping")
    _reject_unknown_keys(pose, {"xyz", "rpy"}, f"{path}.pose_in_cell")
    _require_vec_decimal(pose, "xyz", 3)
    _require_vec_decimal(pose, "rpy", 3)

    dims = region.get("dims")
    if not isinstance(dims, Mapping):
        raise ContractValidationError(f"{path}.dims must be a mapping")

    region_type = region["type"]
    if region_type == "box":
        _reject_unknown_keys(dims, {"x", "y", "z"}, f"{path}.dims")
        _require_decimal(dims.get("x"), f"{path}.dims.x")
        _require_decimal(dims.get("y"), f"{path}.dims.y")
        _require_decimal(dims.get("z"), f"{path}.dims.z")
    elif region_type == "cylinder":
        _reject_unknown_keys(dims, {"radius", "height"}, f"{path}.dims")
        _require_decimal(dims.get("radius"), f"{path}.dims.radius")
        _require_decimal(dims.get("height"), f"{path}.dims.height")
    else:  # sphere
        _reject_unknown_keys(dims, {"radius"}, f"{path}.dims")
        _require_decimal(dims.get("radius"), f"{path}.dims.radius")

    tolerance = region.get("tolerance")
    if not isinstance(tolerance, Mapping):
        raise ContractValidationError(f"{path}.tolerance must be a mapping")
    _reject_unknown_keys(tolerance, {"inflate_m", "deflate_m"}, f"{path}.tolerance")
    if is_keepout:
        if "inflate_m" not in tolerance or "deflate_m" in tolerance:
            raise ContractValidationError(f"{path}.tolerance must contain inflate_m only for keepouts")
        _require_decimal(tolerance.get("inflate_m"), f"{path}.tolerance.inflate_m")
    else:
        if "deflate_m" not in tolerance or "inflate_m" in tolerance:
            raise ContractValidationError(f"{path}.tolerance must contain deflate_m only for allowed_regions")
        _require_decimal(tolerance.get("deflate_m"), f"{path}.tolerance.deflate_m")

    enforcement = region.get("enforcement")
    if not isinstance(enforcement, Mapping):
        raise ContractValidationError(f"{path}.enforcement must be a mapping")
    allowed_enforcement = {"hard_keepout", "caps_when_active"}
    _reject_unknown_keys(enforcement, allowed_enforcement, f"{path}.enforcement")
    hk = enforcement.get("hard_keepout")
    if hk is not True and hk is not False:
        raise ContractValidationError(f"{path}.enforcement.hard_keepout must be boolean")

    caps = enforcement.get("caps_when_active")
    if caps is not None:
        if not isinstance(caps, Mapping):
            raise ContractValidationError(f"{path}.enforcement.caps_when_active must be a mapping")
        _reject_unknown_keys(caps, {"max_tcp_speed_mps"}, f"{path}.enforcement.caps_when_active")
        _require_decimal(caps.get("max_tcp_speed_mps"), f"{path}.enforcement.caps_when_active.max_tcp_speed_mps")


def _validate_anchor(anchor: Mapping[str, Any], path: str) -> None:
    allowed = {"name", "type", "sensor", "expected_pose_in_cell", "tolerance", "required"}
    _reject_unknown_keys(anchor, allowed, path)
    if not isinstance(anchor.get("name"), str):
        raise ContractValidationError(f"{path}.name must be a string")
    if anchor.get("type") != "fiducial":
        raise ContractValidationError(f"{path}.type must be 'fiducial' in v0.1")
    if not isinstance(anchor.get("sensor"), str):
        raise ContractValidationError(f"{path}.sensor must be a string")

    expected = anchor.get("expected_pose_in_cell")
    if not isinstance(expected, Mapping):
        raise ContractValidationError(f"{path}.expected_pose_in_cell must be a mapping")
    _reject_unknown_keys(expected, {"xyz", "rpy"}, f"{path}.expected_pose_in_cell")
    _require_vec_decimal(expected, "xyz", 3)
    _require_vec_decimal(expected, "rpy", 3)

    tol = anchor.get("tolerance")
    if not isinstance(tol, Mapping):
        raise ContractValidationError(f"{path}.tolerance must be a mapping")
    _reject_unknown_keys(tol, {"pos_m", "rot_rad"}, f"{path}.tolerance")
    _require_decimal(tol.get("pos_m"), f"{path}.tolerance.pos_m")
    _require_decimal(tol.get("rot_rad"), f"{path}.tolerance.rot_rad")

    req = anchor.get("required")
    if req is not True and req is not False:
        raise ContractValidationError(f"{path}.required must be boolean")


def _validate_monitor(monitor: Mapping[str, Any], path: str) -> None:
    allowed = {"name", "type", "inputs", "params", "debounce_ms", "hysteresis_ms", "severity", "action"}
    _reject_unknown_keys(monitor, allowed, path)
    if not isinstance(monitor.get("name"), str):
        raise ContractValidationError(f"{path}.name must be a string")
    mtype = monitor.get("type")
    if mtype not in {"anchor_pose_consistency", "safety_io_health", "sensor_health", "robot_mount_integrity"}:
        raise ContractValidationError(f"{path}.type unsupported in v0.1: {mtype!r}")

    inputs = monitor.get("inputs")
    if not isinstance(inputs, list) or not all(isinstance(x, str) for x in inputs):
        raise ContractValidationError(f"{path}.inputs must be a list of strings")

    params = monitor.get("params")
    if not isinstance(params, Mapping):
        raise ContractValidationError(f"{path}.params must be a mapping")
    _validate_monitor_params(params, f"{path}.params", mtype)

    debounce = monitor.get("debounce_ms")
    hysteresis = monitor.get("hysteresis_ms")
    _require_nonneg_int_decimal(debounce, f"{path}.debounce_ms")
    _require_nonneg_int_decimal(hysteresis, f"{path}.hysteresis_ms")

    sev = monitor.get("severity")
    if sev not in {"degrade", "revoke", "rta_stop"}:
        raise ContractValidationError(f"{path}.severity must be degrade/revoke/rta_stop")
    if not isinstance(monitor.get("action"), str):
        raise ContractValidationError(f"{path}.action must be a string")


def _validate_monitor_params(params: Mapping[str, Any], path: str, mtype: str) -> None:
    if mtype == "anchor_pose_consistency":
        _reject_unknown_keys(params, {"anchors", "aggregation"}, path)
        anchors = params.get("anchors")
        if not isinstance(anchors, list) or not all(isinstance(x, str) for x in anchors):
            raise ContractValidationError(f"{path}.anchors must be a list of strings")
        if not isinstance(params.get("aggregation"), str):
            raise ContractValidationError(f"{path}.aggregation must be a string")
    elif mtype == "safety_io_health":
        _reject_unknown_keys(params, {"signal", "expected"}, path)
        if not isinstance(params.get("signal"), str):
            raise ContractValidationError(f"{path}.signal must be a string")
        expected = params.get("expected")
        if expected is not True and expected is not False:
            raise ContractValidationError(f"{path}.expected must be boolean")
    elif mtype == "sensor_health":
        _reject_unknown_keys(params, {"max_frame_age_ms", "monotonic_timestamps", "calibration_checksum"}, path)
        _require_nonneg_int_decimal(params.get("max_frame_age_ms"), f"{path}.max_frame_age_ms")
        mono = params.get("monotonic_timestamps")
        if mono is not True and mono is not False:
            raise ContractValidationError(f"{path}.monotonic_timestamps must be boolean")
        if not isinstance(params.get("calibration_checksum"), str):
            raise ContractValidationError(f"{path}.calibration_checksum must be a string")
    else:  # robot_mount_integrity
        _reject_unknown_keys(params, {"base_pose_drift_pos_m", "base_pose_drift_rot_rad"}, path)
        _require_decimal(params.get("base_pose_drift_pos_m"), f"{path}.base_pose_drift_pos_m")
        _require_decimal(params.get("base_pose_drift_rot_rad"), f"{path}.base_pose_drift_rot_rad")


def _require_nonneg_int_decimal(value: Any, path: str) -> None:
    if not isinstance(value, Decimal):
        raise ContractValidationError(f"{path} must be a number")
    if value != value.to_integral_value():
        raise ContractValidationError(f"{path} must be an integer")
    if value < 0:
        raise ContractValidationError(f"{path} must be non-negative")
