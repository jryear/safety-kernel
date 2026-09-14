"""Canonical producer bytes and exact float evidence for telemetry 0.2.0."""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from safety_kernel.telemetry.errors import TelemetryError

ZERO_HASH = "0" * 64
FLOAT_HEX_FIELD = "float_hex"
_SIX = Decimal("0.000001")
_WIRE_FLOAT = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}")


def encode_float(value: float) -> str:
    """The non-authoritative six-place display form for a finite binary float."""
    if not math.isfinite(value):
        raise TelemetryError("non-finite float is not telemetry evidence")
    rounded = Decimal.from_float(float(value)).quantize(_SIX, rounding=ROUND_HALF_UP)
    return format(Decimal(0) if rounded == 0 else rounded, ".6f")


def prepare_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Encode floats once as display strings and once as authoritative hex values."""
    if not isinstance(event.get("event_type"), str):
        raise TelemetryError("event_type is required")
    if FLOAT_HEX_FIELD in event:
        raise TelemetryError("float_hex is owned by the canonical encoder")
    evidence: dict[str, str] = {}
    prepared = _normalise(dict(event), "", evidence)
    prepared[FLOAT_HEX_FIELD] = evidence
    return prepared


def event_bytes(event: Mapping[str, Any]) -> bytes:
    """Compact UTF-8 JSON with event_type physically first on every line."""
    if "event_type" not in event:
        raise TelemetryError("event_type is required")
    ordered = {"event_type": _sort_nested(event["event_type"])}
    ordered.update((key, _sort_nested(event[key])) for key in sorted(event) if key != "event_type")
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def event_hash(event: Mapping[str, Any]) -> str:
    return hashlib.sha256(event_bytes(event)).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_float_evidence(event: Mapping[str, Any]) -> list[str]:
    """Require a bijection between every wire float and its canonical float.hex value."""
    errors: list[str] = []
    evidence = event.get(FLOAT_HEX_FIELD)
    if not isinstance(evidence, dict):
        return ["float_hex must be an object"]
    wire = _wire_float_pointers(event)
    if set(evidence) != set(wire):
        missing, orphaned = sorted(set(wire) - set(evidence)), sorted(set(evidence) - set(wire))
        if missing:
            errors.append(f"float_hex missing wire float entries: {missing!r}")
        if orphaned:
            errors.append(f"float_hex has orphan entries: {orphaned!r}")
    for pointer, hex_value in evidence.items():
        if not isinstance(pointer, str) or not isinstance(hex_value, str):
            errors.append("float_hex entries must map pointers to strings")
            continue
        try:
            exact = float.fromhex(hex_value)
        except ValueError:
            errors.append(f"invalid float hex at {pointer!r}")
            continue
        if not math.isfinite(exact) or exact.hex() != hex_value:
            errors.append(f"non-canonical or non-finite float hex at {pointer!r}")
        elif wire.get(pointer) != encode_float(exact):
            errors.append(f"float evidence mismatch at {pointer!r}")
    return errors


def authoritative_float(event: Mapping[str, Any], pointer: str) -> float:
    """Resolve a numeric decision from exact evidence, never its display summary."""
    evidence = event.get(FLOAT_HEX_FIELD)
    if not isinstance(evidence, dict) or not isinstance(evidence.get(pointer), str):
        raise TelemetryError(f"missing authoritative float evidence at {pointer}")
    exact = float.fromhex(evidence[pointer])
    if not math.isfinite(exact) or exact.hex() != evidence[pointer]:
        raise TelemetryError(f"invalid authoritative float evidence at {pointer}")
    return exact


def _normalise(value: Any, pointer: str, evidence: dict[str, str]) -> Any:
    if isinstance(value, float):
        evidence[pointer or "/"] = value.hex()
        return encode_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_normalise(item, f"{pointer}/{index}", evidence) for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        return {str(key): _normalise(item, f"{pointer}/{_escape(str(key))}", evidence) for key, item in value.items()}
    raise TelemetryError(f"unsupported telemetry value type: {type(value).__name__}")


def _wire_float_pointers(value: Any, pointer: str = "") -> dict[str, str]:
    if isinstance(value, str) and _WIRE_FLOAT.fullmatch(value):
        return {pointer or "/": value}
    if isinstance(value, list):
        result: dict[str, str] = {}
        for index, item in enumerate(value):
            result.update(_wire_float_pointers(item, f"{pointer}/{index}"))
        return result
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, item in value.items():
            if key != FLOAT_HEX_FIELD:
                result.update(_wire_float_pointers(item, f"{pointer}/{_escape(str(key))}"))
        return result
    return {}


def _escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _sort_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_sort_nested(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _sort_nested(value[key]) for key in sorted(value)}
    return value
