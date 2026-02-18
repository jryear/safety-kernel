"""RFC-0001 contract canonicalization and hashing (FFSK Canon v1).

Implements Section 5 of RFC-0001: deterministic canonicalization of
L0-C contract YAML into canonical bytes + SHA-256 body hash.

Canonicalization output uses numbers-as-strings (6 decimal places).
This is separate from runtime schema types — runtime code uses native
floats; canonical form is for identity/hashing only.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any

import yaml


# ── RFC-0001 Section 5.2 normative constants ─────────────────────────

_QUANTIZATION = Decimal("0.000001")  # 1e-6

# Lists sorted by 'name' field (RFC-0001 Section 5.2 step 6)
_NAME_SORTED_PATHS = {
    ("geometry", "keepouts"),
    ("geometry", "allowed_regions"),
    ("anchors",),
    ("assumption_monitors",),
}

# Scalar lists sorted lexicographically
_SCALAR_SORTED_PATHS = {
    ("metadata", "robot_compat", "robot_model"),
}

# Per-item scalar lists inside assumption_monitors
_MONITOR_SCALAR_SORTED_KEYS = {"inputs"}
_MONITOR_PARAMS_SCALAR_SORTED_KEYS = {"anchors"}

# Known top-level keys for schema 0.1.x (strict unknown-field rejection)
_KNOWN_TOP_LEVEL_KEYS = {
    "schema_version", "metadata", "frames", "geometry",
    "anchors", "assumption_monitors", "revocation_policy", "integrity",
}


def canonicalize_contract(yaml_text: str) -> tuple[bytes, str]:
    """Canonicalize a contract YAML and compute body hash.

    Args:
        yaml_text: Raw YAML text of the contract.

    Returns:
        (canonical_body_bytes, body_sha256_hex)

    Raises:
        ValueError: If the contract violates RFC-0001 constraints.
    """
    # Step 1: Parse YAML strict mode
    data = _parse_yaml_strict(yaml_text)

    # Step 2: Extract body (remove integrity)
    body = {k: v for k, v in data.items() if k != "integrity"}

    # Step 3: Validate units
    _validate_units(body)

    # Step 4: Normalize strings (NFC, \n newlines)
    body = _normalize_strings(body)

    # Step 5: Normalize numbers (quantize to 1e-6, ROUND_HALF_UP, 6dp)
    body = _normalize_numbers(body)

    # Step 6: Normalize list ordering
    _normalize_list_ordering(body)

    # Steps 7+8: Serialize as compact JSON with sorted keys
    canonical_json = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    canonical_bytes = canonical_json.encode("utf-8")

    # Step 9: SHA-256 hash
    body_hash = hashlib.sha256(canonical_bytes).hexdigest().lower()

    return canonical_bytes, body_hash


def verify_contract_hash(yaml_text: str) -> bool:
    """Recompute body hash and compare to declared integrity.body_sha256.

    Returns True if match, False if mismatch or integrity block missing.
    """
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        return False

    integrity = data.get("integrity")
    if not isinstance(integrity, dict):
        return False

    declared = integrity.get("body_sha256")
    if not declared:
        return False

    _, computed = canonicalize_contract(yaml_text)
    return declared == computed


# ── Step 1: Strict YAML parsing ──────────────────────────────────────


class _StrictYAMLLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys."""
    pass


def _strict_duplicate_key_check(loader, node, deep=False):
    """Raise on duplicate keys in YAML mappings."""
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError(
                f"Non-string mapping key: {key!r} (type {type(key).__name__})"
            )
        if key in mapping:
            raise ValueError(f"Duplicate key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictYAMLLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _strict_duplicate_key_check,
)


def _parse_yaml_strict(yaml_text: str) -> dict:
    """Parse YAML with strict mode: reject duplicate keys, non-string keys."""
    data = yaml.load(yaml_text, Loader=_StrictYAMLLoader)
    if not isinstance(data, dict):
        raise ValueError("Contract must be a YAML mapping at top level")

    # Reject NaN/Infinity
    _reject_special_floats(data)

    return data


def _reject_special_floats(obj: Any) -> None:
    """Reject NaN and Infinity float values recursively."""
    if isinstance(obj, float):
        if obj != obj:  # NaN
            raise ValueError("NaN values not allowed in contract")
        if abs(obj) == float("inf"):
            raise ValueError("Infinity values not allowed in contract")
    elif isinstance(obj, dict):
        for v in obj.values():
            _reject_special_floats(v)
    elif isinstance(obj, list):
        for item in obj:
            _reject_special_floats(item)


# ── Step 3: Unit validation ──────────────────────────────────────────


def _validate_units(body: dict) -> None:
    """Validate frames.units: length must be 'm', angle must be 'rad'."""
    frames = body.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("Missing 'frames' in contract body")

    units = frames.get("units")
    if not isinstance(units, dict):
        raise ValueError("Missing 'frames.units' in contract body")

    if units.get("length") != "m":
        raise ValueError(
            f"frames.units.length must be 'm', got {units.get('length')!r}"
        )
    if units.get("angle") != "rad":
        raise ValueError(
            f"frames.units.angle must be 'rad', got {units.get('angle')!r}"
        )


# ── Step 4: String normalization ─────────────────────────────────────


def _normalize_strings(obj: Any) -> Any:
    """Normalize strings: \\n newlines, Unicode NFC."""
    if isinstance(obj, str):
        # Normalize newlines to \n
        s = obj.replace("\r\n", "\n").replace("\r", "\n")
        # Unicode NFC normalization
        s = unicodedata.normalize("NFC", s)
        return s
    elif isinstance(obj, dict):
        return {
            _normalize_strings(k): _normalize_strings(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_normalize_strings(item) for item in obj]
    else:
        return obj


# ── Step 5: Number normalization ─────────────────────────────────────


def _normalize_numbers(obj: Any) -> Any:
    """Quantize all numeric values to 1e-6 using ROUND_HALF_UP.

    All numbers become strings with exactly 6 decimal places:
      2       → "2.000000"
      250     → "250.000000"
      -0.0    → "0.000000"
      0.12346 → "0.123460"
    """
    if isinstance(obj, bool):
        # bool is a subclass of int in Python — preserve as bool
        return obj
    elif isinstance(obj, (int, float)):
        return _quantize_number(obj)
    elif isinstance(obj, dict):
        return {k: _normalize_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_numbers(item) for item in obj]
    else:
        return obj


def _quantize_number(value: int | float) -> str:
    """Quantize a single number to 6 decimal places with ROUND_HALF_UP."""
    try:
        dec = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"Cannot quantize value: {value!r}")

    # Handle -0.0
    if dec == 0:
        dec = Decimal("0")

    quantized = dec.quantize(_QUANTIZATION, rounding=ROUND_HALF_UP)
    # Format with exactly 6 decimal places
    return f"{quantized}"


# ── Step 6: List ordering normalization ──────────────────────────────


def _normalize_list_ordering(body: dict) -> None:
    """Sort semantic-set lists per RFC-0001 Section 5.2 step 6."""

    # Sort lists by 'name' field
    for path in _NAME_SORTED_PATHS:
        container = body
        for key in path[:-1]:
            if isinstance(container, dict) and key in container:
                container = container[key]
            else:
                container = None
                break
        if container is None:
            continue

        last_key = path[-1]
        if isinstance(container, dict) and last_key in container:
            lst = container[last_key]
            if isinstance(lst, list):
                lst.sort(key=lambda x: x.get("name", "") if isinstance(x, dict) else "")

    # Sort scalar lists
    for path in _SCALAR_SORTED_PATHS:
        container = body
        for key in path[:-1]:
            if isinstance(container, dict) and key in container:
                container = container[key]
            else:
                container = None
                break
        if container is None:
            continue

        last_key = path[-1]
        if isinstance(container, dict) and last_key in container:
            lst = container[last_key]
            if isinstance(lst, list):
                lst.sort()

    # Sort per-monitor scalar lists (inputs, params.anchors)
    monitors = body.get("assumption_monitors", [])
    if isinstance(monitors, list):
        for monitor in monitors:
            if not isinstance(monitor, dict):
                continue
            for key in _MONITOR_SCALAR_SORTED_KEYS:
                if key in monitor and isinstance(monitor[key], list):
                    monitor[key].sort()
            params = monitor.get("params")
            if isinstance(params, dict):
                for key in _MONITOR_PARAMS_SCALAR_SORTED_KEYS:
                    if key in params and isinstance(params[key], list):
                        params[key].sort()


# ── Schema validation helpers ────────────────────────────────────────


def validate_schema_version(data: dict) -> None:
    """Validate schema_version is 0.1.x (fail-closed)."""
    version = data.get("schema_version", "")
    if not isinstance(version, str):
        raise ValueError(f"schema_version must be a string, got {type(version).__name__}")

    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"schema_version must be MAJOR.MINOR.PATCH, got {version!r}")

    try:
        major, minor, _patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError(f"schema_version components must be integers, got {version!r}")

    if major != 0 or minor != 1:
        raise ValueError(
            f"Unsupported schema_version {version!r}: kernel supports 0.1.x only"
        )


def validate_no_unknown_fields(data: dict) -> None:
    """Reject unknown top-level fields outside metadata.extensions."""
    for key in data:
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            raise ValueError(f"Unknown top-level field: {key!r}")
