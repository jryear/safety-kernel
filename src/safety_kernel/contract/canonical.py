from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence

from safety_kernel.contract.errors import ContractValidationError


_QUANT = Decimal("0.000001")


@dataclass(frozen=True)
class CanonicalBody:
    canonical_obj: Any
    canonical_bytes: bytes
    body_sha256: str


def _normalize_string(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", value)


def _quantize_decimal(value: Decimal) -> str:
    q = value.quantize(_QUANT, rounding=ROUND_HALF_UP)
    if q == 0:
        q = q.copy_abs()
    return format(q, "f")


def _walk(obj: Any, visit: Callable[[Any], Any]) -> Any:
    """Recursively apply visit() to scalars and rebuild containers."""
    obj = visit(obj)
    if isinstance(obj, dict):
        return {k: _walk(v, visit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, visit) for v in obj]
    return obj


def _sort_list_of_dicts_by_name(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    def key_fn(item: Mapping[str, Any]) -> str:
        name = item.get("name")
        if not isinstance(name, str):
            raise ContractValidationError("Expected 'name' field to be a string for sortable lists")
        return name

    return sorted(items, key=key_fn)


def _sort_scalar_list(values: Sequence[Any]) -> list[Any]:
    for v in values:
        if not isinstance(v, str):
            raise ContractValidationError("Expected list elements to be strings for sortable scalar sets")
    return sorted(values)


def canonicalize_contract_body(contract: Mapping[str, Any]) -> CanonicalBody:
    """Canonicalize a full contract mapping (including integrity), returning body bytes and hash.

    Canonicalization profile: RFC-0001 (FFSK Canon v1).
    """
    if not isinstance(contract, Mapping):
        raise ContractValidationError("Contract must be a mapping")

    # Extract body (remove integrity)
    body: dict[str, Any] = dict(contract)
    body.pop("integrity", None)

    # Units normalization: required and must match exactly
    try:
        frames = body["frames"]
        units = frames["units"]
        length_unit = units["length"]
        angle_unit = units["angle"]
    except Exception as exc:  # pragma: no cover
        raise ContractValidationError("Missing required frames.units") from exc

    if length_unit != "m" or angle_unit != "rad":
        raise ContractValidationError("Unsupported units; expected frames.units.length='m' and angle='rad'")

    # Normalize strings (NFC + newlines)
    def normalize_scalars(value: Any) -> Any:
        if isinstance(value, str):
            return _normalize_string(value)
        return value

    normalized = _walk(body, normalize_scalars)

    # Quantize numbers and convert to fixed-point strings
    def normalize_numbers(value: Any) -> Any:
        if isinstance(value, Decimal):
            return _quantize_decimal(value)
        if isinstance(value, float):  # pragma: no cover
            raise ContractValidationError("Binary floats are not allowed in parsed contract representation")
        return value

    normalized = _walk(normalized, normalize_numbers)

    # Sort semantic sets
    normalized = _apply_semantic_sorts(normalized)

    canonical_json = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    canonical_bytes = canonical_json.encode("utf-8")
    body_sha256 = hashlib.sha256(canonical_bytes).hexdigest()

    return CanonicalBody(canonical_obj=normalized, canonical_bytes=canonical_bytes, body_sha256=body_sha256)


def _apply_semantic_sorts(body_obj: Any) -> Any:
    """Apply RFC-0001 list ordering normalizations.

    Operates on the post-normalized object where:
    - mapping keys are strings
    - all numbers are strings
    """
    if not isinstance(body_obj, dict):
        return body_obj

    # Apply only the RFC-defined semantic sorts at their canonical paths.
    geometry = body_obj.get("geometry")
    if isinstance(geometry, dict):
        keepouts = geometry.get("keepouts")
        if isinstance(keepouts, list):
            geometry["keepouts"] = _sort_list_of_dicts_by_name(_expect_list_of_dicts(keepouts))

        allowed = geometry.get("allowed_regions")
        if isinstance(allowed, list):
            geometry["allowed_regions"] = _sort_list_of_dicts_by_name(_expect_list_of_dicts(allowed))

    top_anchors = body_obj.get("anchors")
    if isinstance(top_anchors, list):
        body_obj["anchors"] = _sort_list_of_dicts_by_name(_expect_list_of_dicts(top_anchors))

    monitors = body_obj.get("assumption_monitors")
    if isinstance(monitors, list):
        sorted_monitors = _sort_list_of_dicts_by_name(_expect_list_of_dicts(monitors))
        for m in sorted_monitors:
            inputs = m.get("inputs")
            if isinstance(inputs, list):
                m["inputs"] = _sort_scalar_list(inputs)
            params = m.get("params")
            if isinstance(params, dict):
                anchors_list = params.get("anchors")
                if isinstance(anchors_list, list):
                    params["anchors"] = _sort_scalar_list(anchors_list)
        body_obj["assumption_monitors"] = sorted_monitors

    metadata = body_obj.get("metadata")
    if isinstance(metadata, dict):
        robot_compat = metadata.get("robot_compat")
        if isinstance(robot_compat, dict):
            robot_model = robot_compat.get("robot_model")
            if isinstance(robot_model, list):
                robot_compat["robot_model"] = _sort_scalar_list(robot_model)

    return body_obj


def _expect_list_of_dicts(items: Sequence[Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ContractValidationError("Expected a list of mappings")
        out.append(item)
    return out
