"""Append-only producer bundles and post-production immutable finalization."""

from __future__ import annotations

import binascii
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from safety_kernel.telemetry.canonical import canonical_json_bytes, event_bytes, event_hash, prepare_event
from safety_kernel.telemetry.errors import BundleImmutableError, TelemetryError
from safety_kernel.telemetry.schema import validate_event
from safety_kernel.contract.loader import load_contract_text

_HASH = re.compile(r"^[0-9a-f]{64}$")


class RunBundle:
    """A new producer bundle; any existing directory is permanently immutable."""

    def __init__(self, path: str | Path, *, manifest: Mapping[str, Any], contract: Mapping[str, Any], environment: Mapping[str, Any]):
        try:
            _contract_identity(contract)
        except ValueError as exc:
            raise TelemetryError("contract component lacks a verifiable RFC-0001 envelope") from exc
        self.path = Path(path)
        if self.path.exists():
            raise BundleImmutableError(f"refusing to overwrite bundle: {self.path}")
        self.path.mkdir(parents=True)
        self._write_once("manifest.json", canonical_json_bytes(dict(manifest)))
        self._write_once("contract.json", canonical_json_bytes(dict(contract)))
        self._write_once("environment.json", canonical_json_bytes(dict(environment)))
        self._events_path = self.path / "events.ndjson"
        self._events_path.touch(exist_ok=False)
        self._previous_hash: str | None = None
        self._event_count = 0
        self._episode_bytes: dict[str, list[bytes]] = {}
        self._sealed = False

    @property
    def manifest_hash(self) -> str:
        return _file_hash(self.path / "manifest.json")

    @property
    def contract_hash(self) -> str:
        try:
            envelope = json.loads(
                (self.path / "contract.json").read_text(encoding="utf-8")
            )
            return _contract_identity(envelope)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise TelemetryError("contract component lacks a verifiable RFC-0001 envelope") from exc

    @property
    def environment_hash(self) -> str:
        return _file_hash(self.path / "environment.json")

    @property
    def previous_hash(self) -> str:
        if self._previous_hash is None:
            raise TelemetryError("producer chain has no events")
        return self._previous_hash

    @property
    def event_count(self) -> int:
        return self._event_count

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if self._sealed:
            raise BundleImmutableError("producer bundle is sealed")
        raw = dict(event)
        if "prev_hash" in raw:
            raise TelemetryError("writer owns prev_hash")
        raw["prev_hash"] = "0" * 64 if self._previous_hash is None else self._previous_hash
        prepared = prepare_event(raw)
        errors = validate_event(prepared, first=self._previous_hash is None)
        if errors:
            raise TelemetryError("invalid event: " + "; ".join(errors))
        line_with_newline = event_bytes(prepared) + b"\n"
        with self._events_path.open("ab") as handle:
            handle.write(line_with_newline)
        episode_id, event_type = prepared.get("episode_id"), prepared["event_type"]
        if event_type == "episode_start" and isinstance(episode_id, str):
            self._episode_bytes[episode_id] = [line_with_newline]
        elif isinstance(episode_id, str) and event_type != "episode_summary" and episode_id in self._episode_bytes:
            self._episode_bytes[episode_id].append(line_with_newline)
        self._previous_hash = event_hash(prepared)
        self._event_count += 1
        return prepared

    def episode_crc32(self, episode_id: str) -> str:
        crc = 0
        for line in self._episode_bytes.get(episode_id, []):
            crc = binascii.crc32(line, crc)
        return f"{crc & 0xffffffff:08x}"

    def seal_producer(self) -> None:
        if self._sealed:
            raise BundleImmutableError("producer bundle is already sealed")
        self._sealed = True

    def _write_once(self, name: str, contents: bytes) -> None:
        target = self.path / name
        if target.exists():
            raise BundleImmutableError(f"refusing to overwrite evidence: {target}")
        _write_once(target, contents)


def _write_once(path: Path, contents: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(contents)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_identity(envelope: Any) -> str:
    """Verify the full source artifact; a detached hash is never identity proof."""
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "contract_envelope_version", "original_yaml"
    }:
        raise ValueError("contract.json must be an RFC-0001 source envelope")
    if envelope.get("contract_envelope_version") != "ffsk.contract-envelope.v1":
        raise ValueError("unsupported contract envelope version")
    yaml_text = envelope.get("original_yaml")
    if not isinstance(yaml_text, str) or not yaml_text:
        raise ValueError("contract envelope original_yaml is required")
    loaded = load_contract_text(yaml_text)
    if not _HASH.fullmatch(loaded.body_sha256):
        raise ValueError("contract loader returned malformed identity")
    return loaded.body_sha256
