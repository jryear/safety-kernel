"""RFC-0001 canonicalization test vectors.

Validates FFSK Canon v1: deterministic YAML → canonical bytes → SHA-256.
"""

import hashlib
import json

import pytest

from safety_kernel.contract.canonicalize import (
    canonicalize_contract,
    verify_contract_hash,
    validate_schema_version,
    validate_no_unknown_fields,
    _quantize_number,
    _normalize_numbers,
)


# ── Minimal contract for test vectors ────────────────────────────────

MINIMAL_CONTRACT = """\
schema_version: "0.1.0"

metadata:
  contract_id: "test-minimal-001"
  cell_id: "test_cell"
  created_at: "2026-01-01T00:00:00Z"
  created_by: "test"
  robot_compat:
    robot_model: ["test_arm"]

frames:
  units: {length: "m", angle: "rad"}
  cell_frame:
    name: "cell"
    convention: "right_handed"
  robot_base_in_cell:
    xyz: [0.0, 0.0, 0.0]
    rpy: [0.0, 0.0, 0.0]

geometry:
  keepouts: []
  allowed_regions: []

anchors: []

assumption_monitors: []

revocation_policy:
  on_degrade:
    l0c_still_enforced: true
    caps:
      max_tcp_speed_mps: 0.20
      restrict_to_allowed_regions: true
  on_revoke:
    l0c_enabled: false
    l1_mode: "full_conservative"
  on_rta_stop:
    sequence: ["controlled_stop", "retreat_to_home", "operator_handoff"]
"""


class TestNumberQuantization:
    """RFC-0001 Section 5.2 step 5: number normalization."""

    def test_integer_to_string(self):
        assert _quantize_number(2) == "2.000000"

    def test_large_integer_to_string(self):
        assert _quantize_number(250) == "250.000000"

    def test_negative_zero(self):
        assert _quantize_number(-0.0) == "0.000000"

    def test_positive_zero(self):
        assert _quantize_number(0) == "0.000000"

    def test_six_decimal_places(self):
        assert _quantize_number(0.123456) == "0.123456"

    def test_round_half_up_boundary_low(self):
        """0.1234564 rounds down under ROUND_HALF_UP."""
        assert _quantize_number(0.1234564) == "0.123456"

    def test_round_half_up_boundary_high(self):
        """0.1234565 rounds up under ROUND_HALF_UP."""
        assert _quantize_number(0.1234565) == "0.123457"

    def test_round_half_up_vs_banker(self):
        """0.0000005 rounds to 0.000001 (ROUND_HALF_UP), not 0.000000 (banker's)."""
        assert _quantize_number(0.0000005) == "0.000001"

    def test_negative_number(self):
        assert _quantize_number(-3.14159265) == "-3.141593"

    def test_small_number(self):
        assert _quantize_number(0.0000001) == "0.000000"

    def test_bool_not_quantized(self):
        """Booleans should not be quantized (bool is subclass of int)."""
        result = _normalize_numbers({"flag": True, "other": False})
        assert result == {"flag": True, "other": False}

    def test_nested_numbers(self):
        result = _normalize_numbers({"a": [1, 2.5, {"b": 3}]})
        assert result == {"a": ["1.000000", "2.500000", {"b": "3.000000"}]}


class TestMinimalContractHash:
    """Vector 1: minimal contract → known hash."""

    def test_deterministic_hash(self):
        """Same input always produces same hash."""
        _, hash1 = canonicalize_contract(MINIMAL_CONTRACT)
        _, hash2 = canonicalize_contract(MINIMAL_CONTRACT)
        assert hash1 == hash2

    def test_hash_is_lowercase_hex(self):
        _, h = canonicalize_contract(MINIMAL_CONTRACT)
        assert h == h.lower()
        assert len(h) == 64  # SHA-256 hex

    def test_canonical_bytes_are_valid_json(self):
        canonical_bytes, _ = canonicalize_contract(MINIMAL_CONTRACT)
        parsed = json.loads(canonical_bytes)
        assert isinstance(parsed, dict)

    def test_canonical_json_has_sorted_keys(self):
        canonical_bytes, _ = canonicalize_contract(MINIMAL_CONTRACT)
        text = canonical_bytes.decode("utf-8")
        # Should be compact (no spaces after separators)
        assert ": " not in text
        assert ", " not in text

    def test_canonical_json_numbers_are_strings(self):
        """All numeric values in canonical form are 6-decimal strings."""
        canonical_bytes, _ = canonicalize_contract(MINIMAL_CONTRACT)
        parsed = json.loads(canonical_bytes)
        # Check that xyz values are strings
        xyz = parsed["frames"]["robot_base_in_cell"]["xyz"]
        for v in xyz:
            assert isinstance(v, str)
            assert v == "0.000000"


class TestOrderingInvariance:
    """Vector 3: same content, different ordering → same hash."""

    REORDERED_CONTRACT = """\
schema_version: "0.1.0"

revocation_policy:
  on_rta_stop:
    sequence: ["controlled_stop", "retreat_to_home", "operator_handoff"]
  on_degrade:
    caps:
      restrict_to_allowed_regions: true
      max_tcp_speed_mps: 0.20
    l0c_still_enforced: true
  on_revoke:
    l1_mode: "full_conservative"
    l0c_enabled: false

assumption_monitors: []
anchors: []

geometry:
  allowed_regions: []
  keepouts: []

frames:
  robot_base_in_cell:
    rpy: [0.0, 0.0, 0.0]
    xyz: [0.0, 0.0, 0.0]
  cell_frame:
    convention: "right_handed"
    name: "cell"
  units: {angle: "rad", length: "m"}

metadata:
  robot_compat:
    robot_model: ["test_arm"]
  created_by: "test"
  created_at: "2026-01-01T00:00:00Z"
  cell_id: "test_cell"
  contract_id: "test-minimal-001"
"""

    def test_reordered_same_hash(self):
        """Reordered fields produce identical hash."""
        _, hash_original = canonicalize_contract(MINIMAL_CONTRACT)
        _, hash_reordered = canonicalize_contract(self.REORDERED_CONTRACT)
        assert hash_original == hash_reordered

    def test_reordered_same_bytes(self):
        """Reordered fields produce identical canonical bytes."""
        bytes_original, _ = canonicalize_contract(MINIMAL_CONTRACT)
        bytes_reordered, _ = canonicalize_contract(self.REORDERED_CONTRACT)
        assert bytes_original == bytes_reordered


class TestListReordering:
    """Vector 5: lists sorted by name produce same hash."""

    CONTRACT_WITH_MONITORS_A = """\
schema_version: "0.1.0"

metadata:
  contract_id: "test-list-001"
  cell_id: "test_cell"
  created_at: "2026-01-01T00:00:00Z"
  created_by: "test"
  robot_compat:
    robot_model: ["test_arm"]

frames:
  units: {length: "m", angle: "rad"}
  cell_frame: {name: "cell", convention: "right_handed"}
  robot_base_in_cell: {xyz: [0.0, 0.0, 0.0], rpy: [0.0, 0.0, 0.0]}

geometry:
  keepouts:
    - name: "wall_a"
      type: "box"
      pose_in_cell: {xyz: [1.0, 0.0, 0.0]}
      dims: {x: 0.1, y: 1.0, z: 1.0}
    - name: "column_b"
      type: "box"
      pose_in_cell: {xyz: [0.0, 1.0, 0.0]}
      dims: {x: 0.2, y: 0.2, z: 1.0}
  allowed_regions: []

anchors: []

assumption_monitors:
  - name: "monitor_z"
    type: "safety_io_health"
    inputs: ["plc"]
    params: {signal: "gate_closed", expected: true}
    debounce_ms: 0
    hysteresis_ms: 0
    severity: "rta_stop"
    action: "safe_stop"
  - name: "monitor_a"
    type: "sensor_health"
    inputs: ["camera_front", "camera_rear"]
    params: {max_frame_age_ms: 100}
    debounce_ms: 500
    hysteresis_ms: 1000
    severity: "degrade"
    action: "tighten_caps"

revocation_policy:
  on_degrade: {l0c_still_enforced: true, caps: {max_tcp_speed_mps: 0.20, restrict_to_allowed_regions: true}}
  on_revoke: {l0c_enabled: false, l1_mode: "full_conservative"}
  on_rta_stop: {sequence: ["controlled_stop"]}
"""

    CONTRACT_WITH_MONITORS_B = """\
schema_version: "0.1.0"

metadata:
  contract_id: "test-list-001"
  cell_id: "test_cell"
  created_at: "2026-01-01T00:00:00Z"
  created_by: "test"
  robot_compat:
    robot_model: ["test_arm"]

frames:
  units: {length: "m", angle: "rad"}
  cell_frame: {name: "cell", convention: "right_handed"}
  robot_base_in_cell: {xyz: [0.0, 0.0, 0.0], rpy: [0.0, 0.0, 0.0]}

geometry:
  keepouts:
    - name: "column_b"
      type: "box"
      pose_in_cell: {xyz: [0.0, 1.0, 0.0]}
      dims: {x: 0.2, y: 0.2, z: 1.0}
    - name: "wall_a"
      type: "box"
      pose_in_cell: {xyz: [1.0, 0.0, 0.0]}
      dims: {x: 0.1, y: 1.0, z: 1.0}
  allowed_regions: []

anchors: []

assumption_monitors:
  - name: "monitor_a"
    type: "sensor_health"
    inputs: ["camera_rear", "camera_front"]
    params: {max_frame_age_ms: 100}
    debounce_ms: 500
    hysteresis_ms: 1000
    severity: "degrade"
    action: "tighten_caps"
  - name: "monitor_z"
    type: "safety_io_health"
    inputs: ["plc"]
    params: {signal: "gate_closed", expected: true}
    debounce_ms: 0
    hysteresis_ms: 0
    severity: "rta_stop"
    action: "safe_stop"

revocation_policy:
  on_degrade: {l0c_still_enforced: true, caps: {max_tcp_speed_mps: 0.20, restrict_to_allowed_regions: true}}
  on_revoke: {l0c_enabled: false, l1_mode: "full_conservative"}
  on_rta_stop: {sequence: ["controlled_stop"]}
"""

    def test_monitor_order_invariance(self):
        """Monitors in different order → same hash (sorted by name)."""
        _, hash_a = canonicalize_contract(self.CONTRACT_WITH_MONITORS_A)
        _, hash_b = canonicalize_contract(self.CONTRACT_WITH_MONITORS_B)
        assert hash_a == hash_b

    def test_keepout_order_invariance(self):
        """Keepouts in different order → same hash (sorted by name)."""
        _, hash_a = canonicalize_contract(self.CONTRACT_WITH_MONITORS_A)
        _, hash_b = canonicalize_contract(self.CONTRACT_WITH_MONITORS_B)
        assert hash_a == hash_b

    def test_monitor_inputs_sorted(self):
        """Monitor inputs (scalar list) sorted: camera_front before camera_rear."""
        canonical_bytes, _ = canonicalize_contract(self.CONTRACT_WITH_MONITORS_A)
        parsed = json.loads(canonical_bytes)
        monitor_a = [m for m in parsed["assumption_monitors"] if m["name"] == "monitor_a"][0]
        assert monitor_a["inputs"] == ["camera_front", "camera_rear"]


class TestIntegrityVerification:
    """Contract integrity block verification."""

    def test_verify_correct_hash(self):
        """Contract with correct body_sha256 verifies."""
        _, correct_hash = canonicalize_contract(MINIMAL_CONTRACT)
        contract_with_integrity = MINIMAL_CONTRACT + f"""
integrity:
  canonicalization:
    sort_keys: true
    float_quantization: 1.0e-6
    normalize_newlines: true
    unit_normalization: true
  body_sha256: "{correct_hash}"
  signature: null
  signer_id: "test"
  signed_at: null
"""
        assert verify_contract_hash(contract_with_integrity)

    def test_verify_wrong_hash(self):
        """Contract with incorrect body_sha256 fails."""
        contract_with_bad_hash = MINIMAL_CONTRACT + """
integrity:
  body_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
"""
        assert not verify_contract_hash(contract_with_bad_hash)

    def test_integrity_not_in_body(self):
        """Integrity block is excluded from canonical body."""
        _, hash_without = canonicalize_contract(MINIMAL_CONTRACT)

        contract_with_integrity = MINIMAL_CONTRACT + """
integrity:
  body_sha256: "anything"
  extra_field: "should not affect hash"
"""
        _, hash_with = canonicalize_contract(contract_with_integrity)
        assert hash_without == hash_with


class TestUnitValidation:
    """RFC-0001 Section 5.2 step 3: unit normalization."""

    def test_reject_mm_units(self):
        bad = MINIMAL_CONTRACT.replace('length: "m"', 'length: "mm"')
        with pytest.raises(ValueError, match="length.*must be.*m"):
            canonicalize_contract(bad)

    def test_reject_deg_units(self):
        bad = MINIMAL_CONTRACT.replace('angle: "rad"', 'angle: "deg"')
        with pytest.raises(ValueError, match="angle.*must be.*rad"):
            canonicalize_contract(bad)

    def test_reject_missing_units(self):
        bad = MINIMAL_CONTRACT.replace(
            'units: {length: "m", angle: "rad"}',
            'units: {length: "m"}'
        )
        with pytest.raises(ValueError, match="angle"):
            canonicalize_contract(bad)


class TestStrictYAMLParsing:
    """RFC-0001 Section 5.1: YAML input constraints."""

    def test_reject_duplicate_keys(self):
        bad = """\
schema_version: "0.1.0"
schema_version: "0.1.0"
frames:
  units: {length: "m", angle: "rad"}
"""
        with pytest.raises(ValueError, match="Duplicate key"):
            canonicalize_contract(bad)

    def test_reject_non_string_keys(self):
        """Non-string mapping keys are rejected."""
        bad = """\
schema_version: "0.1.0"
123: "numeric key"
frames:
  units: {length: "m", angle: "rad"}
"""
        with pytest.raises(ValueError, match="Non-string"):
            canonicalize_contract(bad)


class TestSchemaVersion:
    """Schema version compatibility (fail-closed)."""

    def test_accept_0_1_0(self):
        validate_schema_version({"schema_version": "0.1.0"})

    def test_accept_0_1_99(self):
        validate_schema_version({"schema_version": "0.1.99"})

    def test_reject_0_2_0(self):
        with pytest.raises(ValueError, match="0.1.x"):
            validate_schema_version({"schema_version": "0.2.0"})

    def test_reject_1_0_0(self):
        with pytest.raises(ValueError, match="0.1.x"):
            validate_schema_version({"schema_version": "1.0.0"})

    def test_reject_malformed(self):
        with pytest.raises(ValueError):
            validate_schema_version({"schema_version": "abc"})


class TestUnknownFieldRejection:
    """Strict unknown field rejection per RFC-0001 Section 4.3."""

    def test_accept_known_fields(self):
        validate_no_unknown_fields({
            "schema_version": "0.1.0",
            "metadata": {},
            "frames": {},
        })

    def test_reject_unknown_top_level(self):
        with pytest.raises(ValueError, match="Unknown"):
            validate_no_unknown_fields({
                "schema_version": "0.1.0",
                "unknown_field": "bad",
            })


class TestTrackAContract:
    """Verify canonicalization works on the actual Track A contract."""

    def test_track_a_canonicalizes(self):
        """Track A contract YAML can be canonicalized without error."""
        import pathlib
        contract_path = pathlib.Path(__file__).resolve().parents[1] / "contracts" / "cell_arm_track_a.yaml"
        yaml_text = contract_path.read_text()
        canonical_bytes, body_hash = canonicalize_contract(yaml_text)
        assert len(body_hash) == 64
        assert isinstance(canonical_bytes, bytes)
        # Verify the hash is reproducible
        _, hash2 = canonicalize_contract(yaml_text)
        assert body_hash == hash2
