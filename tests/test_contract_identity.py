from __future__ import annotations

import json

import pytest

from safety_kernel.contract.canonical import canonicalize_contract_body
from safety_kernel.contract.errors import (
    ContractHashMismatchError,
    ContractParseError,
)
from safety_kernel.contract.loader import load_contract_text
from safety_kernel.contract.strict_yaml import load_yaml_strict


def _minimal_contract_yaml(body_sha256: str) -> str:
    return f"""
schema_version: "0.1.0"

metadata:
  contract_id: "test_contract"
  cell_id: "cell_test_01"
  created_at: "2026-02-11T00:00:00Z"
  created_by: "integrator:test"
  description: "test"
  robot_compat:
    robot_model: ["cell_arm_5dof"]
  validity:
    valid_from: "2026-02-11T00:00:00Z"
    expires_at: null

frames:
  units: {{length: "m", angle: "rad"}}
  cell_frame: {{name: "cell", convention: "right_handed"}}
  robot_base_in_cell:
    xyz: [0.0, 0.0, 0.0]
    rpy: [0.0, 0.0, 0.0]
    tolerance: {{pos_m: 0.005, rot_rad: 0.01}}

geometry:
  keepouts:
    - name: "b_keepout"
      type: "box"
      pose_in_cell: {{xyz: [0.6, 0.35, 0.3], rpy: [0.0, 0.0, 0.0]}}
      dims: {{x: 0.2, y: 0.2, z: 0.6}}
      tolerance: {{inflate_m: 0.02}}
      enforcement: {{hard_keepout: true}}
    - name: "a_keepout"
      type: "box"
      pose_in_cell: {{xyz: [0.4, 0.0, 0.5], rpy: [0.0, 0.0, 0.0]}}
      dims: {{x: 0.1, y: 0.8, z: 1.0}}
      tolerance: {{inflate_m: 0.02}}
      enforcement: {{hard_keepout: true}}
  allowed_regions:
    - name: "work_envelope"
      type: "box"
      pose_in_cell: {{xyz: [0.4, 0.0, 0.5], rpy: [0.0, 0.0, 0.0]}}
      dims: {{x: 0.8, y: 0.8, z: 1.0}}
      tolerance: {{deflate_m: 0.02}}
      enforcement:
        hard_keepout: false
        caps_when_active: {{max_tcp_speed_mps: 0.6}}

anchors:
  - name: "tag_B"
    type: "fiducial"
    sensor: "camera_front"
    expected_pose_in_cell: {{xyz: [0.2, 0.1, 1.0], rpy: [0.0, 0.0, 0.0]}}
    tolerance: {{pos_m: 0.01, rot_rad: 0.03}}
    required: true
  - name: "tag_A"
    type: "fiducial"
    sensor: "camera_front"
    expected_pose_in_cell: {{xyz: [1.5, 0.1, 1.0], rpy: [0.0, 0.0, 0.0]}}
    tolerance: {{pos_m: 0.01, rot_rad: 0.03}}
    required: true

assumption_monitors:
  - name: "anchors_consistent"
    type: "anchor_pose_consistency"
    inputs: ["camera_front"]
    params:
      anchors: ["tag_B", "tag_A"]
      aggregation: "any_required_out_of_tolerance"
    debounce_ms: 250
    hysteresis_ms: 500
    severity: "revoke"
    action: "revoke_L0C_enable_L1_full"

revocation_policy:
  on_degrade:
    l0c_still_enforced: true
    caps: {{max_tcp_speed_mps: 0.2, restrict_to_allowed_regions: true}}
  on_revoke:
    l0c_enabled: false
    l1_mode: "full_conservative"
  on_rta_stop:
    sequence: ["controlled_stop", "retreat_to_home", "operator_handoff"]

integrity:
  canonicalization:
    sort_keys: true
    float_quantization: 1.0e-6
    normalize_newlines: true
    unit_normalization: true
  body_sha256: "{body_sha256}"
  signature: null
  signer_id: "integrator:test"
  signed_at: null
""".lstrip()


def test_strict_yaml_rejects_duplicate_keys():
    bad = "a: 1\\na: 2\\n"
    with pytest.raises(ContractParseError):
        load_yaml_strict(bad)


def test_strict_yaml_rejects_anchors_and_aliases():
    bad = "a: &x 1\\nb: *x\\n"
    with pytest.raises(ContractParseError):
        load_yaml_strict(bad)


def test_strict_yaml_rejects_explicit_tags():
    bad = 'a: !!int "1"\\n'
    with pytest.raises(ContractParseError):
        load_yaml_strict(bad)


def test_strict_yaml_numeric_and_bool_semantics():
    obj = load_yaml_strict("x: 012\ny: yes\nz: true\n")
    assert str(obj["x"]) == "12"
    assert obj["y"] == "yes"
    assert obj["z"] is True


def test_contract_hash_verification_pass_and_fail():
    base = _minimal_contract_yaml("0" * 64)
    raw = load_yaml_strict(base)
    canon = canonicalize_contract_body(raw)
    good = _minimal_contract_yaml(canon.body_sha256)

    loaded = load_contract_text(good)
    assert loaded.body_sha256 == canon.body_sha256

    with pytest.raises(ContractHashMismatchError):
        load_contract_text(base)


def test_canonicalization_sorts_sets_and_quantizes_numbers():
    base = _minimal_contract_yaml("0" * 64)
    raw = load_yaml_strict(base)
    canon = canonicalize_contract_body(raw)
    obj = json.loads(canon.canonical_bytes.decode("utf-8"))

    # keepouts sorted by name
    assert [k["name"] for k in obj["geometry"]["keepouts"]] == ["a_keepout", "b_keepout"]
    # anchors sorted by name
    assert [a["name"] for a in obj["anchors"]] == ["tag_A", "tag_B"]
    # monitor inputs + params.anchors sorted
    m = obj["assumption_monitors"][0]
    assert m["inputs"] == ["camera_front"]
    assert m["params"]["anchors"] == ["tag_A", "tag_B"]
    # integer-like fields canonicalized as fixed-point strings
    assert m["debounce_ms"] == "250.000000"
    # float quantization boundary
    assert obj["frames"]["robot_base_in_cell"]["tolerance"]["pos_m"] == "0.005000"
