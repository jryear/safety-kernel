"""Contract loader tests.

Validates: load, hash verification, schema rejection, geometry extraction.
"""

import pathlib

import numpy as np
import pytest

from safety_kernel.contract.loader import load_contract_file, load_contract_text
from safety_kernel.contract.errors import (
    ContractHashMismatchError,
    ContractValidationError,
    ContractParseError,
)

_CONTRACTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "contracts"
_TRACK_A = _CONTRACTS_DIR / "cell_arm_track_a.yaml"


class TestLoadTrackAContract:
    """Load the actual Track A contract YAML."""

    def test_load_succeeds(self):
        lc = load_contract_file(_TRACK_A)
        assert lc.schema_version == "0.1.0"
        assert len(lc.body_sha256) == 64

    def test_contract_id(self):
        lc = load_contract_file(_TRACK_A)
        assert lc.contract.contract_id == "cell-arm-track-a-001"

    def test_keepouts_extracted(self):
        lc = load_contract_file(_TRACK_A)
        c = lc.contract
        assert len(c.keepouts) == 2

        fixture = [k for k in c.keepouts if k.name == "fixture_column"][0]
        np.testing.assert_allclose(fixture.center, [0.6, 0.35, 0.3])
        np.testing.assert_allclose(fixture.half_dims, [0.1, 0.1, 0.3])
        assert fixture.inflate_m == pytest.approx(0.02)
        assert fixture.hard_keepout is True

        wall = [k for k in c.keepouts if k.name == "back_wall"][0]
        np.testing.assert_allclose(wall.center, [-0.3, 0.0, 0.5])
        np.testing.assert_allclose(wall.half_dims, [0.05, 0.4, 0.5])

    def test_keepouts_match_hardcoded(self):
        """Keepouts from contract must match hardcoded values in cell_arm.py."""
        lc = load_contract_file(_TRACK_A)
        c = lc.contract

        # These must match _KEEPOUTS in cell_arm.py
        fixture = [k for k in c.keepouts if k.name == "fixture_column"][0]
        np.testing.assert_allclose(fixture.center, [0.6, 0.35, 0.3])
        np.testing.assert_allclose(fixture.half_dims, [0.1, 0.1, 0.3])

        wall = [k for k in c.keepouts if k.name == "back_wall"][0]
        np.testing.assert_allclose(wall.center, [-0.3, 0.0, 0.5])
        np.testing.assert_allclose(wall.half_dims, [0.05, 0.4, 0.5])

    def test_allowed_regions(self):
        lc = load_contract_file(_TRACK_A)
        c = lc.contract
        assert len(c.allowed_regions) == 1
        wr = c.allowed_regions[0]
        assert wr.name == "work_envelope"
        np.testing.assert_allclose(wr.center, [0.4, 0.0, 0.5])
        np.testing.assert_allclose(wr.half_dims, [0.4, 0.4, 0.5])
        assert wr.max_tcp_speed_mps == pytest.approx(0.6)

    def test_anchors(self):
        lc = load_contract_file(_TRACK_A)
        c = lc.contract
        assert len(c.anchors) == 2
        for a in c.anchors:
            assert a.required is True
            assert a.pos_tol == pytest.approx(0.01)
            assert a.rot_tol == pytest.approx(0.03)

    def test_monitors(self):
        lc = load_contract_file(_TRACK_A)
        c = lc.contract
        assert len(c.monitors) == 5
        names = {m.name for m in c.monitors}
        assert names == {
            "anchors_consistent",
            "gate_interlock_ok",
            "estop_chain_ok",
            "front_cam_health",
            "robot_mount_integrity",
        }

    def test_revocation_policy(self):
        lc = load_contract_file(_TRACK_A)
        c = lc.contract
        assert c.revocation_policy.on_degrade_max_tcp_speed_mps == pytest.approx(0.20)
        assert c.revocation_policy.on_degrade_restrict_to_allowed is True
        assert c.revocation_policy.on_revoke_l0c_enabled is False
        assert c.revocation_policy.on_revoke_l1_mode == "full_conservative"

    def test_hash_reproducible(self):
        lc1 = load_contract_file(_TRACK_A)
        lc2 = load_contract_file(_TRACK_A)
        assert lc1.body_sha256 == lc2.body_sha256


class TestContractRejection:
    """Contracts that should be rejected."""

    def _load_modified(self, modification: str, replacement: str):
        yaml_text = _TRACK_A.read_text()
        modified = yaml_text.replace(modification, replacement)
        return load_contract_text(modified)

    def test_hash_mismatch(self):
        """Modified body with original hash → reject."""
        yaml_text = _TRACK_A.read_text()
        # Change contract_id but keep old hash
        modified = yaml_text.replace("cell-arm-track-a-001", "cell-arm-MODIFIED")
        with pytest.raises(ContractHashMismatchError):
            load_contract_text(modified)

    def test_missing_integrity(self):
        """Contract without integrity block → reject."""
        yaml_text = _TRACK_A.read_text()
        # Remove integrity block (everything from "integrity:" to end)
        lines = yaml_text.split("\n")
        integrity_start = None
        for i, line in enumerate(lines):
            if line.startswith("integrity:"):
                integrity_start = i
                break
        assert integrity_start is not None
        modified = "\n".join(lines[:integrity_start])
        with pytest.raises(ContractValidationError):
            load_contract_text(modified)

    def test_schema_version_0_2(self):
        """Schema 0.2.0 → reject."""
        with pytest.raises(ContractValidationError, match="0.1"):
            self._load_modified('"0.1.0"', '"0.2.0"')

    def test_schema_version_1_0(self):
        """Schema 1.0.0 → reject."""
        with pytest.raises(ContractValidationError, match="0.1"):
            self._load_modified('"0.1.0"', '"1.0.0"')
