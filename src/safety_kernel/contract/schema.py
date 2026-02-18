"""Contract schema dataclasses (runtime types).

These use native Python types (float, not string) — separate from the
canonical form used by canonicalize.py for hashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Pose:
    """3D pose: position + orientation."""
    xyz: np.ndarray  # (3,)
    rpy: np.ndarray  # (3,)

    @classmethod
    def from_dict(cls, d: dict) -> Pose:
        return cls(
            xyz=np.array(d["xyz"], dtype=np.float64),
            rpy=np.array(d.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float64),
        )


@dataclass(frozen=True)
class PoseTolerance:
    """Tolerance for pose comparison."""
    pos_m: float
    rot_rad: float

    @classmethod
    def from_dict(cls, d: dict) -> PoseTolerance:
        return cls(pos_m=float(d["pos_m"]), rot_rad=float(d["rot_rad"]))


@dataclass(frozen=True)
class Keepout:
    """Keepout zone geometry."""
    name: str
    type: str  # "box", "cylinder", "sphere"
    center: np.ndarray  # (3,) pose_in_cell.xyz
    half_dims: np.ndarray  # (3,) for box: dims/2
    inflate_m: float
    hard_keepout: bool

    @classmethod
    def from_dict(cls, d: dict) -> Keepout:
        dims = d["dims"]
        return cls(
            name=d["name"],
            type=d["type"],
            center=np.array(d["pose_in_cell"]["xyz"], dtype=np.float64),
            half_dims=np.array([dims["x"] / 2, dims["y"] / 2, dims["z"] / 2], dtype=np.float64),
            inflate_m=float(d.get("tolerance", {}).get("inflate_m", 0.0)),
            hard_keepout=bool(d.get("enforcement", {}).get("hard_keepout", True)),
        )


@dataclass(frozen=True)
class AllowedRegion:
    """Allowed region geometry."""
    name: str
    type: str
    center: np.ndarray  # (3,)
    half_dims: np.ndarray  # (3,)
    deflate_m: float
    max_tcp_speed_mps: float

    @classmethod
    def from_dict(cls, d: dict) -> AllowedRegion:
        dims = d["dims"]
        caps = d.get("enforcement", {}).get("caps_when_active", {})
        return cls(
            name=d["name"],
            type=d["type"],
            center=np.array(d["pose_in_cell"]["xyz"], dtype=np.float64),
            half_dims=np.array([dims["x"] / 2, dims["y"] / 2, dims["z"] / 2], dtype=np.float64),
            deflate_m=float(d.get("tolerance", {}).get("deflate_m", 0.0)),
            max_tcp_speed_mps=float(caps.get("max_tcp_speed_mps", float("inf"))),
        )


@dataclass(frozen=True)
class Anchor:
    """Anchor (fiducial) definition."""
    name: str
    type: str
    sensor: str
    expected_pose: Pose
    pos_tol: float
    rot_tol: float
    required: bool

    @classmethod
    def from_dict(cls, d: dict) -> Anchor:
        return cls(
            name=d["name"],
            type=d["type"],
            sensor=d["sensor"],
            expected_pose=Pose.from_dict(d["expected_pose_in_cell"]),
            pos_tol=float(d["tolerance"]["pos_m"]),
            rot_tol=float(d["tolerance"]["rot_rad"]),
            required=bool(d.get("required", False)),
        )


@dataclass(frozen=True)
class MonitorConfig:
    """Assumption monitor configuration."""
    name: str
    type: str  # "anchor_pose_consistency", "safety_io_health", etc.
    inputs: list[str]
    params: dict[str, Any]
    debounce_ms: float
    hysteresis_ms: float
    severity: str  # "degrade", "revoke", "rta_stop"
    action: str

    @classmethod
    def from_dict(cls, d: dict) -> MonitorConfig:
        return cls(
            name=d["name"],
            type=d["type"],
            inputs=list(d["inputs"]),
            params=dict(d.get("params", {})),
            debounce_ms=float(d["debounce_ms"]),
            hysteresis_ms=float(d["hysteresis_ms"]),
            severity=d["severity"],
            action=d["action"],
        )


@dataclass(frozen=True)
class RevocationPolicy:
    """Revocation policy from contract."""
    on_degrade_max_tcp_speed_mps: float
    on_degrade_restrict_to_allowed: bool
    on_degrade_l0c_enforced: bool
    on_revoke_l0c_enabled: bool
    on_revoke_l1_mode: str
    on_rta_stop_sequence: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> RevocationPolicy:
        degrade = d.get("on_degrade", {})
        caps = degrade.get("caps", {})
        revoke = d.get("on_revoke", {})
        rta = d.get("on_rta_stop", {})
        return cls(
            on_degrade_max_tcp_speed_mps=float(caps.get("max_tcp_speed_mps", float("inf"))),
            on_degrade_restrict_to_allowed=bool(caps.get("restrict_to_allowed_regions", False)),
            on_degrade_l0c_enforced=bool(degrade.get("l0c_still_enforced", True)),
            on_revoke_l0c_enabled=bool(revoke.get("l0c_enabled", False)),
            on_revoke_l1_mode=str(revoke.get("l1_mode", "full_conservative")),
            on_rta_stop_sequence=list(rta.get("sequence", ["controlled_stop"])),
        )


@dataclass(frozen=True)
class Contract:
    """Parsed contract (runtime representation).

    All numeric values are native floats (not canonical strings).
    """
    schema_version: str
    contract_id: str
    cell_id: str
    robot_base_pose: Pose
    robot_base_tolerance: PoseTolerance
    keepouts: tuple[Keepout, ...]
    allowed_regions: tuple[AllowedRegion, ...]
    anchors: tuple[Anchor, ...]
    monitors: tuple[MonitorConfig, ...]
    revocation_policy: RevocationPolicy

    @classmethod
    def from_dict(cls, data: dict) -> Contract:
        """Parse contract from raw YAML dict into typed dataclasses."""
        metadata = data["metadata"]
        frames = data["frames"]
        geometry = data.get("geometry", {})
        base = frames["robot_base_in_cell"]

        return cls(
            schema_version=data["schema_version"],
            contract_id=metadata["contract_id"],
            cell_id=metadata["cell_id"],
            robot_base_pose=Pose.from_dict(base),
            robot_base_tolerance=PoseTolerance.from_dict(base.get("tolerance", {"pos_m": 0.01, "rot_rad": 0.05})),
            keepouts=tuple(
                Keepout.from_dict(k) for k in geometry.get("keepouts", [])
            ),
            allowed_regions=tuple(
                AllowedRegion.from_dict(r) for r in geometry.get("allowed_regions", [])
            ),
            anchors=tuple(
                Anchor.from_dict(a) for a in data.get("anchors", [])
            ),
            monitors=tuple(
                MonitorConfig.from_dict(m) for m in data.get("assumption_monitors", [])
            ),
            revocation_policy=RevocationPolicy.from_dict(
                data.get("revocation_policy", {})
            ),
        )
