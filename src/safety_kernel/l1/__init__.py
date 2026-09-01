"""L1 learned failure-manifold components.

The L1 package is intentionally additive: it consumes L0-R/L0-C rollout
evidence and never relaxes hard safety constraints.
"""

from safety_kernel.l1.critic import FailureManifoldCritic
from safety_kernel.l1.calibration import ConformalThreshold, calibrate_threshold
from safety_kernel.l1.rollouts import (
    DeterministicPPOBaseline,
    LegacyVelocitySaturatingBaseline,
    RolloutDataset,
)

__all__ = [
    "ConformalThreshold",
    "DeterministicPPOBaseline",
    "FailureManifoldCritic",
    "LegacyVelocitySaturatingBaseline",
    "RolloutDataset",
    "calibrate_threshold",
]
