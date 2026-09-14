"""Frozen Gate 2 telemetry ABI with lazy exports.

The independent checker must be importable without loading producer modules.
"""

__all__ = ["RunBundle", "TelemetryRun", "check_bundle", "finalize_bundle", "structural_replay"]


def __getattr__(name: str):
    if name == "TelemetryRun":
        from safety_kernel.telemetry.adapter import TelemetryRun
        return TelemetryRun
    if name == "RunBundle":
        from safety_kernel.telemetry.bundle import RunBundle
        return RunBundle
    if name in {"check_bundle", "finalize_bundle", "structural_replay"}:
        from safety_kernel.telemetry import checker
        return getattr(checker, name)
    raise AttributeError(name)
