"""Fail-closed telemetry errors."""


class TelemetryError(ValueError):
    """Raised when telemetry cannot be safely interpreted."""


class BundleImmutableError(TelemetryError):
    """Raised when a caller attempts to replace bundle evidence."""
