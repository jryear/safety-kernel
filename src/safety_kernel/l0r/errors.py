"""Fail-closed L0-R enforcement errors."""


class L0RSafetyInvariantError(RuntimeError):
    """Raised when a projected L0-R command violates a required invariant."""
