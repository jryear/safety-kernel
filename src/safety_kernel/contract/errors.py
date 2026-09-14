class ContractError(Exception):
    """Base class for contract-related failures."""


class ContractParseError(ContractError):
    """The contract artifact could not be parsed (invalid YAML subset)."""


class ContractValidationError(ContractError):
    """The contract artifact is well-formed but violates the schema/semantics."""


class ContractHashMismatchError(ContractError):
    """The contract's integrity hash does not match the canonicalized body."""


class ShiftValidationError(ContractValidationError):
    """A deterministic shift specification is malformed or targets no contract input."""
