from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

import yaml

from safety_kernel.contract.errors import ContractParseError


_TAG_BOOL = "tag:yaml.org,2002:bool"
_TAG_NULL = "tag:yaml.org,2002:null"
_TAG_INT = "tag:yaml.org,2002:int"
_TAG_FLOAT = "tag:yaml.org,2002:float"


class StrictYAMLLoader(yaml.SafeLoader):
    """YAML loader restricted to a JSON-compatible subset with deterministic typing.

    Properties:
    - rejects anchors/aliases (no shared references)
    - rejects explicit tags
    - rejects duplicate keys
    - restricts implicit typing to: null/bool/int/float (base-10 only)
    - constructs all numeric scalars as Decimal from lexical text
    """

    # Override implicit resolvers entirely (PyYAML defaults are YAML 1.1-ish and too permissive)
    yaml_implicit_resolvers: dict[str, list[tuple[str, re.Pattern[str]]]] = {}

    def compose_node(self, parent, index):  # type: ignore[override]
        if self.check_event(yaml.AliasEvent):
            raise ContractParseError("YAML aliases are not allowed")

        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise ContractParseError("YAML anchors are not allowed")
        if getattr(event, "tag", None) is not None:
            raise ContractParseError("YAML explicit tags are not allowed")

        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep: bool = False):  # type: ignore[override]
        if not isinstance(node, yaml.MappingNode):
            raise ContractParseError("Expected a YAML mapping node")

        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ContractParseError("Mapping keys must be strings")
            if key == "<<":
                raise ContractParseError("YAML merge keys (<<) are not allowed")
            if key in mapping:
                raise ContractParseError(f"Duplicate key {key!r} is not allowed")

            mapping[key] = self.construct_object(value_node, deep=deep)

        return mapping


def _construct_decimal(loader: StrictYAMLLoader, node: yaml.ScalarNode) -> Decimal:
    if not isinstance(node, yaml.ScalarNode):
        raise ContractParseError("Expected a scalar node")
    text = loader.construct_scalar(node)
    try:
        value = Decimal(text)
    except Exception as exc:  # pragma: no cover
        raise ContractParseError(f"Invalid numeric scalar: {text!r}") from exc

    if not value.is_finite():
        raise ContractParseError("NaN/Infinity are not allowed in numeric scalars")
    return value


# Strict implicit typing rules (base-10 only)
StrictYAMLLoader.add_implicit_resolver(  # type: ignore[attr-defined]
    _TAG_NULL, re.compile(r"^(?:null)$"), ["n"]
)
StrictYAMLLoader.add_implicit_resolver(  # type: ignore[attr-defined]
    _TAG_BOOL, re.compile(r"^(?:true|false)$"), ["t", "f"]
)
StrictYAMLLoader.add_implicit_resolver(  # type: ignore[attr-defined]
    _TAG_INT, re.compile(r"^[+-]?(?:[0-9]+)$"), list("+-0123456789")
)
StrictYAMLLoader.add_implicit_resolver(  # type: ignore[attr-defined]
    _TAG_FLOAT,
    re.compile(
        r"^[+-]?(?:"
        r"(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
        r"|"
        r"(?:[0-9]+)(?:[eE][+-]?[0-9]+)"
        r")$"
    ),
    list("+-0123456789."),
)


StrictYAMLLoader.add_constructor(_TAG_INT, _construct_decimal)  # type: ignore[arg-type]
StrictYAMLLoader.add_constructor(_TAG_FLOAT, _construct_decimal)  # type: ignore[arg-type]


def load_yaml_strict(yaml_text: str) -> Any:
    try:
        return yaml.load(yaml_text, Loader=StrictYAMLLoader)
    except ContractParseError:
        raise
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ContractParseError("Invalid YAML") from exc
