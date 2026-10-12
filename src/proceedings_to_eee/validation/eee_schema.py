"""Pinned JSON Schema validation with version-equality guard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from proceedings_to_eee.io import read_json, sha256_file
from proceedings_to_eee.resources import (
    DEFAULT_EEE_SCHEMA_PATH,
    EEE_SCHEMA_SHA256_BY_VERSION,
    eee_schema_path,
)

#: First EEE schema version whose ``metric_config`` accepts ``null`` for ``min_score`` and
#: ``max_score``. Earlier versions require both numbers for a continuous score, so a metric
#: with no established scale cannot be expressed in them at all.
_NULL_BOUNDS_SINCE = (0, 3, 0)


def schema_version_allows_null_bounds(version: str) -> bool:
    """Report whether records under this schema version may carry unresolved bounds.

    A version that is not dotted integers is not a known schema, so it gets the older,
    stricter behaviour rather than a guess.
    """

    parts = version.split(".")
    if not all(part.isdigit() for part in parts):
        return False
    return tuple(int(part) for part in parts) >= _NULL_BOUNDS_SINCE


@dataclass(frozen=True)
class SchemaAuthority:
    path: Path
    version: str
    sha256: str

    @property
    def allows_null_bounds(self) -> bool:
        return schema_version_allows_null_bounds(self.version)


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def load_schema(
    path: Path = DEFAULT_EEE_SCHEMA_PATH, expected_sha256: str | None = None
) -> tuple[dict[str, Any], SchemaAuthority]:
    """Load one EEE schema and bind it to its version and content hash.

    An explicit ``expected_sha256`` is enforced as given. Without one, a schema whose
    version is vendored must still match the vendored pin, so selecting an older version
    by path alone cannot silently load an edited copy of it.
    """

    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"EEE schema hash mismatch: expected {expected_sha256}, got {digest}")
    schema = read_json(path)
    version = schema.get("version")
    if not isinstance(version, str):
        raise ValueError("EEE schema has no string version")
    pinned = EEE_SCHEMA_SHA256_BY_VERSION.get(version)
    if expected_sha256 is None and pinned is not None and digest != pinned:
        raise ValueError(f"EEE schema {version} hash mismatch: vendored pin {pinned}, got {digest}")
    Draft7Validator.check_schema(schema)
    return schema, SchemaAuthority(path=path, version=version, sha256=digest)


def vendored_schema_authority(version: object, sha256: object) -> SchemaAuthority:
    """Bind a recorded ``version``/``sha256`` pair to the vendored schema it names.

    An artifact records the schema it was validated against, and a path that re-validates
    it must use exactly that schema. The pair is accepted only when this package ships
    it: an unknown version or a hash that disagrees with the pin is refused rather than
    quietly validated against the default authority.
    """

    if not isinstance(version, str) or version not in EEE_SCHEMA_SHA256_BY_VERSION:
        raise ValueError(f"EEE schema version {version!r} is not vendored")
    pinned = EEE_SCHEMA_SHA256_BY_VERSION[version]
    if sha256 != pinned:
        raise ValueError(f"EEE schema {version} hash {sha256!r} does not match the vendored pin")
    return SchemaAuthority(path=eee_schema_path(version), version=version, sha256=pinned)


def validate_eee_record(record: dict[str, Any], schema: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if record.get("schema_version") != schema.get("version"):
        issues.append(
            ValidationIssue(
                path="schema_version",
                message=(
                    f"record version {record.get('schema_version')!r} does not equal "
                    f"schema version {schema.get('version')!r}"
                ),
            )
        )
    validator = Draft7Validator(schema)
    for error in sorted(validator.iter_errors(record), key=lambda item: list(item.absolute_path)):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        issues.append(ValidationIssue(path=path, message=error.message))
    return issues
