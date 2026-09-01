"""Pinned JSON Schema validation with version-equality guard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from proceedings_to_eee.io import read_json, sha256_file
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH


@dataclass(frozen=True)
class SchemaAuthority:
    path: Path
    version: str
    sha256: str


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def load_schema(
    path: Path = DEFAULT_EEE_SCHEMA_PATH, expected_sha256: str | None = None
) -> tuple[dict[str, Any], SchemaAuthority]:
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"EEE schema hash mismatch: expected {expected_sha256}, got {digest}")
    schema = read_json(path)
    version = schema.get("version")
    if not isinstance(version, str):
        raise ValueError("EEE schema has no string version")
    Draft7Validator.check_schema(schema)
    return schema, SchemaAuthority(path=path, version=version, sha256=digest)


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
