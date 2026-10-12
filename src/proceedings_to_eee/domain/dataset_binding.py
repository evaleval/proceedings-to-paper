"""Sidecar that binds a candidate's raw dataset string to an inventory entry.

Attached by deterministic code, never by a model, and deliberately outside the
candidate's stable id, its field-provenance hashes and the review gate payload hash:
binding says what the raw string names, it never replaces the raw string.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from proceedings_to_eee.domain.base import StrictModel

DATASET_BINDING_SCHEMA_VERSION = "dataset-binding/0.1"


class DatasetBindingClass(StrEnum):
    """How a candidate reached its entry, strongest first."""

    EXACT = "exact"
    CAPTION = "caption"
    UNIQUE = "unique"
    NONE = "none"

    @property
    def names_entry(self) -> bool:
        return self is not DatasetBindingClass.NONE


class SplitSource(StrEnum):
    """Where a derived split was read from."""

    COLUMN_HEADER = "column_header"
    CAPTION = "caption"


class DatasetBinding(StrictModel):
    """One candidate's deterministic binding to a paper's dataset inventory."""

    schema_version: Literal["dataset-binding/0.1"] = DATASET_BINDING_SCHEMA_VERSION
    #: Hash of the inventory gate file the binding was computed from.
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_class: DatasetBindingClass
    rule_id: str = Field(min_length=1)
    entry_id: str | None = None
    entry_name: str | None = None
    entry_kind: str | None = None
    languages: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)
    #: Split derived from the anchor when the candidate's scope states none.
    split: str | None = None
    split_source: SplitSource | None = None
    split_page: int | None = Field(default=None, ge=1)
