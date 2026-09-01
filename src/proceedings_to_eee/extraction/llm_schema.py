"""Strict provider wire schema; converted into richer domain objects after validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from proceedings_to_eee.domain.status import ActorRole, ClaimType, EvidenceKind, ValueComparator
from proceedings_to_eee.extraction.row_enumeration import RowDisposition


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)


class WireRole(WireModel):
    role: ActorRole
    raw_name: str
    version: str | None
    provider: str | None
    confidence: float = Field(ge=0.0, le=1.0)


class WireScope(WireModel):
    dataset_raw: str
    dataset_id: str | None
    dataset_url: str | None
    dataset_version: str | None
    split: str | None
    subset: str | None
    group: str | None
    language: str | None
    sample_count: int | None
    aggregation: str | None
    raw_scope: str | None


class WireMetric(WireModel):
    raw_name: str
    canonical_id: str | None
    kind: str | None
    unit: str | None
    lower_is_better: bool | None
    min_score: float | None
    max_score: float | None
    parameters: dict[str, str | int | float | bool | None]


class WireUncertainty(WireModel):
    standard_error: float | None
    standard_deviation: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    confidence_level: float | None
    method: str | None
    num_samples: int | None


class WireValue(WireModel):
    raw: str
    numeric: float
    unit: str | None
    comparator: ValueComparator
    uncertainty: WireUncertainty | None


class WireEvidence(WireModel):
    kind: EvidenceKind
    label: str | None
    row: str | None
    column: str | None
    quote: str


class WireObservation(WireModel):
    claim_type: ClaimType
    roles: list[WireRole]
    scope: WireScope | None
    metric: WireMetric | None
    value: WireValue | None
    evidence: list[WireEvidence]
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    evaluation_construct: str | None = Field(alias="construct")
    operationalization: str | None
    decision_rule: str | None
    evaluation_date: str | None
    notes: list[str]


class WireExtraction(WireModel):
    observations: list[WireObservation]
    page_summary: str
    warnings: list[str]


class WireRowDisposition(WireModel):
    row_id: str
    disposition: RowDisposition
    observations: list[WireObservation]
    note: str | None


class WireRowExtraction(WireModel):
    dispositions: list[WireRowDisposition]
    warnings: list[str]


def provider_json_schema() -> dict[str, Any]:
    """Schema where every property is present and nullable unknowns are explicit."""

    schema = WireExtraction.model_json_schema(mode="validation")
    schema.pop("title", None)
    return schema


def row_provider_json_schema() -> dict[str, Any]:
    """Strict provider schema for row dispositions; input-ID coverage is checked locally."""

    schema = WireRowExtraction.model_json_schema(mode="validation")
    schema.pop("title", None)
    return schema
