"""Evidence-first intermediate representation for reported paper results."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import Field, field_validator, model_validator

from proceedings_to_eee.domain.attribution import AttributionVerdict
from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    EvidenceKind,
    ExportStatus,
    ReferentialStatus,
    ReportingStatus,
    TextSupportStatus,
    ValueComparator,
)
from proceedings_to_eee.domain.units import canonicalize_unit


class EvidenceAnchor(StrictModel):
    """Reversible pointer from a candidate back to a source fragment."""

    source_id: str
    page: int = Field(ge=1)
    kind: EvidenceKind
    label: str | None = None
    row: str | None = None
    column: str | None = None
    quote: str = Field(min_length=1)
    quote_sha256: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def populate_quote_hash(self) -> EvidenceAnchor:
        digest = hashlib.sha256(self.quote.encode("utf-8")).hexdigest()
        if self.quote_sha256 is None:
            self.quote_sha256 = digest
        elif self.quote_sha256 != digest:
            raise ValueError("quote_sha256 does not match quote")
        return self


class RoleAssignment(StrictModel):
    """A raw actor name and its role in this exact observation."""

    role: ActorRole
    raw_name: str = Field(min_length=1)
    canonical_id: str | None = None
    version: str | None = None
    provider: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ObservationScope(StrictModel):
    """Dataset and exact subset on which the value applies."""

    dataset_raw: str = Field(min_length=1)
    dataset_id: str | None = None
    dataset_url: str | None = None
    dataset_version: str | None = None
    split: str | None = None
    subset: str | None = None
    group: str | None = None
    language: str | None = None
    sample_count: int | None = Field(default=None, ge=0)
    aggregation: str | None = None
    raw_scope: str | None = None


class MetricSpec(StrictModel):
    """Metric identity, scale, and parameters."""

    raw_name: str = Field(min_length=1)
    canonical_id: str | None = None
    kind: str | None = None
    unit: str | None = None
    lower_is_better: bool | None = None
    min_score: float | None = None
    max_score: float | None = None
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_explicit_unit(cls, value: str | None) -> str | None:
        return canonicalize_unit(value)


class Uncertainty(StrictModel):
    """Uncertainty attached to the same point estimate."""

    standard_error: float | None = None
    standard_deviation: float | None = None
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    confidence_level: float | None = Field(default=None, ge=0.0, le=1.0)
    method: str | None = None
    num_samples: int | None = Field(default=None, ge=0)


class ReportedValue(StrictModel):
    """Raw source value plus an explicitly scaled numeric projection."""

    raw: str = Field(min_length=1)
    numeric: float
    unit: str | None = None
    comparator: ValueComparator = ValueComparator.EXACT
    uncertainty: Uncertainty | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_explicit_unit(cls, value: str | None) -> str | None:
        return canonicalize_unit(value)


class CandidateObservation(StrictModel):
    """One independently checkable result statement from a paper."""

    schema_version: str = "candidate-observation/0.2"
    observation_id: str | None = None
    paper_id: str = Field(min_length=1)
    claim_type: ClaimType
    reporting_status: ReportingStatus = ReportingStatus.PRESENT
    roles: list[RoleAssignment]
    scope: ObservationScope | None = None
    metric: MetricSpec | None = None
    value: ReportedValue | None = None
    evidence: list[EvidenceAnchor] = Field(min_length=1)
    text_support: TextSupportStatus = TextSupportStatus.UNVERIFIED
    referential_status: ReferentialStatus = ReferentialStatus.UNVERIFIED
    export_status: ExportStatus = ExportStatus.NEEDS_REVIEW
    export_reason: str | None = None
    extraction_method: str = "unknown"
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluation_construct: str | None = Field(default=None, alias="construct")
    operationalization: str | None = None
    decision_rule: str | None = None
    evaluation_date: str | None = None
    notes: list[str] = Field(default_factory=list)
    raw_payload_hash: str | None = None
    # Deterministic provenance attribution. Produced by code from structure, never by a
    # model, and deliberately not a ClaimType member: extraction/llm_schema.py types the
    # wire model on ClaimType, so adding one would move provider_json_schema() and void
    # the holdout seal. Absent from the stable_id() payload below, so attaching it leaves
    # observation IDs and EEE filenames byte-identical.
    attribution: AttributionVerdict | None = None

    @model_validator(mode="after")
    def enforce_atomic_invariants(self) -> CandidateObservation:
        if self.reporting_status == ReportingStatus.PRESENT:
            if self.value is None and self.claim_type == ClaimType.PRIMARY_RESULT:
                raise ValueError("present primary_result requires value")
            if self.metric is None and self.claim_type == ClaimType.PRIMARY_RESULT:
                raise ValueError("present primary_result requires metric")
            if self.scope is None and self.claim_type == ClaimType.PRIMARY_RESULT:
                raise ValueError("present primary_result requires scope")
        evaluated = [role for role in self.roles if role.role == ActorRole.EVALUATED_SYSTEM]
        if self.claim_type == ClaimType.PRIMARY_RESULT and len(evaluated) != 1:
            raise ValueError("primary_result requires exactly one evaluated_system")
        if self.observation_id is None:
            self.observation_id = self.stable_id()
        return self

    def stable_id(self) -> str:
        """Return a deterministic semantic ID independent of review state."""

        payload: dict[str, Any] = {
            "paper_id": self.paper_id,
            "claim_type": self.claim_type,
            "roles": [
                {"role": role.role, "raw_name": role.raw_name, "version": role.version}
                for role in self.roles
            ],
            "scope": self.scope.model_dump(mode="json") if self.scope else None,
            "metric": self.metric.model_dump(mode="json") if self.metric else None,
            "value": self.value.model_dump(mode="json") if self.value else None,
            "evidence": [
                {
                    "source_id": anchor.source_id,
                    "page": anchor.page,
                    "kind": anchor.kind,
                    "label": anchor.label,
                    "row": anchor.row,
                    "column": anchor.column,
                }
                for anchor in self.evidence
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "obs_" + hashlib.sha256(encoded).hexdigest()[:20]
