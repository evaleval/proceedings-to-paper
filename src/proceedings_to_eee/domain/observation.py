"""Evidence-first intermediate representation for reported paper results."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import Field, field_validator, model_validator

from proceedings_to_eee.domain.attribution import AttributionVerdict
from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.domain.provenance import (
    CandidateField,
    CandidateFieldProvenance,
    ProposalTrace,
    value_sha256,
)
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
    region_id: str | None = None
    planned_row_id: str | None = None
    cell_id: str | None = None
    numeric_token_id: str | None = None
    header_ids: list[str] = Field(default_factory=list)
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
    min_score: float | None = Field(default=None, allow_inf_nan=False)
    max_score: float | None = Field(default=None, allow_inf_nan=False)
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_explicit_unit(cls, value: str | None) -> str | None:
        return canonicalize_unit(value)


class Uncertainty(StrictModel):
    """Uncertainty attached to the same point estimate."""

    standard_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    standard_deviation: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    confidence_interval_lower: float | None = Field(default=None, allow_inf_nan=False)
    confidence_interval_upper: float | None = Field(default=None, allow_inf_nan=False)
    confidence_level: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    method: str | None = None
    num_samples: int | None = Field(default=None, ge=0)


class ReportedValue(StrictModel):
    """Raw source value plus an explicitly scaled numeric projection."""

    raw: str = Field(min_length=1)
    numeric: float = Field(allow_inf_nan=False)
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
    field_provenance: list[CandidateFieldProvenance] = Field(default_factory=list)
    proposal_traces: list[ProposalTrace] = Field(default_factory=list)
    # Deterministic provenance attribution. Produced by code from structure, never by a
    # model, and deliberately not a ClaimType member: extraction/llm_schema.py types the
    # wire model on ClaimType, so adding one would change provider_json_schema() and its
    # recorded contract hash. Absent from stable_id() below, so attaching it leaves
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
        if self.field_provenance:
            fields = [item.field for item in self.field_provenance]
            if len(fields) != len(set(fields)) or set(fields) != set(CandidateField):
                raise ValueError("field provenance must bind every candidate field exactly once")
            expected = self.field_value_sha256s()
            if any(
                item.value_sha256 != expected[item.field]
                and not (
                    item.field is CandidateField.SETTING
                    and item.value_sha256 == self._legacy_setting_value_sha256()
                )
                for item in self.field_provenance
            ):
                raise ValueError("field provenance does not match the candidate tuple")
        trace_ids = [trace.proposal_id for trace in self.proposal_traces]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("candidate repeats a proposal lineage identity")
        if any(trace.paper_id != self.paper_id for trace in self.proposal_traces):
            raise ValueError("proposal lineage belongs to another paper")
        if self.observation_id is None:
            self.observation_id = self.stable_id()
        return self

    def _legacy_setting_value_sha256(self) -> str:
        """Hash the pre-role-governance SETTING projection for sealed-ledger parsing."""

        scope = self.scope
        metric = self.metric
        return value_sha256(
            {
                "scope": (
                    {
                        "split": scope.split,
                        "subset": scope.subset,
                        "group": scope.group,
                        "language": scope.language,
                        "sample_count": scope.sample_count,
                        "aggregation": scope.aggregation,
                        "raw_scope": scope.raw_scope,
                    }
                    if scope is not None
                    else None
                ),
                "metric_parameters": metric.parameters if metric is not None else None,
                "construct": self.evaluation_construct,
                "operationalization": self.operationalization,
                "decision_rule": self.decision_rule,
                "evaluation_date": self.evaluation_date,
            }
        )

    def field_value_sha256s(self) -> dict[CandidateField, str]:
        """Hash the exact tuple projection governed by each provenance field."""

        scope = self.scope
        metric = self.metric
        value = self.value
        evaluated = [
            role.model_dump(mode="json")
            for role in self.roles
            if role.role == ActorRole.EVALUATED_SYSTEM
        ]
        non_evaluated_roles = [
            role.model_dump(mode="json")
            for role in self.roles
            if role.role is not ActorRole.EVALUATED_SYSTEM
        ]
        dataset_scope = (
            {
                "dataset_raw": scope.dataset_raw,
                "dataset_id": scope.dataset_id,
                "dataset_url": scope.dataset_url,
                "dataset_version": scope.dataset_version,
            }
            if scope is not None
            else None
        )
        setting = {
            "scope": (
                {
                    "split": scope.split,
                    "subset": scope.subset,
                    "group": scope.group,
                    "language": scope.language,
                    "sample_count": scope.sample_count,
                    "aggregation": scope.aggregation,
                    "raw_scope": scope.raw_scope,
                }
                if scope is not None
                else None
            ),
            "metric_parameters": metric.parameters if metric is not None else None,
            "non_evaluated_roles": non_evaluated_roles,
            "construct": self.evaluation_construct,
            "operationalization": self.operationalization,
            "decision_rule": self.decision_rule,
            "evaluation_date": self.evaluation_date,
        }
        metric_identity = (
            {
                "raw_name": metric.raw_name,
                "canonical_id": metric.canonical_id,
                "kind": metric.kind,
                "lower_is_better": metric.lower_is_better,
                "min_score": metric.min_score,
                "max_score": metric.max_score,
            }
            if metric is not None
            else None
        )
        value_identity = (
            {
                "raw": value.raw,
                "numeric": value.numeric,
                "comparator": value.comparator,
                "uncertainty": (
                    value.uncertainty.model_dump(mode="json") if value.uncertainty else None
                ),
            }
            if value is not None
            else None
        )
        return {
            CandidateField.SYSTEM: value_sha256(evaluated),
            CandidateField.DATASET_SCOPE: value_sha256(dataset_scope),
            CandidateField.METRIC: value_sha256(metric_identity),
            CandidateField.SETTING: value_sha256(setting),
            CandidateField.VALUE: value_sha256(value_identity),
            CandidateField.UNIT: value_sha256(
                {
                    "metric_unit": metric.unit if metric is not None else None,
                    "value_unit": value.unit if value is not None else None,
                }
            ),
        }

    def populated_fields(self) -> set[CandidateField]:
        """Return semantic fields whose tuple projection carries material data."""

        populated: set[CandidateField] = set()
        if any(role.role is ActorRole.EVALUATED_SYSTEM for role in self.roles):
            populated.add(CandidateField.SYSTEM)
        if self.scope is not None:
            populated.add(CandidateField.DATASET_SCOPE)
        if self.metric is not None:
            populated.add(CandidateField.METRIC)
        if self.value is not None:
            populated.add(CandidateField.VALUE)
        if (self.metric is not None and self.metric.unit is not None) or (
            self.value is not None and self.value.unit is not None
        ):
            populated.add(CandidateField.UNIT)

        scope = self.scope
        metric = self.metric
        setting_values: tuple[object, ...] = (
            scope.split if scope else None,
            scope.subset if scope else None,
            scope.group if scope else None,
            scope.language if scope else None,
            scope.sample_count if scope else None,
            scope.aggregation if scope else None,
            scope.raw_scope if scope else None,
            metric.parameters if metric else None,
            [
                role.model_dump(mode="json")
                for role in self.roles
                if role.role is not ActorRole.EVALUATED_SYSTEM
            ],
            self.evaluation_construct,
            self.operationalization,
            self.decision_rule,
            self.evaluation_date,
        )
        if any(value is not None and value not in ("", (), [], {}) for value in setting_values):
            populated.add(CandidateField.SETTING)
        return populated

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
                    "region_id": anchor.region_id,
                    "planned_row_id": anchor.planned_row_id,
                    "cell_id": anchor.cell_id,
                    "numeric_token_id": anchor.numeric_token_id,
                    "header_ids": anchor.header_ids,
                }
                for anchor in self.evidence
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "obs_" + hashlib.sha256(encoded).hexdigest()[:20]
