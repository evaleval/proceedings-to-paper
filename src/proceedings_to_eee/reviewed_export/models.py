"""Strict private review and quote-free derived-export artifact models."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.domain.export_provenance import ExportProvenanceMode
from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
)
from proceedings_to_eee.domain.provenance import (
    CandidateField,
    CandidateFieldProvenance,
    FieldBindingStatus,
    FieldSourceKind,
    value_sha256,
)
from proceedings_to_eee.domain.status import ActorRole, EvidenceKind

HEX_64 = r"^[0-9a-f]{64}$"
SAFE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
PROTOCOL_ID = "reviewed-export-protocol/0.1"
DERIVED_COUNT_KEYS = (
    "review_items",
    "decisions_completed",
    "decisions_pending",
    "outcomes_exported",
    "outcomes_withheld",
    "outcomes_failed",
    "eee_records",
    "eee_observations",
)


def _normalized_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value in {".", ""}
    ):
        raise ValueError("artifact path must be a normalized relative path")
    return path


class ArtifactRef(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=HEX_64)
    size_bytes: int = Field(ge=0)
    records: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_path(self) -> ArtifactRef:
        _normalized_relative_path(self.path)
        return self


class BoundArtifact(StrictModel):
    run_path: str = Field(min_length=1)
    review_copy: ArtifactRef

    @model_validator(mode="after")
    def validate_run_path(self) -> BoundArtifact:
        _normalized_relative_path(self.run_path)
        return self

    @property
    def sha256(self) -> str:
        return self.review_copy.sha256


class RunSealBinding(StrictModel):
    schema_version: Literal["run-tree-seal/0.1"] = "run-tree-seal/0.1"
    seal_sha256: str = Field(pattern=HEX_64)
    tree_sha256: str = Field(pattern=HEX_64)
    file_count: int = Field(ge=1)
    total_bytes: int = Field(ge=1)
    source_run_name: str = Field(min_length=1)


class PaperRunBinding(StrictModel):
    paper_id: str = Field(pattern=SAFE_ID)
    paper_root: str
    run_manifest: BoundArtifact
    observations: BoundArtifact
    candidate_lineage: BoundArtifact | None = None
    tuple_resolution: BoundArtifact | None = None
    verifier_gates: BoundArtifact | None = None
    source_manifest: BoundArtifact
    layout: BoundArtifact
    result_blocks: BoundArtifact
    layout_parser: str = Field(min_length=1)
    layout_parser_version: str = Field(min_length=1)
    extractor_prompt_sha256: str | None = Field(default=None, pattern=HEX_64)
    extractor_request_contract_sha256: str | None = Field(default=None, pattern=HEX_64)
    row_prompt_sha256: str | None = Field(default=None, pattern=HEX_64)
    row_request_contract_sha256: str | None = Field(default=None, pattern=HEX_64)
    schema_version: str = Field(min_length=1)
    schema_sha256: str = Field(pattern=HEX_64)
    code_git_commit: str | None = None
    code_source_tree_sha256: str | None = Field(default=None, pattern=HEX_64)

    @model_validator(mode="after")
    def validate_paper_root(self) -> PaperRunBinding:
        if self.paper_root != ".":
            _normalized_relative_path(self.paper_root)
        return self


class ReviewManifest(StrictModel):
    schema_version: Literal["reviewed-export-manifest/0.1"] = "reviewed-export-manifest/0.1"
    status: Literal["prepared"] = "prepared"
    run_kind: Literal["paper", "corpus"]
    run_seal: RunSealBinding
    corpus_run: BoundArtifact | None = None
    papers: list[PaperRunBinding] = Field(min_length=1)
    validation_policy_id: Literal["reviewed-export-non-origin/0.1"] = (
        "reviewed-export-non-origin/0.1"
    )
    min_confidence: float = Field(ge=0.0, le=1.0)
    protocol_id: Literal["reviewed-export-protocol/0.1"] = PROTOCOL_ID
    protocol: ArtifactRef
    items: ArtifactRef
    decision_template: ArtifactRef
    item_count: int = Field(ge=0)
    candidates_were_not_mutated: Literal[True] = True
    contains_private_source_text: Literal[True] = True
    public_commit_forbidden: Literal[True] = True

    @model_validator(mode="after")
    def validate_counts(self) -> ReviewManifest:
        if self.items.records != self.item_count:
            raise ValueError("review item count does not match manifest")
        if self.decision_template.records != self.item_count:
            raise ValueError("review decision count does not match manifest")
        ids = [paper.paper_id for paper in self.papers]
        if len(ids) != len(set(ids)):
            raise ValueError("review paper IDs must be unique")
        if self.run_kind == "corpus" and self.corpus_run is None:
            raise ValueError("corpus review requires a corpus-run binding")
        if self.run_kind == "paper" and self.corpus_run is not None:
            raise ValueError("paper review cannot bind a corpus-run artifact")
        return self


class CandidateFingerprint(StrictModel):
    schema_version: Literal["candidate-fingerprint/0.1"] = "candidate-fingerprint/0.1"
    canonicalization: Literal["pydantic-json-by-alias-include-null/canonical-json/0.1"] = (
        "pydantic-json-by-alias-include-null/canonical-json/0.1"
    )
    paper_id: str = Field(pattern=SAFE_ID)
    observation_id: str = Field(min_length=1)
    observation_ledger_path: str = Field(min_length=1)
    observation_ledger_sha256: str = Field(pattern=HEX_64)
    observation_ledger_line: int = Field(ge=1)
    observation_ledger_line_sha256: str = Field(pattern=HEX_64)
    candidate_schema_version: str = Field(min_length=1)
    candidate_payload_sha256: str = Field(pattern=HEX_64)
    structural_identity_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def validate_path(self) -> CandidateFingerprint:
        _normalized_relative_path(self.observation_ledger_path)
        return self


class ReviewedTuple(StrictModel):
    roles: list[RoleAssignment]
    scope: ObservationScope | None = None
    metric: MetricSpec | None = None
    value: ReportedValue | None = None
    evaluation_construct: str | None = None
    operationalization: str | None = None
    decision_rule: str | None = None
    evaluation_date: str | None = None
    field_provenance: list[CandidateFieldProvenance] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    candidate_occurrence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def lineage_lists_are_exact(self) -> ReviewedTuple:
        if self.proposal_ids != sorted(set(self.proposal_ids)):
            raise ValueError("reviewed tuple proposal ids are not canonical")
        if self.candidate_occurrence_ids != sorted(set(self.candidate_occurrence_ids)):
            raise ValueError("reviewed tuple candidate occurrence ids are not canonical")
        if len(self.proposal_ids) != len(self.candidate_occurrence_ids):
            raise ValueError("reviewed tuple proposal and candidate occurrence counts differ")
        return self

    @classmethod
    def from_candidate(cls, candidate: CandidateObservation) -> ReviewedTuple:
        return cls(
            roles=[role.model_copy(deep=True) for role in candidate.roles],
            scope=candidate.scope.model_copy(deep=True) if candidate.scope else None,
            metric=candidate.metric.model_copy(deep=True) if candidate.metric else None,
            value=candidate.value.model_copy(deep=True) if candidate.value else None,
            evaluation_construct=candidate.evaluation_construct,
            operationalization=candidate.operationalization,
            decision_rule=candidate.decision_rule,
            evaluation_date=candidate.evaluation_date,
            field_provenance=[item.model_copy(deep=True) for item in candidate.field_provenance],
            proposal_ids=sorted(trace.proposal_id for trace in candidate.proposal_traces),
            candidate_occurrence_ids=sorted(
                trace.candidate_occurrence_id for trace in candidate.proposal_traces
            ),
        )

    @property
    def is_complete(self) -> bool:
        return self.scope is not None and self.metric is not None and self.value is not None

    def populated_fields(self) -> set[CandidateField]:
        """Return semantic tuple fields that carry a non-empty reviewed value."""

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
        non_evaluated_roles = [
            role.model_dump(mode="json")
            for role in self.roles
            if role.role is not ActorRole.EVALUATED_SYSTEM
        ]
        setting_values: tuple[object, ...] = (
            scope.split if scope else None,
            scope.subset if scope else None,
            scope.group if scope else None,
            scope.language if scope else None,
            scope.sample_count if scope else None,
            scope.aggregation if scope else None,
            scope.raw_scope if scope else None,
            metric.parameters if metric else None,
            non_evaluated_roles,
            self.evaluation_construct,
            self.operationalization,
            self.decision_rule,
            self.evaluation_date,
        )
        if any(value is not None and value not in ("", (), [], {}) for value in setting_values):
            populated.add(CandidateField.SETTING)
        return populated

    def field_value_sha256s(self) -> dict[CandidateField, str]:
        """Hash the same field projections as the immutable candidate tuple."""

        evaluated = [
            role.model_dump(mode="json")
            for role in self.roles
            if role.role is ActorRole.EVALUATED_SYSTEM
        ]
        non_evaluated_roles = [
            role.model_dump(mode="json")
            for role in self.roles
            if role.role is not ActorRole.EVALUATED_SYSTEM
        ]
        scope = self.scope
        metric = self.metric
        value = self.value
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


def review_evidence_span_id(
    *,
    source_id: str,
    page: int,
    page_text_sha256: str,
    char_start: int,
    char_end: int,
    excerpt_sha256: str,
) -> str:
    """Return the stable identity of one exact occurrence in a frozen layout page."""

    payload = {
        "source_id": source_id,
        "page": page,
        "page_text_sha256": page_text_sha256,
        "char_start": char_start,
        "char_end": char_end,
        "excerpt_sha256": excerpt_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "span_" + hashlib.sha256(encoded).hexdigest()


class ReviewEvidenceAnchor(StrictModel):
    schema_version: Literal["review-evidence-anchor/0.1"] = "review-evidence-anchor/0.1"
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=HEX_64)
    source_manifest_sha256: str = Field(pattern=HEX_64)
    layout_sha256: str = Field(pattern=HEX_64)
    parser: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    page: int = Field(ge=1)
    page_text_sha256: str = Field(pattern=HEX_64)
    span_id: str | None = Field(default=None, pattern=r"^span_[0-9a-f]{64}$")
    exact_excerpt: str = Field(min_length=1)
    excerpt_sha256: str = Field(pattern=HEX_64)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    kind: EvidenceKind
    region_id: str | None = None
    label: str | None = None
    row: str | None = None
    column: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_excerpt_hash_and_spans(self) -> ReviewEvidenceAnchor:
        digest = hashlib.sha256(self.exact_excerpt.encode("utf-8")).hexdigest()
        if self.excerpt_sha256 != digest:
            raise ValueError("excerpt_sha256 does not match exact_excerpt")
        if self.char_end <= self.char_start:
            raise ValueError("evidence character span is empty or reversed")
        if self.end_line < self.start_line:
            raise ValueError("evidence line span is reversed")
        expected_span_id = review_evidence_span_id(
            source_id=self.source_id,
            page=self.page,
            page_text_sha256=self.page_text_sha256,
            char_start=self.char_start,
            char_end=self.char_end,
            excerpt_sha256=self.excerpt_sha256,
        )
        if self.span_id is None:
            object.__setattr__(self, "span_id", expected_span_id)
        return self


class ReviewItem(StrictModel):
    schema_version: Literal["reviewed-export-item/0.1"] = "reviewed-export-item/0.1"
    item_id: str = Field(pattern=r"^review_[0-9a-f]{24}$")
    fingerprint: CandidateFingerprint
    candidate: CandidateObservation
    reviewed_tuple: ReviewedTuple
    base_gate_status: str = Field(min_length=1)
    base_gate_reason: str | None = None
    source_export_mode: ExportProvenanceMode = ExportProvenanceMode.LEGACY_MANUAL
    source_tuple_sidecar_sha256: str | None = Field(default=None, pattern=HEX_64)
    source_tuple_gate_sha256: str | None = Field(default=None, pattern=HEX_64)
    source_verifier_sidecar_sha256: str | None = Field(default=None, pattern=HEX_64)
    source_verifier_gate_sha256: str | None = Field(default=None, pattern=HEX_64)
    source_verifier_gate_passed: bool | None = None
    suggested_result_evidence: list[ReviewEvidenceAnchor] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_export_binding(self) -> ReviewItem:
        if self.source_export_mode.has_tuple_audit != (
            self.source_tuple_sidecar_sha256 is not None
            and self.source_tuple_gate_sha256 is not None
        ):
            raise ValueError("review item tuple provenance binding is incomplete")
        if self.source_verifier_sidecar_sha256 is None and (
            self.source_verifier_gate_sha256 is not None
            or self.source_verifier_gate_passed is not None
        ):
            raise ValueError("review item verifier provenance binding is incomplete")
        if self.source_verifier_sidecar_sha256 is not None and (
            self.source_verifier_gate_passed is None
            or (
                self.source_verifier_gate_passed is True
                and self.source_verifier_gate_sha256 is None
            )
        ):
            raise ValueError("review item verifier outcome is incomplete")
        return self


class DecisionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"


class TupleDecision(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class OriginDecision(StrEnum):
    PAPER_PRODUCED = "paper_produced"
    EXTERNALLY_SOURCED = "externally_sourced"
    UNRESOLVED = "unresolved"


class ReviewAuthorityMode(StrEnum):
    SINGLE_EXPERT = "single_expert"
    DUAL_CONSENSUS = "dual_consensus"
    ADJUDICATED = "adjudicated"


class DecisionAuthority(StrictModel):
    mode: ReviewAuthorityMode
    reviewer_ids: list[str] = Field(min_length=1)
    adjudicator_id: str | None = None
    protocol_id: Literal["reviewed-export-protocol/0.1"] = PROTOCOL_ID
    protocol_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def validate_authority(self) -> DecisionAuthority:
        if len(self.reviewer_ids) != len(set(self.reviewer_ids)):
            raise ValueError("reviewer IDs must be distinct")
        if any(not value.strip() for value in self.reviewer_ids):
            raise ValueError("reviewer IDs cannot be blank")
        if self.mode is ReviewAuthorityMode.SINGLE_EXPERT:
            if len(self.reviewer_ids) != 1 or self.adjudicator_id is not None:
                raise ValueError("single_expert requires exactly one reviewer")
        elif self.mode is ReviewAuthorityMode.DUAL_CONSENSUS:
            if len(self.reviewer_ids) != 2 or self.adjudicator_id is not None:
                raise ValueError("dual_consensus requires exactly two reviewers")
        else:
            if len(self.reviewer_ids) < 2 or not self.adjudicator_id:
                raise ValueError("adjudicated authority requires reviewers and an adjudicator")
            if self.adjudicator_id in self.reviewer_ids:
                raise ValueError("adjudicator must be distinct from reviewers")
        return self


class ReviewFieldAttestation(StrictModel):
    """Human binding of one unchanged tuple field to retained exact result spans."""

    schema_version: Literal["review-field-attestation/0.1"] = "review-field-attestation/0.1"
    field: CandidateField
    value_sha256: str = Field(pattern=HEX_64)
    span_ids: list[str] = Field(min_length=1)

    @field_validator("span_ids")
    @classmethod
    def span_ids_are_canonical(cls, value: list[str]) -> list[str]:
        if any(not re.fullmatch(r"span_[0-9a-f]{64}", item) for item in value):
            raise ValueError("field attestation contains an invalid span id")
        if value != sorted(set(value)):
            raise ValueError("field attestation span ids are not canonical")
        return value


class ReviewedExportDecision(StrictModel):
    schema_version: Literal["reviewed-export-decision/0.1"] = "reviewed-export-decision/0.1"
    decision_id: str = Field(pattern=r"^decision_[0-9a-f]{24}$")
    item_id: str = Field(pattern=r"^review_[0-9a-f]{24}$")
    candidate_payload_sha256: str = Field(pattern=HEX_64)
    status: DecisionStatus = DecisionStatus.PENDING
    reviewed_tuple: ReviewedTuple
    tuple_decision: TupleDecision | None = None
    origin_decision: OriginDecision | None = None
    result_evidence: list[ReviewEvidenceAnchor] = Field(default_factory=list)
    field_attestations: list[ReviewFieldAttestation] = Field(default_factory=list)
    origin_evidence: list[ReviewEvidenceAnchor] = Field(default_factory=list)
    authority: DecisionAuthority | None = None
    decided_at: datetime | None = None
    candidate_mutated: Literal[False] = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_completion_shape(self) -> ReviewedExportDecision:
        if self.status is DecisionStatus.PENDING:
            if any(
                value is not None
                for value in (
                    self.tuple_decision,
                    self.origin_decision,
                    self.authority,
                    self.decided_at,
                )
            ):
                raise ValueError("pending decisions cannot contain completed decision fields")
            if self.origin_evidence:
                raise ValueError("pending decisions cannot contain origin evidence")
            if self.field_attestations:
                raise ValueError("pending decisions cannot claim field attestations")
            return self
        if any(
            value is None
            for value in (
                self.tuple_decision,
                self.origin_decision,
                self.authority,
                self.decided_at,
            )
        ):
            raise ValueError("completed decision is missing required fields")
        assert self.decided_at is not None
        if self.decided_at.tzinfo is None or self.decided_at.utcoffset() is None:
            raise ValueError("decided_at must include a UTC offset")
        if not self.result_evidence:
            raise ValueError("completed decision requires result evidence")
        if (
            self.origin_decision
            in {
                OriginDecision.PAPER_PRODUCED,
                OriginDecision.EXTERNALLY_SOURCED,
            }
            and not self.origin_evidence
        ):
            raise ValueError("decisive origin requires separate origin evidence")
        if self.tuple_decision is TupleDecision.CONFIRMED and not self.reviewed_tuple.is_complete:
            raise ValueError("confirmed tuple must contain scope, metric, and value")
        if self.tuple_decision is TupleDecision.CONFIRMED and not self.field_attestations:
            raise ValueError("confirmed tuple requires explicit field attestations")
        if self.tuple_decision is not TupleDecision.CONFIRMED and self.field_attestations:
            raise ValueError("unconfirmed tuple cannot claim field attestations")
        fields = [attestation.field for attestation in self.field_attestations]
        if fields != sorted(set(fields), key=lambda item: item.value):
            raise ValueError("decision field attestations are not canonical")
        return self


class LockedDecision(StrictModel):
    decision_id: str
    item_id: str
    decision_sha256: str = Field(pattern=HEX_64)
    status: DecisionStatus


class ReviewLock(StrictModel):
    schema_version: Literal["reviewed-export-lock/0.1"] = "reviewed-export-lock/0.1"
    status: Literal["locked"] = "locked"
    review_manifest_sha256: str = Field(pattern=HEX_64)
    items_sha256: str = Field(pattern=HEX_64)
    decisions_sha256: str = Field(pattern=HEX_64)
    protocol_sha256: str = Field(pattern=HEX_64)
    source_run_seal_sha256: str = Field(pattern=HEX_64)
    source_run_tree_sha256: str = Field(pattern=HEX_64)
    decision_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    locked_at: datetime | None = None
    decisions: list[LockedDecision]
    candidates_were_not_mutated: Literal[True] = True

    @model_validator(mode="after")
    def validate_counts(self) -> ReviewLock:
        if self.decision_count != len(self.decisions):
            raise ValueError("locked decision count does not match records")
        if self.completed_count + self.pending_count != self.decision_count:
            raise ValueError("locked completion counts do not partition decisions")
        return self


class ExportOutcomeState(StrEnum):
    EXPORTED = "exported"
    WITHHELD = "withheld"
    FAILED = "failed"


class ExportOutcome(StrictModel):
    schema_version: Literal["reviewed-export-outcome/0.1"] = "reviewed-export-outcome/0.1"
    item_id: str
    decision_id: str
    paper_id: str
    observation_id: str
    candidate_payload_sha256: str = Field(pattern=HEX_64)
    state: ExportOutcomeState
    failure_codes: list[str] = Field(default_factory=list)
    evaluation_id: str | None = None
    evaluation_result_id: str | None = None
    eee_path: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ExportOutcome:
        if self.state is ExportOutcomeState.EXPORTED:
            if self.failure_codes or not all(
                (self.evaluation_id, self.evaluation_result_id, self.eee_path)
            ):
                raise ValueError("exported outcome requires output identities and no failures")
        elif not self.failure_codes:
            raise ValueError("non-exported outcome requires at least one failure code")
        if self.eee_path is not None:
            _normalized_relative_path(self.eee_path)
        return self


class QuoteFreeEvidence(StrictModel):
    source_id: str
    source_sha256: str = Field(pattern=HEX_64)
    page: int = Field(ge=1)
    page_text_sha256: str = Field(pattern=HEX_64)
    span_id: str | None = Field(default=None, pattern=r"^span_[0-9a-f]{64}$")
    excerpt_sha256: str = Field(pattern=HEX_64)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    kind: EvidenceKind
    region_id: str | None = None
    label: str | None = None
    row: str | None = None
    column: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def span_identity_is_exact(self) -> QuoteFreeEvidence:
        expected_span_id = review_evidence_span_id(
            source_id=self.source_id,
            page=self.page,
            page_text_sha256=self.page_text_sha256,
            char_start=self.char_start,
            char_end=self.char_end,
            excerpt_sha256=self.excerpt_sha256,
        )
        if self.span_id is None:
            object.__setattr__(self, "span_id", expected_span_id)
        elif self.span_id != expected_span_id:
            raise ValueError("evidence span id does not match its exact occurrence")
        return self

    @classmethod
    def from_private(cls, anchor: ReviewEvidenceAnchor) -> QuoteFreeEvidence:
        payload = anchor.model_dump(mode="json", exclude={"exact_excerpt"})
        payload.pop("schema_version", None)
        payload.pop("source_manifest_sha256", None)
        payload.pop("layout_sha256", None)
        payload.pop("parser", None)
        payload.pop("parser_version", None)
        return cls.model_validate(payload)


class ReviewedExportProvenance(StrictModel):
    schema_version: Literal["reviewed-export-provenance/0.1"] = "reviewed-export-provenance/0.1"
    paper_id: str
    observation_id: str
    evaluation_id: str
    evaluation_result_id: str
    eee_path: str
    candidate_payload_sha256: str = Field(pattern=HEX_64)
    review_manifest_sha256: str = Field(pattern=HEX_64)
    review_lock_sha256: str = Field(pattern=HEX_64)
    review_decision_sha256: str = Field(pattern=HEX_64)
    export_provenance_mode: ExportProvenanceMode
    export_composition_sha256: str = Field(pattern=HEX_64)
    source_tuple_sidecar_sha256: str | None = Field(default=None, pattern=HEX_64)
    source_tuple_gate_sha256: str | None = Field(default=None, pattern=HEX_64)
    source_verifier_sidecar_sha256: str | None = Field(default=None, pattern=HEX_64)
    source_verifier_gate_sha256: str | None = Field(default=None, pattern=HEX_64)
    source_verifier_gate_passed: bool | None = None
    origin_decision: Literal["paper_produced"] = "paper_produced"
    decision_authority: ReviewAuthorityMode
    decision_protocol_id: str
    decision_protocol_sha256: str = Field(pattern=HEX_64)
    result_evidence: list[QuoteFreeEvidence] = Field(min_length=1)
    origin_evidence: list[QuoteFreeEvidence] = Field(min_length=1)
    proposal_ids: list[str] = Field(default_factory=list)
    candidate_occurrence_ids: list[str] = Field(default_factory=list)
    field_provenance: list[CandidateFieldProvenance] = Field(default_factory=list)
    field_attestations: list[ReviewFieldAttestation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_path(self) -> ReviewedExportProvenance:
        _normalized_relative_path(self.eee_path)
        if self.export_provenance_mode.has_tuple_audit != (
            self.source_tuple_sidecar_sha256 is not None
            and self.source_tuple_gate_sha256 is not None
        ):
            raise ValueError("reviewed provenance tuple binding is incomplete")
        if not self.export_provenance_mode.is_human_reviewed:
            raise ValueError("reviewed provenance must use a human-reviewed export mode")
        if self.source_verifier_sidecar_sha256 is None and (
            self.source_verifier_gate_sha256 is not None
            or self.source_verifier_gate_passed is not None
        ):
            raise ValueError("reviewed provenance verifier binding is incomplete")
        if self.source_verifier_sidecar_sha256 is not None and (
            self.source_verifier_gate_passed is None
            or (
                self.source_verifier_gate_passed is True
                and self.source_verifier_gate_sha256 is None
            )
        ):
            raise ValueError("reviewed provenance verifier outcome is incomplete")

        expected_fields = list(CandidateField)
        if [binding.field for binding in self.field_provenance] != expected_fields:
            raise ValueError("reviewed field provenance is not complete and canonical")
        for binding in self.field_provenance:
            source_payloads = [
                json.dumps(source.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
                for source in binding.sources
            ]
            if source_payloads != sorted(set(source_payloads)):
                raise ValueError("reviewed field provenance sources are not canonical")

        attested_fields = [attestation.field for attestation in self.field_attestations]
        if attested_fields != sorted(set(attested_fields), key=lambda item: item.value):
            raise ValueError("reviewed field attestations are not canonical")

        result_by_span: dict[str, QuoteFreeEvidence] = {}
        for anchor in self.result_evidence:
            if anchor.span_id is None or anchor.span_id in result_by_span:
                raise ValueError("reviewed result evidence span ids are missing or duplicated")
            result_by_span[anchor.span_id] = anchor

        provenance_by_field = {binding.field: binding for binding in self.field_provenance}
        attestation_by_field = {
            attestation.field: attestation for attestation in self.field_attestations
        }
        bound_fields = {
            field
            for field, binding in provenance_by_field.items()
            if binding.status is FieldBindingStatus.BOUND
        }
        if bound_fields != set(attestation_by_field):
            raise ValueError(
                "reviewed field attestations do not exactly identify bound populated fields"
            )

        for field, binding in provenance_by_field.items():
            review_sources = [
                source for source in binding.sources if source.kind is FieldSourceKind.REVIEW_SPAN
            ]
            attestation = attestation_by_field.get(field)
            if attestation is None:
                if review_sources:
                    raise ValueError("unattested field contains a dangling review-span source")
                continue
            if binding.value_sha256 != attestation.value_sha256:
                raise ValueError("reviewed field attestation and provenance value hashes differ")
            source_by_span = {source.review_span_id: source for source in review_sources}
            if (
                None in source_by_span
                or len(source_by_span) != len(review_sources)
                or set(source_by_span) != set(attestation.span_ids)
            ):
                raise ValueError(
                    "reviewed field provenance does not exactly cover attested result spans"
                )
            for span_id, source in source_by_span.items():
                assert span_id is not None
                anchor = result_by_span.get(span_id)
                if anchor is None or (
                    source.source_id != anchor.source_id
                    or source.page != anchor.page
                    or source.region_id != anchor.region_id
                    or source.quote_sha256 != anchor.excerpt_sha256
                ):
                    raise ValueError(
                        "reviewed field provenance does not cross-bind its exact result span"
                    )
        return self


class DerivedRunManifest(StrictModel):
    schema_version: Literal["reviewed-derived-run/0.1"] = "reviewed-derived-run/0.1"
    status: Literal["verified-reviewed-export"] = "verified-reviewed-export"
    source_run_name: str
    source_run_seal_sha256: str = Field(pattern=HEX_64)
    source_run_tree_sha256: str = Field(pattern=HEX_64)
    review_manifest_sha256: str = Field(pattern=HEX_64)
    review_lock_sha256: str = Field(pattern=HEX_64)
    decisions_sha256: str = Field(pattern=HEX_64)
    eee_schema_version: str
    eee_schema_sha256: str = Field(pattern=HEX_64)
    export_provenance_modes: list[ExportProvenanceMode]
    tuple_sidecar_sha256s: list[str] = Field(default_factory=list)
    verifier_sidecar_sha256s: list[str] = Field(default_factory=list)
    locked_at: datetime | None = None
    counts: dict[str, int]
    payload_files: list[ArtifactRef]
    payload_tree_sha256: str = Field(pattern=HEX_64)
    sha256s_sha256: str = Field(pattern=HEX_64)
    source_run_unchanged: Literal[True] = True
    deterministic_recomposition: Literal[True] = True
    contains_evidence_quotations: Literal[False] = False
    contains_reviewer_identities: Literal[False] = False
    contains_absolute_paths: Literal[False] = False

    @field_validator("counts", mode="before")
    @classmethod
    def validate_counts(cls, value: Any) -> Any:
        if not isinstance(value, dict) or set(value) != set(DERIVED_COUNT_KEYS):
            raise ValueError("derived counts must contain the exact supported fields")
        if any(type(count) is not int or count < 0 for count in value.values()):
            raise ValueError("derived counts must be non-negative integers")
        return value

    @model_validator(mode="after")
    def validate_export_provenance(self) -> DerivedRunManifest:
        if self.export_provenance_modes != sorted(
            set(self.export_provenance_modes), key=lambda item: item.value
        ):
            raise ValueError("derived export provenance modes are not canonical")
        if any(not mode.is_human_reviewed for mode in self.export_provenance_modes):
            raise ValueError("derived export provenance must be human reviewed")
        if self.tuple_sidecar_sha256s != sorted(set(self.tuple_sidecar_sha256s)):
            raise ValueError("derived tuple sidecar hashes are not canonical")
        if any(mode.has_tuple_audit for mode in self.export_provenance_modes) != bool(
            self.tuple_sidecar_sha256s
        ):
            raise ValueError("derived tuple provenance binding is inconsistent")
        if self.verifier_sidecar_sha256s != sorted(set(self.verifier_sidecar_sha256s)):
            raise ValueError("derived verifier sidecar hashes are not canonical")
        if self.verifier_sidecar_sha256s and not any(
            mode is ExportProvenanceMode.TUPLE_AUDITED_HUMAN_REVIEWED
            for mode in self.export_provenance_modes
        ):
            raise ValueError("derived verifier audit requires tuple-audited human review")
        return self


class DerivedVerification(StrictModel):
    schema_version: Literal["reviewed-derived-verification/0.1"] = (
        "reviewed-derived-verification/0.1"
    )
    status: Literal["verified"] = "verified"
    derived_run_sha256: str = Field(pattern=HEX_64)
    sha256s_sha256: str = Field(pattern=HEX_64)
    payload_tree_sha256: str = Field(pattern=HEX_64)
    payload_file_count: int = Field(ge=0)
    eee_record_count: int = Field(ge=0)
    exported_observation_count: int = Field(ge=0)
    quote_free_dual_provenance_count: int = Field(ge=0)
    no_unexpected_files: Literal[True] = True
    all_eee_schema_valid: Literal[True] = True
    all_exports_have_dual_provenance: Literal[True] = True
    no_duplicate_evaluation_results: Literal[True] = True


def model_payload(value: StrictModel) -> dict[str, Any]:
    """Canonical hash payload that retains every explicit null field."""

    return value.model_dump(mode="json", by_alias=True, exclude_none=False)
