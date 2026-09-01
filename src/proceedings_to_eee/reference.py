"""Multi-layer reference annotations that remain outside the extraction prompt path."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from proceedings_to_eee.domain.observation import (
    MetricSpec,
    ObservationScope,
    ReportedValue,
    StrictModel,
)
from proceedings_to_eee.domain.status import ActorRole, ClaimType, EvidenceKind


class EvidencePurpose(StrEnum):
    RESULT = "result"
    TABLE_HEADER = "table_header"
    CAPTION = "caption"
    METHOD = "method"
    FOOTNOTE = "footnote"
    APPENDIX = "appendix"
    SYSTEM_DOCUMENTATION = "system_documentation"
    VERSION = "version"
    EVALUATION_CONDITION = "evaluation_condition"
    NEGATIVE_CONTROL = "negative_control"


class EvidenceVerificationMode(StrEnum):
    """How an annotation's quote was checked against the frozen source."""

    TEXT_NORMALIZED = "text_normalized"
    VISUAL = "visual"


class ReferenceEvidence(StrictModel):
    evidence_id: str
    purpose: EvidencePurpose
    page: int = Field(ge=1)
    kind: EvidenceKind
    label: str | None = None
    row: str | None = None
    column: str | None = None
    exact_quote: str = Field(min_length=1)
    source_role: str = "paper"
    verification_mode: EvidenceVerificationMode = EvidenceVerificationMode.TEXT_NORMALIZED
    visual_reviewed: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def validate_verification_mode(self) -> ReferenceEvidence:
        if self.verification_mode == EvidenceVerificationMode.VISUAL and not self.visual_reviewed:
            raise ValueError("visual evidence must be explicitly marked visual_reviewed")
        if self.verification_mode != EvidenceVerificationMode.VISUAL and self.visual_reviewed:
            raise ValueError("visual_reviewed requires visual verification_mode")
        return self


class ReferenceActor(StrictModel):
    role: ActorRole
    raw_name: str
    canonical_id: str | None = None
    version: str | None = None
    provider: str | None = None


class ReferenceObservation(StrictModel):
    reference_id: str
    claim_type: ClaimType
    actors: list[ReferenceActor]
    scope: ObservationScope
    metric: MetricSpec
    value: ReportedValue
    result_evidence_ids: list[str] = Field(min_length=1)
    context_evidence_ids: list[str] = Field(default_factory=list)
    expected_missing_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class NegativeControl(StrictModel):
    control_id: str
    expected_claim_type: ClaimType
    evidence_ids: list[str] = Field(min_length=1)
    reason_not_primary: str


class AnnotationCoverage(StrictModel):
    fully_annotated_labels: list[str] = Field(default_factory=list)
    sampled_labels: list[str] = Field(default_factory=list)
    excluded_modalities: list[str] = Field(default_factory=list)
    inclusion_rule: str
    exclusion_rule: str


class PaperReference(StrictModel):
    schema_version: str = "paper-reference/0.2"
    paper_id: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    annotation_protocol: str
    annotation_status: str
    coverage: AnnotationCoverage
    evidence: list[ReferenceEvidence]
    observations: list[ReferenceObservation]
    negative_controls: list[NegativeControl] = Field(default_factory=list)
    documented_conflicts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> PaperReference:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("reference evidence IDs must be unique")
        known = set(evidence_ids)
        referenced: set[str] = set()
        for observation in self.observations:
            referenced.update(observation.result_evidence_ids)
            referenced.update(observation.context_evidence_ids)
            evaluated = [
                actor for actor in observation.actors if actor.role == ActorRole.EVALUATED_SYSTEM
            ]
            if len(evaluated) != 1:
                raise ValueError(
                    f"reference {observation.reference_id} requires one evaluated system"
                )
        for control in self.negative_controls:
            referenced.update(control.evidence_ids)
            if control.expected_claim_type == ClaimType.PRIMARY_RESULT:
                raise ValueError("negative control cannot expect primary_result")
        missing = referenced - known
        if missing:
            raise ValueError(f"unknown evidence IDs: {sorted(missing)}")
        result_ids = {
            item.evidence_id for item in self.evidence if item.purpose == EvidencePurpose.RESULT
        }
        for observation in self.observations:
            if not set(observation.result_evidence_ids) <= result_ids:
                raise ValueError(
                    f"{observation.reference_id} result evidence has a non-result purpose"
                )
        return self


def load_reference(path: Path) -> PaperReference:
    return PaperReference.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
