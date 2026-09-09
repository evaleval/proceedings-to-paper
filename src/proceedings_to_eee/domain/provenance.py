"""Quote-free candidate field and extraction-proposal provenance."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import Field, model_validator

from proceedings_to_eee.domain.base import StrictModel


class CandidateField(StrEnum):
    SYSTEM = "system"
    DATASET_SCOPE = "dataset_scope"
    METRIC = "metric"
    SETTING = "setting"
    VALUE = "value"
    UNIT = "unit"


class FieldBindingStatus(StrEnum):
    BOUND = "bound"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    CONFLICT = "conflict"


class FieldSourceKind(StrEnum):
    EVIDENCE_QUOTE = "evidence_quote"
    REVIEW_SPAN = "review_span"
    ROW_LABEL = "row_label"
    TABLE_CAPTION = "table_caption"
    HEADER_PATH = "header_path"
    PHYSICAL_CELL = "physical_cell"
    NUMERIC_TOKEN = "numeric_token"


class FieldSourceRef(StrictModel):
    """Quote-free pointer to the exact source structure used for one tuple field."""

    kind: FieldSourceKind
    source_id: str
    page: int = Field(ge=1)
    region_id: str | None = None
    planned_row_id: str | None = None
    row_label_cell_id: str | None = None
    physical_cell_id: str | None = None
    numeric_token_id: str | None = None
    header_ids: list[str] = Field(default_factory=list)
    quote_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    review_span_id: str | None = Field(default=None, pattern=r"^span_[0-9a-f]{64}$")

    @model_validator(mode="after")
    def required_identity_is_present(self) -> FieldSourceRef:
        if self.kind is FieldSourceKind.EVIDENCE_QUOTE and self.quote_sha256 is None:
            raise ValueError("evidence-quote field source requires a quote hash")
        if self.kind is FieldSourceKind.REVIEW_SPAN and (
            self.quote_sha256 is None or self.review_span_id is None
        ):
            raise ValueError("review-span field source requires span and quote hashes")
        if self.kind is not FieldSourceKind.REVIEW_SPAN and self.review_span_id is not None:
            raise ValueError("only a review-span field source may carry a review span id")
        if self.kind is FieldSourceKind.ROW_LABEL and (
            self.planned_row_id is None or self.row_label_cell_id is None
        ):
            raise ValueError("row-label field source requires row and label-cell ids")
        if self.kind is FieldSourceKind.HEADER_PATH and not self.header_ids:
            raise ValueError("header-path field source requires at least one header id")
        if self.kind is FieldSourceKind.PHYSICAL_CELL and self.physical_cell_id is None:
            raise ValueError("physical-cell field source requires a physical cell id")
        if self.kind is FieldSourceKind.NUMERIC_TOKEN and (
            self.physical_cell_id is None or self.numeric_token_id is None
        ):
            raise ValueError("numeric-token field source requires cell and token ids")
        if len(self.header_ids) != len(set(self.header_ids)):
            raise ValueError("field source repeats a header id")
        return self


class CandidateFieldProvenance(StrictModel):
    """Binding of one semantic tuple field to quote-free source identities."""

    field: CandidateField
    value_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: FieldBindingStatus
    sources: list[FieldSourceRef] = Field(default_factory=list)
    alternate_value_sha256s: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def status_matches_sources(self) -> CandidateFieldProvenance:
        if (
            self.status
            in {
                FieldBindingStatus.BOUND,
                FieldBindingStatus.AMBIGUOUS,
                FieldBindingStatus.CONFLICT,
            }
            and not self.sources
        ):
            raise ValueError("field provenance state requires a source binding")
        if self.status is FieldBindingStatus.CONFLICT and not self.reason:
            raise ValueError("field provenance conflict requires a reason")
        if self.status is FieldBindingStatus.CONFLICT and not self.alternate_value_sha256s:
            raise ValueError("field provenance conflict requires alternate value hashes")
        if self.status is not FieldBindingStatus.CONFLICT and self.alternate_value_sha256s:
            raise ValueError("only conflicting field provenance can retain alternate values")
        if self.alternate_value_sha256s != sorted(set(self.alternate_value_sha256s)):
            raise ValueError("alternate field-value hashes are not canonical")
        if self.value_sha256 in self.alternate_value_sha256s:
            raise ValueError("primary field-value hash cannot repeat as an alternate")
        canonical = [
            json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            for item in self.sources
        ]
        if len(canonical) != len(set(canonical)):
            raise ValueError("field provenance repeats a source binding")
        return self


class ProposalStage(StrEnum):
    LEGACY_BLOCK = "legacy_block"
    ROW_ENUMERATION = "row_enumeration"


class ProposalTrace(StrictModel):
    """Stable quote-free identity of one accepted domain proposal."""

    proposal_id: str = Field(pattern=r"^proposal_[0-9a-f]{24}$")
    candidate_occurrence_id: str = Field(pattern=r"^candidate_occurrence_[0-9a-f]{24}$")
    input_candidate_id: str = Field(pattern=r"^obs_[0-9a-f]{20}$")
    stage: ProposalStage
    paper_id: str
    source_id: str
    page: int = Field(ge=1)
    fragment_id: str | None = None
    batch_id: str | None = None
    planned_row_id: str | None = None
    raw_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_ordinal: int = Field(ge=1)

    @model_validator(mode="after")
    def stage_and_id_are_canonical(self) -> ProposalTrace:
        if self.stage is ProposalStage.LEGACY_BLOCK:
            if (
                self.fragment_id is None
                or self.batch_id is not None
                or self.planned_row_id is not None
            ):
                raise ValueError("legacy proposal requires only a fragment identity")
        elif self.batch_id is None or self.planned_row_id is None:
            raise ValueError("row proposal requires batch and planned-row identities")
        if self.proposal_id != proposal_id(
            stage=self.stage,
            paper_id=self.paper_id,
            source_id=self.source_id,
            page=self.page,
            fragment_id=self.fragment_id,
            batch_id=self.batch_id,
            planned_row_id=self.planned_row_id,
            raw_payload_sha256=self.raw_payload_sha256,
            proposal_ordinal=self.proposal_ordinal,
        ):
            raise ValueError("proposal id does not match its complete extraction identity")
        if self.candidate_occurrence_id != candidate_occurrence_id(
            proposal_id=self.proposal_id,
            input_candidate_id=self.input_candidate_id,
        ):
            raise ValueError("candidate occurrence id does not match its proposal and candidate")
        return self


def _hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def value_sha256(payload: object) -> str:
    return _hash(payload)


def proposal_id(
    *,
    stage: ProposalStage,
    paper_id: str,
    source_id: str,
    page: int,
    fragment_id: str | None,
    batch_id: str | None,
    planned_row_id: str | None,
    raw_payload_sha256: str,
    proposal_ordinal: int,
) -> str:
    return (
        "proposal_"
        + _hash(
            {
                "stage": stage,
                "paper_id": paper_id,
                "source_id": source_id,
                "page": page,
                "fragment_id": fragment_id,
                "batch_id": batch_id,
                "planned_row_id": planned_row_id,
                "raw_payload_sha256": raw_payload_sha256,
                "proposal_ordinal": proposal_ordinal,
            }
        )[:24]
    )


def candidate_occurrence_id(*, proposal_id: str, input_candidate_id: str) -> str:
    return (
        "candidate_occurrence_"
        + _hash(
            {
                "proposal_id": proposal_id,
                "input_candidate_id": input_candidate_id,
            }
        )[:24]
    )


def make_proposal_trace(
    *,
    stage: ProposalStage,
    paper_id: str,
    source_id: str,
    page: int,
    raw_payload_sha256: str,
    proposal_ordinal: int,
    input_candidate_id: str,
    fragment_id: str | None = None,
    batch_id: str | None = None,
    planned_row_id: str | None = None,
) -> ProposalTrace:
    identity = proposal_id(
        stage=stage,
        paper_id=paper_id,
        source_id=source_id,
        page=page,
        fragment_id=fragment_id,
        batch_id=batch_id,
        planned_row_id=planned_row_id,
        raw_payload_sha256=raw_payload_sha256,
        proposal_ordinal=proposal_ordinal,
    )
    return ProposalTrace(
        proposal_id=identity,
        candidate_occurrence_id=candidate_occurrence_id(
            proposal_id=identity,
            input_candidate_id=input_candidate_id,
        ),
        input_candidate_id=input_candidate_id,
        stage=stage,
        paper_id=paper_id,
        source_id=source_id,
        page=page,
        fragment_id=fragment_id,
        batch_id=batch_id,
        planned_row_id=planned_row_id,
        raw_payload_sha256=raw_payload_sha256,
        proposal_ordinal=proposal_ordinal,
    )
