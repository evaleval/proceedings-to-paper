"""Deterministic quote-free lineage from extraction proposals to final outcomes."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.domain.export_provenance import (
    ExportCompositionProvenance,
    ExportProvenanceMode,
)
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.provenance import CandidateFieldProvenance, ProposalTrace
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes


class LineageMergeKind(StrEnum):
    SINGLETON = "singleton"
    SEMANTIC = "semantic"
    PHYSICAL_CELL = "physical_cell"


class LineageOutcome(StrEnum):
    EXPORTED = "exported"
    INVALID_EEE = "invalid_eee"
    NEEDS_REVIEW = "needs_review"
    NOT_ELIGIBLE = "not_eligible"
    ELIGIBLE_NO_EEE = "eligible_no_eee"


class LineageProposalRecord(StrictModel):
    trace: ProposalTrace
    final_observation_id: str = Field(pattern=r"^obs_[0-9a-f]{20}$")


class LineageCandidateRecord(StrictModel):
    final_observation_id: str = Field(pattern=r"^obs_[0-9a-f]{20}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_ids: list[str] = Field(min_length=1)
    candidate_occurrence_ids: list[str] = Field(min_length=1)
    input_candidate_ids: list[str] = Field(min_length=1)
    merge_kind: LineageMergeKind
    export_status: str
    outcome: LineageOutcome
    evaluation_ids: list[str] = Field(default_factory=list)
    field_provenance: list[CandidateFieldProvenance]
    export_authorization_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuple_gate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verifier_gate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_trace_partition(self) -> LineageCandidateRecord:
        lengths = {
            len(self.proposal_ids),
            len(self.candidate_occurrence_ids),
            len(self.input_candidate_ids),
        }
        if len(lengths) != 1:
            raise ValueError("candidate lineage trace columns have different lengths")
        if len(self.proposal_ids) != len(set(self.proposal_ids)):
            raise ValueError("candidate lineage repeats a proposal")
        if len(self.candidate_occurrence_ids) != len(set(self.candidate_occurrence_ids)):
            raise ValueError("candidate lineage repeats a candidate occurrence")
        if self.evaluation_ids != sorted(set(self.evaluation_ids)):
            raise ValueError("candidate lineage evaluation ids are not canonical")
        return self


class LineageCounts(StrictModel):
    proposals: int = Field(ge=0)
    candidate_occurrences: int = Field(ge=0)
    final_candidates: int = Field(ge=0)
    singleton_candidates: int = Field(ge=0)
    merged_candidates: int = Field(ge=0)
    exported: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    not_eligible: int = Field(ge=0)
    invalid_eee: int = Field(ge=0)
    eligible_no_eee: int = Field(ge=0)


class CandidateLineageArtifact(StrictModel):
    schema_version: str = "candidate-lineage/0.1"
    paper_id: str
    observations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_provenance_mode: ExportProvenanceMode | None = None
    export_composition_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuple_sidecar_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verifier_gate_required: bool | None = None
    verifier_sidecar_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    proposals: list[LineageProposalRecord]
    candidates: list[LineageCandidateRecord]
    counts: LineageCounts

    @model_validator(mode="after")
    def complete_bijections_and_counts(self) -> CandidateLineageArtifact:
        if (self.export_provenance_mode is None) != (self.export_composition_sha256 is None):
            raise ValueError("candidate lineage composition binding is incomplete")
        if self.export_provenance_mode is not None:
            if self.export_provenance_mode.has_tuple_audit != (
                self.tuple_sidecar_sha256 is not None
            ):
                raise ValueError("candidate lineage tuple-sidecar binding is inconsistent")
            if not self.export_provenance_mode.has_tuple_audit and any(
                item.tuple_gate_sha256 is not None for item in self.candidates
            ):
                raise ValueError("legacy candidate lineage cannot claim tuple gates")
            if self.export_provenance_mode.has_tuple_audit and any(
                item.export_authorization_sha256 is not None and item.tuple_gate_sha256 is None
                for item in self.candidates
            ):
                raise ValueError("tuple-gated export authorization lacks candidate gate lineage")
            if self.verifier_gate_required is None:
                raise ValueError("candidate lineage lacks an explicit verifier-gate mode")
            if (
                self.export_provenance_mode is ExportProvenanceMode.TUPLE_GATED_PRODUCTION
                and self.verifier_gate_required is not True
            ):
                raise ValueError("automatic production lineage requires verifier acceptance")
            if (
                self.export_provenance_mode is ExportProvenanceMode.TUPLE_GATED_UNVERIFIED
                and self.verifier_gate_required is not False
            ):
                raise ValueError("tuple-only lineage cannot claim verified production")
            if self.verifier_gate_required != (self.verifier_sidecar_sha256 is not None):
                raise ValueError("candidate lineage verifier-sidecar binding is inconsistent")
            if not self.verifier_gate_required and any(
                item.verifier_gate_sha256 is not None for item in self.candidates
            ):
                raise ValueError("verifier-disabled lineage cannot claim verifier gates")
            if self.verifier_gate_required and any(
                item.export_authorization_sha256 is not None and item.verifier_gate_sha256 is None
                for item in self.candidates
            ):
                raise ValueError("verifier-gated export lacks candidate verifier lineage")
        if self.proposals != sorted(
            self.proposals,
            key=lambda item: item.trace.proposal_id,
        ):
            raise ValueError("lineage proposals are not in canonical order")
        if self.candidates != sorted(
            self.candidates,
            key=lambda item: item.final_observation_id,
        ):
            raise ValueError("lineage candidates are not in canonical order")
        proposal_ids = [item.trace.proposal_id for item in self.proposals]
        occurrence_ids = [item.trace.candidate_occurrence_id for item in self.proposals]
        candidate_ids = [item.final_observation_id for item in self.candidates]
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("lineage artifact repeats a proposal")
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise ValueError("lineage artifact repeats a candidate occurrence")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("lineage artifact repeats a final candidate")
        proposal_owner = {
            item.trace.proposal_id: item.final_observation_id for item in self.proposals
        }
        candidate_proposals = {
            proposal_id: candidate.final_observation_id
            for candidate in self.candidates
            for proposal_id in candidate.proposal_ids
        }
        if proposal_owner != candidate_proposals:
            raise ValueError("lineage proposals and candidate owners are not exact inverses")
        expected = LineageCounts(
            proposals=len(self.proposals),
            candidate_occurrences=len(occurrence_ids),
            final_candidates=len(self.candidates),
            singleton_candidates=sum(
                item.merge_kind is LineageMergeKind.SINGLETON for item in self.candidates
            ),
            merged_candidates=sum(
                item.merge_kind is not LineageMergeKind.SINGLETON for item in self.candidates
            ),
            exported=sum(item.outcome is LineageOutcome.EXPORTED for item in self.candidates),
            needs_review=sum(
                item.outcome is LineageOutcome.NEEDS_REVIEW for item in self.candidates
            ),
            not_eligible=sum(
                item.outcome is LineageOutcome.NOT_ELIGIBLE for item in self.candidates
            ),
            invalid_eee=sum(item.outcome is LineageOutcome.INVALID_EEE for item in self.candidates),
            eligible_no_eee=sum(
                item.outcome is LineageOutcome.ELIGIBLE_NO_EEE for item in self.candidates
            ),
        )
        if self.counts != expected:
            raise ValueError("lineage counts do not match its exact partitions")
        return self


def _result_owners(records: list[dict[str, object]]) -> dict[str, set[str]]:
    owners: dict[str, set[str]] = {}
    for record in records:
        evaluation_id = str(record["evaluation_id"])
        for result in record["evaluation_results"]:  # type: ignore[index]
            observation_id = str(result["evaluation_result_id"])
            owners.setdefault(observation_id, set()).add(evaluation_id)
    return owners


def build_candidate_lineage(
    *,
    paper_id: str,
    candidates: list[CandidateObservation],
    merge_kinds: dict[str, str],
    observations_sha256: str,
    valid_records: list[dict[str, object]],
    invalid_records: list[dict[str, object]],
    export_provenance: ExportCompositionProvenance,
    tuple_gates: dict[str, str],
    verifier_gates: dict[str, str],
) -> CandidateLineageArtifact:
    """Build one total proposal/candidate partition without copying source evidence."""

    valid_owners = _result_owners(valid_records)
    invalid_owners = _result_owners(invalid_records)
    authorization_by_id = {item.observation_id: item for item in export_provenance.authorizations}
    proposals: list[LineageProposalRecord] = []
    records: list[LineageCandidateRecord] = []
    for candidate in candidates:
        observation_id = str(candidate.observation_id)
        if candidate.paper_id != paper_id:
            raise ValueError("candidate lineage contains another paper")
        if (
            candidate.schema_version == "candidate-observation/0.3"
            and not candidate.proposal_traces
        ):
            raise ValueError("new-schema candidate has no extraction proposal lineage")
        traces = sorted(candidate.proposal_traces, key=lambda trace: trace.proposal_id)
        if not traces:
            raise ValueError("candidate lineage cannot account an untraced candidate")
        proposals.extend(
            LineageProposalRecord(trace=trace, final_observation_id=observation_id)
            for trace in traces
        )
        if observation_id in valid_owners and observation_id in invalid_owners:
            raise ValueError("candidate appears in both valid and invalid EEE records")
        if observation_id in valid_owners:
            outcome = LineageOutcome.EXPORTED
            evaluation_ids = sorted(valid_owners[observation_id])
        elif observation_id in invalid_owners:
            outcome = LineageOutcome.INVALID_EEE
            evaluation_ids = sorted(invalid_owners[observation_id])
        elif candidate.export_status.value == "needs_review":
            outcome = LineageOutcome.NEEDS_REVIEW
            evaluation_ids = []
        elif candidate.export_status.value == "not_eligible":
            outcome = LineageOutcome.NOT_ELIGIBLE
            evaluation_ids = []
        else:
            outcome = LineageOutcome.ELIGIBLE_NO_EEE
            evaluation_ids = []
        records.append(
            LineageCandidateRecord(
                final_observation_id=observation_id,
                candidate_sha256=sha256_bytes(canonical_json_bytes(candidate)),
                proposal_ids=[trace.proposal_id for trace in traces],
                candidate_occurrence_ids=[trace.candidate_occurrence_id for trace in traces],
                input_candidate_ids=[trace.input_candidate_id for trace in traces],
                merge_kind=LineageMergeKind(merge_kinds[observation_id]),
                export_status=candidate.export_status.value,
                outcome=outcome,
                evaluation_ids=evaluation_ids,
                field_provenance=candidate.field_provenance,
                export_authorization_sha256=(
                    sha256_bytes(canonical_json_bytes(authorization_by_id[observation_id]))
                    if observation_id in authorization_by_id
                    else None
                ),
                tuple_gate_sha256=tuple_gates.get(observation_id),
                verifier_gate_sha256=verifier_gates.get(observation_id),
            )
        )
    proposals.sort(key=lambda item: item.trace.proposal_id)
    records.sort(key=lambda item: item.final_observation_id)
    return CandidateLineageArtifact(
        paper_id=paper_id,
        observations_sha256=observations_sha256,
        export_provenance_mode=export_provenance.mode,
        export_composition_sha256=export_provenance.sha256,
        tuple_sidecar_sha256=export_provenance.tuple_sidecar_sha256,
        verifier_gate_required=export_provenance.verifier_gate_required,
        verifier_sidecar_sha256=export_provenance.verifier_sidecar_sha256,
        proposals=proposals,
        candidates=records,
        counts=LineageCounts(
            proposals=len(proposals),
            candidate_occurrences=len(proposals),
            final_candidates=len(records),
            singleton_candidates=sum(
                item.merge_kind is LineageMergeKind.SINGLETON for item in records
            ),
            merged_candidates=sum(
                item.merge_kind is not LineageMergeKind.SINGLETON for item in records
            ),
            exported=sum(item.outcome is LineageOutcome.EXPORTED for item in records),
            needs_review=sum(item.outcome is LineageOutcome.NEEDS_REVIEW for item in records),
            not_eligible=sum(item.outcome is LineageOutcome.NOT_ELIGIBLE for item in records),
            invalid_eee=sum(item.outcome is LineageOutcome.INVALID_EEE for item in records),
            eligible_no_eee=sum(item.outcome is LineageOutcome.ELIGIBLE_NO_EEE for item in records),
        ),
    )
