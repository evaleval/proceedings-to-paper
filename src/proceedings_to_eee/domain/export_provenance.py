"""Typed authorization for every canonical EEE composition path.

The authorization is deliberately separate from the candidate payload.  A model
proposal can therefore never make a candidate exportable by changing the tuple,
and legacy/manual composition remains visibly distinct from tuple-gated
production output.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from proceedings_to_eee.domain.attribution import OriginBasis, OriginExportPolicy, ReviewTier
from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes


class ExportProvenanceMode(StrEnum):
    TUPLE_GATED_PRODUCTION = "tuple_gated_production"
    TUPLE_GATED_UNVERIFIED = "tuple_gated_unverified"
    TUPLE_AUDITED_HUMAN_REVIEWED = "tuple_audited_human_reviewed"
    LEGACY_MANUAL = "legacy_manual"
    LEGACY_HUMAN_REVIEWED = "legacy_human_reviewed"

    @property
    def is_tuple_gated(self) -> bool:
        return self is ExportProvenanceMode.TUPLE_GATED_PRODUCTION

    @property
    def has_tuple_audit(self) -> bool:
        return self in {
            ExportProvenanceMode.TUPLE_GATED_PRODUCTION,
            ExportProvenanceMode.TUPLE_GATED_UNVERIFIED,
            ExportProvenanceMode.TUPLE_AUDITED_HUMAN_REVIEWED,
        }

    @property
    def is_human_reviewed(self) -> bool:
        return self in {
            ExportProvenanceMode.TUPLE_AUDITED_HUMAN_REVIEWED,
            ExportProvenanceMode.LEGACY_HUMAN_REVIEWED,
        }


class CandidateExportAuthorization(StrictModel):
    observation_id: str = Field(min_length=1)
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: What the record may claim about who produced the number. Left unset on the
    #: positive-only path, where the only bases that can appear are positive ones and
    #: leaving it unset keeps every existing composition hash byte-identical, because
    #: the hash excludes null fields.
    origin_basis: OriginBasis | None = None
    #: How much review stands behind this record. Unset means the deterministic tier.
    review_tier: ReviewTier | None = None
    tuple_gate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuple_gate_passed: bool | None = None
    verifier_gate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verifier_gate_passed: bool | None = None


class ExportCompositionProvenance(StrictModel):
    schema_version: str = "export-composition-provenance/0.1"
    mode: ExportProvenanceMode
    authorizations: list[CandidateExportAuthorization]
    #: Which origin bases this composition run was willing to export. Unset means the
    #: historical `positive_only` policy, which keeps existing hashes unchanged.
    origin_policy: OriginExportPolicy | None = None
    #: Commit the composing pipeline ran at, recorded in every record it writes.
    pipeline_git_commit: str | None = None
    tuple_sidecar_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verifier_gate_required: bool
    verifier_sidecar_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    review_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    legacy_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def exact_mode_shape(self) -> ExportCompositionProvenance:
        if self.authorizations != sorted(self.authorizations, key=lambda item: item.observation_id):
            raise ValueError("export authorizations are not in canonical order")
        ids = [item.observation_id for item in self.authorizations]
        if len(ids) != len(set(ids)):
            raise ValueError("export authorizations repeat an observation")
        if self.mode.has_tuple_audit:
            if self.tuple_sidecar_sha256 is None or self.legacy_reason is not None:
                raise ValueError("tuple-audited provenance requires only a tuple sidecar binding")
            if self.mode.is_tuple_gated and any(
                item.tuple_gate_sha256 is None or item.tuple_gate_passed is not True
                for item in self.authorizations
            ):
                raise ValueError("tuple-gated provenance requires an explicit passing gate")
            if self.mode is ExportProvenanceMode.TUPLE_AUDITED_HUMAN_REVIEWED and any(
                item.tuple_gate_sha256 is None or item.tuple_gate_passed is None
                for item in self.authorizations
            ):
                raise ValueError("tuple-audited review requires every source gate outcome")
        else:
            if self.tuple_sidecar_sha256 is not None or self.legacy_reason is None:
                raise ValueError("legacy provenance requires an explicit legacy reason")
            if any(
                item.tuple_gate_sha256 is not None or item.tuple_gate_passed is not None
                for item in self.authorizations
            ):
                raise ValueError("legacy provenance cannot contain tuple-gate claims")
        if self.mode.is_human_reviewed != (self.review_manifest_sha256 is not None):
            raise ValueError("human-reviewed provenance requires the exact review manifest")
        if self.mode is ExportProvenanceMode.TUPLE_GATED_PRODUCTION and not (
            self.verifier_gate_required
        ):
            raise ValueError("automatic production requires an accepting verifier gate")
        if self.mode is ExportProvenanceMode.TUPLE_GATED_UNVERIFIED and self.authorizations:
            raise ValueError("tuple-only unverified mode is review-only and cannot authorize EEE")
        if self.verifier_gate_required:
            if self.verifier_sidecar_sha256 is None or any(
                item.verifier_gate_sha256 is None or item.verifier_gate_passed is not True
                for item in self.authorizations
            ):
                raise ValueError("verifier-gated provenance requires an explicit passing gate")
        elif self.verifier_sidecar_sha256 is not None:
            if not self.mode.is_human_reviewed or any(
                item.verifier_gate_passed is None
                or (item.verifier_gate_passed is True and item.verifier_gate_sha256 is None)
                for item in self.authorizations
            ):
                raise ValueError("verifier-audited review requires every source gate outcome")
        elif any(
            item.verifier_gate_sha256 is not None or item.verifier_gate_passed is not None
            for item in self.authorizations
        ):
            raise ValueError("verifier-disabled provenance cannot contain verifier gates")
        return self

    @property
    def effective_origin_policy(self) -> OriginExportPolicy:
        """The policy in force, naming the historical default when none was recorded."""

        return self.origin_policy or OriginExportPolicy.POSITIVE_ONLY

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))


def candidate_export_sha256(candidate: CandidateObservation) -> str:
    """Hash export semantics, normalizing only the eligible→exported transition."""

    payload = candidate.model_dump(mode="json")
    if payload.get("export_status") in {"eligible", "exported"}:
        payload["export_status"] = "eligible"
    return sha256_bytes(canonical_json_bytes(payload))


def legacy_export_provenance(
    candidates: list[CandidateObservation],
    *,
    reason: str,
    review_manifest_sha256: str | None = None,
) -> ExportCompositionProvenance:
    """Create an explicitly non-tuple-gated manual/review authorization."""

    mode = (
        ExportProvenanceMode.LEGACY_HUMAN_REVIEWED
        if review_manifest_sha256 is not None
        else ExportProvenanceMode.LEGACY_MANUAL
    )
    return ExportCompositionProvenance(
        mode=mode,
        authorizations=_candidate_authorizations(candidates),
        verifier_gate_required=False,
        review_manifest_sha256=review_manifest_sha256,
        legacy_reason=reason,
    )


def tuple_gated_export_provenance(
    candidates: list[CandidateObservation],
    *,
    tuple_sidecar_sha256: str,
    tuple_gates: dict[str, str],
    verifier_sidecar_sha256: str | None = None,
    verifier_gates: dict[str, str] | None = None,
) -> ExportCompositionProvenance:
    """Bind already-passing deterministic gates; never infer or invent a pass."""

    authorizations = _candidate_authorizations(candidates)
    expected = {item.observation_id for item in authorizations}
    if set(tuple_gates) != expected:
        raise ValueError("tuple gates do not exactly cover export candidates")
    if verifier_sidecar_sha256 is None or verifier_gates is None:
        raise ValueError("automatic production requires verifier sidecar and candidate gates")
    if set(verifier_gates) != expected:
        raise ValueError("verifier gates do not exactly cover export candidates")
    return ExportCompositionProvenance(
        mode=ExportProvenanceMode.TUPLE_GATED_PRODUCTION,
        authorizations=[
            item.model_copy(
                update={
                    "tuple_gate_sha256": tuple_gates[item.observation_id],
                    "tuple_gate_passed": True,
                    "verifier_gate_sha256": verifier_gates[item.observation_id],
                    "verifier_gate_passed": True,
                }
            )
            for item in authorizations
        ],
        tuple_sidecar_sha256=tuple_sidecar_sha256,
        verifier_gate_required=True,
        verifier_sidecar_sha256=verifier_sidecar_sha256,
    )


def tuple_unverified_review_provenance(*, tuple_sidecar_sha256: str) -> ExportCompositionProvenance:
    """Record a tuple-only automatic stage that is forbidden to emit canonical EEE."""

    return ExportCompositionProvenance(
        mode=ExportProvenanceMode.TUPLE_GATED_UNVERIFIED,
        authorizations=[],
        tuple_sidecar_sha256=tuple_sidecar_sha256,
        verifier_gate_required=False,
    )


def tuple_audited_review_export_provenance(
    candidates: list[CandidateObservation],
    *,
    tuple_sidecar_sha256: str,
    tuple_gates: dict[str, tuple[str, bool]],
    review_manifest_sha256: str,
    verifier_sidecar_sha256: str | None = None,
    verifier_gates: dict[str, tuple[str | None, bool]] | None = None,
) -> ExportCompositionProvenance:
    """Bind source tuple outcomes while leaving export authority to human review."""

    authorizations = _candidate_authorizations(candidates)
    expected = {item.observation_id for item in authorizations}
    if set(tuple_gates) != expected:
        raise ValueError("tuple gates do not exactly cover reviewed export candidates")
    if (verifier_sidecar_sha256 is None) != (verifier_gates is None):
        raise ValueError("reviewed verifier sidecar and gates must be supplied together")
    if verifier_gates is not None and set(verifier_gates) != expected:
        raise ValueError("verifier gates do not exactly cover reviewed export candidates")
    return ExportCompositionProvenance(
        mode=ExportProvenanceMode.TUPLE_AUDITED_HUMAN_REVIEWED,
        authorizations=[
            item.model_copy(
                update={
                    "tuple_gate_sha256": tuple_gates[item.observation_id][0],
                    "tuple_gate_passed": tuple_gates[item.observation_id][1],
                    "verifier_gate_sha256": (
                        verifier_gates[item.observation_id][0]
                        if verifier_gates is not None
                        else None
                    ),
                    "verifier_gate_passed": (
                        verifier_gates[item.observation_id][1]
                        if verifier_gates is not None
                        else None
                    ),
                }
            )
            for item in authorizations
        ],
        tuple_sidecar_sha256=tuple_sidecar_sha256,
        verifier_gate_required=False,
        verifier_sidecar_sha256=verifier_sidecar_sha256,
        review_manifest_sha256=review_manifest_sha256,
    )


def _candidate_authorizations(
    candidates: list[CandidateObservation],
) -> list[CandidateExportAuthorization]:
    authorizations: list[CandidateExportAuthorization] = []
    for candidate in candidates:
        observation_id = candidate.observation_id or candidate.stable_id()
        authorizations.append(
            CandidateExportAuthorization(
                observation_id=observation_id,
                candidate_sha256=candidate_export_sha256(candidate),
            )
        )
    authorizations.sort(key=lambda item: item.observation_id)
    return authorizations
