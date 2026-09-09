"""Regression tests for observation-ID collision resolution.

Collision resolution must preserve distinct values and descriptive fields, and every
returned candidate's `observation_id` must equal its `stable_id()`.
"""

from __future__ import annotations

from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
)
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    EvidenceKind,
)
from proceedings_to_eee.validation.candidates import (
    DeduplicationKind,
    _collapse_residual_id_collisions,
)

PAPER_ID = "synthetic-collisions"
SOURCE_ID = "src_synthetic_collisions"


def _anchor(quote: str) -> EvidenceAnchor:
    return EvidenceAnchor(
        source_id=SOURCE_ID,
        page=4,
        kind=EvidenceKind.TABLE,
        quote=quote,
    )


def _result(value: float, quote: str, **overrides: object) -> CandidateObservation:
    payload: dict[str, object] = {
        "paper_id": PAPER_ID,
        "claim_type": ClaimType.PRIMARY_RESULT,
        "roles": [RoleAssignment(role=ActorRole.EVALUATED_SYSTEM, raw_name="SysA")],
        "scope": ObservationScope(dataset_raw="SynthSet"),
        "metric": MetricSpec(raw_name="F1", unit="proportion"),
        "value": ReportedValue(raw=str(value), numeric=value, unit="proportion"),
        "evidence": [_anchor(quote)],
    }
    payload.update(overrides)
    return CandidateObservation(**payload)


def _metadata(quote: str) -> CandidateObservation:
    return CandidateObservation(
        paper_id=PAPER_ID,
        claim_type=ClaimType.METHOD_METADATA,
        roles=[],
        evidence=[_anchor(quote)],
    )


def test_differing_values_are_never_collapsed() -> None:
    """A stale shared ID is repaired by re-stamping, not by merging two results."""

    first = _result(0.81, "SysA 0.81")
    second = _result(0.42, "SysA 0.42").model_copy(update={"observation_id": first.observation_id})
    assert first.observation_id == second.observation_id

    resolved, kinds = _collapse_residual_id_collisions(
        [first, second], {str(first.observation_id): DeduplicationKind.SINGLETON}
    )

    assert len(resolved) == 2
    assert {candidate.value.numeric for candidate in resolved} == {0.81, 0.42}
    assert len({str(candidate.observation_id) for candidate in resolved}) == 2
    for candidate in resolved:
        assert candidate.observation_id == candidate.stable_id()
        assert str(candidate.observation_id) in kinds


def test_collapse_records_the_discarded_operationalization() -> None:
    """Descriptive fields sit outside stable_id(), so a collapse must say what it drops."""

    kept = _metadata("annotators agreed on the guideline").model_copy(
        update={"operationalization": "three annotators, majority vote"}
    )
    losing = _metadata("annotators agreed on the guideline").model_copy(
        update={"operationalization": "two annotators, adjudicated"}
    )
    assert kept.stable_id() == losing.stable_id()

    resolved, kinds = _collapse_residual_id_collisions(
        [kept, losing], {str(kept.observation_id): DeduplicationKind.SEMANTIC}
    )

    assert len(resolved) == 1
    collapsed = resolved[0]
    notes = " | ".join(collapsed.notes)
    assert "two annotators, adjudicated" in notes
    assert "operationalization" in notes
    assert kinds[str(collapsed.observation_id)] is DeduplicationKind.SEMANTIC


def test_collapsed_candidate_id_equals_stable_id() -> None:
    """The census collision: same anchor, different quotation, one observation."""

    first = _metadata("Perspective API scores were collected in 2021")
    second = _metadata("scores were collected in 2021 with Perspective API")
    assert first.observation_id == second.observation_id

    resolved, _ = _collapse_residual_id_collisions(
        [first, second], {str(first.observation_id): DeduplicationKind.SEMANTIC}
    )

    assert len(resolved) == 1
    collapsed = resolved[0]
    assert collapsed.observation_id == collapsed.stable_id()
    assert len(collapsed.evidence) == 2


def test_unique_ids_are_left_untouched() -> None:
    """No collision means no re-stamping, so IDs stay byte-identical."""

    first = _result(0.81, "SysA 0.81")
    second = _result(0.42, "SysA 0.42")
    kinds = {
        str(first.observation_id): DeduplicationKind.SINGLETON,
        str(second.observation_id): DeduplicationKind.SINGLETON,
    }

    resolved, resolved_kinds = _collapse_residual_id_collisions([first, second], kinds)

    assert resolved == [first, second]
    assert resolved_kinds == kinds


def test_proposals_that_disagree_on_an_unsupported_field_do_not_abort_the_paper() -> None:
    """Two physical-cell duplicates can disagree on an unsupported field.

    The merge raised the field to CONFLICT while the union of their sources was empty,
    and `CandidateFieldProvenance` rejects that, because BOUND, AMBIGUOUS and CONFLICT
    all assert evidence. One invalid field ended the whole paper.
    """

    from proceedings_to_eee.domain.provenance import (
        CandidateField,
        CandidateFieldProvenance,
        FieldBindingStatus,
    )
    from proceedings_to_eee.validation.candidates import _merge_matching_field_provenance

    def provenance(candidate: CandidateObservation, setting_hash: str) -> CandidateObservation:
        expected = candidate.field_value_sha256s()
        return candidate.model_copy(
            update={
                "field_provenance": [
                    CandidateFieldProvenance(
                        field=field,
                        value_sha256=(
                            setting_hash if field is CandidateField.SETTING else expected[field]
                        ),
                        status=FieldBindingStatus.UNSUPPORTED,
                        reason="field_not_supported_by_evidence_quotes",
                    )
                    for field in CandidateField
                ]
            }
        )

    first = provenance(_result(0.81, "SysA 0.81"), "a" * 64)
    second = provenance(_result(0.81, "SysA 0.81 again"), "b" * 64)

    merged = _merge_matching_field_provenance([first, second], first)

    setting = next(item for item in merged if item.field is CandidateField.SETTING)
    assert setting.status is FieldBindingStatus.UNSUPPORTED
    assert setting.sources == []
    assert setting.alternate_value_sha256s == []
    # The disagreement is recorded rather than lost.
    assert setting.reason is not None
    assert "disagree" in setting.reason
