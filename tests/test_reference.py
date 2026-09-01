from __future__ import annotations

from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
)
from proceedings_to_eee.domain.status import ActorRole, ClaimType, EvidenceKind
from proceedings_to_eee.evaluation.reference_score import score_reference
from proceedings_to_eee.reference import (
    AnnotationCoverage,
    EvidencePurpose,
    PaperReference,
    ReferenceActor,
    ReferenceEvidence,
    ReferenceObservation,
)


def _reference() -> PaperReference:
    return PaperReference(
        paper_id="synthetic-audit-study",
        source_sha256="a" * 64,
        annotation_protocol="protocol/0.2",
        annotation_status="reviewed",
        coverage=AnnotationCoverage(
            fully_annotated_labels=["Table 2"],
            inclusion_rule="Every numeric metric cell in Table 2.",
            exclusion_rule="Headers and sample sizes are context, not results.",
        ),
        evidence=[
            ReferenceEvidence(
                evidence_id="ev-result",
                purpose=EvidencePurpose.RESULT,
                page=7,
                kind=EvidenceKind.TABLE,
                label="Table 2",
                row="Atlas Moderation API · Synthetic Speech Set",
                column="AUC",
                exact_quote="Atlas Moderation API  61.3  74.6%  58.2",
            ),
            ReferenceEvidence(
                evidence_id="ev-header",
                purpose=EvidencePurpose.TABLE_HEADER,
                page=7,
                kind=EvidenceKind.TABLE,
                label="Table 2",
                exact_quote="RATE INDEX DELTA FLOOR CEILING",
            ),
        ],
        observations=[
            ReferenceObservation(
                reference_id="ref-1",
                claim_type=ClaimType.PRIMARY_RESULT,
                actors=[
                    ReferenceActor(
                        role=ActorRole.EVALUATED_SYSTEM,
                        raw_name="Atlas Moderation API",
                        canonical_id="example/atlas-moderation-api",
                    )
                ],
                scope=ObservationScope(dataset_raw="Synthetic Speech Set", split="test"),
                metric=MetricSpec(
                    raw_name="AUC",
                    canonical_id="auroc",
                    unit="percent",
                    lower_is_better=False,
                ),
                value=ReportedValue(raw="74.6%", numeric=74.6, unit="percent"),
                result_evidence_ids=["ev-result"],
                context_evidence_ids=["ev-header"],
                expected_missing_fields=["evaluated_system.version", "evaluation_date"],
            )
        ],
    )


def _unmatched_primary(*, observation_id: str, label: str) -> CandidateObservation:
    return CandidateObservation(
        observation_id=observation_id,
        paper_id="synthetic-audit-study",
        claim_type=ClaimType.PRIMARY_RESULT,
        roles=[
            RoleAssignment(
                role=ActorRole.EVALUATED_SYSTEM,
                raw_name="Different API",
                confidence=1.0,
            )
        ],
        scope=ObservationScope(dataset_raw="Different dataset"),
        metric=MetricSpec(raw_name="Precision", unit="percent"),
        value=ReportedValue(raw="12.3%", numeric=12.3, unit="percent"),
        evidence=[
            EvidenceAnchor(
                source_id="src_paper",
                page=7,
                kind=EvidenceKind.TABLE,
                label=label,
                row="Different API",
                column="Precision",
                quote="Different API 12.3%",
            )
        ],
        extraction_confidence=1.0,
    )


def _candidate_from_reference(
    reference: PaperReference, expected: ReferenceObservation
) -> CandidateObservation:
    evidence_by_id = {item.evidence_id: item for item in reference.evidence}
    return CandidateObservation(
        observation_id=f"obs-{expected.reference_id}",
        paper_id=reference.paper_id,
        claim_type=expected.claim_type,
        roles=[
            RoleAssignment(
                role=actor.role,
                raw_name=actor.raw_name,
                canonical_id=actor.canonical_id,
                version=actor.version,
                provider=actor.provider,
                confidence=1.0,
            )
            for actor in expected.actors
        ],
        scope=expected.scope.model_copy(deep=True),
        metric=expected.metric.model_copy(deep=True),
        value=expected.value.model_copy(deep=True),
        evidence=[
            EvidenceAnchor(
                source_id="reference-source",
                page=evidence_by_id[evidence_id].page,
                kind=evidence_by_id[evidence_id].kind,
                label=evidence_by_id[evidence_id].label,
                row=evidence_by_id[evidence_id].row,
                column=evidence_by_id[evidence_id].column,
                quote=evidence_by_id[evidence_id].exact_quote,
            )
            for evidence_id in expected.result_evidence_ids
        ],
        extraction_confidence=1.0,
    )


def test_reference_scores_joint_semantics(eligible_candidate: CandidateObservation) -> None:
    result = score_reference(_reference(), [eligible_candidate])
    assert result["detection"]["recall"] == 1.0
    assert result["detection"]["precision"] == 1.0
    assert result["primary_candidates_total"] == 1
    assert result["primary_candidates_in_coverage"] == 1
    assert result["precision_basis"] == 1
    assert result["recall_basis"] == 1
    assert result["field_matching_basis"] == 1
    assert result["coverage"]["recall_scope"] == "annotated_reference_observations"
    assert result["coverage"]["precision_scope"] == "fully_annotated_labels"
    assert result["field_accuracy"]["joint_semantics"] == 1.0
    assert result["field_accuracy"]["page"] == 1.0
    assert result["field_accuracy"]["evidence_structure"] == 1.0
    assert result["field_accuracy"]["missingness"] == 1.0
    assert result["claim_type_classification"]["macro_f1"] == 1.0


def test_precision_only_penalizes_unmatched_primary_candidates_in_coverage(
    eligible_candidate: CandidateObservation,
) -> None:
    covered_extra = _unmatched_primary(observation_id="obs-covered-extra", label="Table 2")
    outside_extra = _unmatched_primary(observation_id="obs-outside-extra", label="Table 9")

    result = score_reference(_reference(), [eligible_candidate, covered_extra, outside_extra])

    assert result["detection"]["recall"] == 1.0
    assert result["field_accuracy"]["joint_semantics"] == 1.0
    assert result["primary_candidates_total"] == 3
    assert result["primary_candidates_in_coverage"] == 2
    assert result["primary_candidates_out_of_coverage"] == 1
    assert result["precision_basis"] == 2
    assert result["detection"]["precision_true_positives"] == 1
    assert result["detection"]["false_positives"] == 1
    assert result["detection"]["precision"] == 0.5
    assert result["unmatched_candidate_ids_in_coverage"] == ["obs-covered-extra"]
    assert result["unmatched_candidate_ids_out_of_coverage"] == ["obs-outside-extra"]
    assert result["unmatched_primary_candidate_ids_in_coverage"] == ["obs-covered-extra"]
    assert result["unmatched_primary_candidate_ids_out_of_coverage"] == ["obs-outside-extra"]


def test_granular_coverage_label_does_not_cover_other_rows(
    eligible_candidate: CandidateObservation,
) -> None:
    reference = _reference().model_copy(
        update={
            "coverage": AnnotationCoverage(
                fully_annotated_labels=["Table 2 · Atlas Moderation API row"],
                inclusion_rule="Only the Atlas Moderation API row.",
                exclusion_rule="Other rows are outside precision coverage.",
            )
        }
    )
    other_row = _unmatched_primary(observation_id="obs-other-row", label="Table 2")

    result = score_reference(reference, [eligible_candidate, other_row])

    assert result["primary_candidates_total"] == 2
    assert result["primary_candidates_in_coverage"] == 1
    assert result["precision_basis"] == 1
    assert result["detection"]["false_positives"] == 0
    assert result["detection"]["precision"] == 1.0
    assert result["unmatched_candidate_ids_out_of_coverage"] == ["obs-other-row"]


def test_non_primary_candidate_inside_coverage_is_not_a_false_positive(
    eligible_candidate: CandidateObservation,
) -> None:
    secondary = _unmatched_primary(observation_id="obs-secondary", label="Table 2").model_copy(
        update={"claim_type": ClaimType.SECONDARY_CLAIM}
    )

    result = score_reference(_reference(), [eligible_candidate, secondary])

    assert result["primary_candidates_total"] == 1
    assert result["precision_basis"] == 1
    assert result["detection"]["false_positives"] == 0
    assert result["unmatched_candidate_ids_in_coverage"] == ["obs-secondary"]
    assert result["unmatched_primary_candidate_ids_in_coverage"] == []


def test_empty_complete_coverage_does_not_invent_false_positives(
    eligible_candidate: CandidateObservation,
) -> None:
    reference = _reference().model_copy(
        update={
            "coverage": AnnotationCoverage(
                inclusion_rule="Sampled reference only.",
                exclusion_rule="No region is exhaustively annotated.",
            )
        }
    )
    extra = _unmatched_primary(observation_id="obs-unscored", label="Table 2")

    result = score_reference(reference, [eligible_candidate, extra])

    assert result["detection"]["recall"] == 1.0
    assert result["primary_candidates_total"] == 2
    assert result["primary_candidates_in_coverage"] == 0
    assert result["precision_basis"] == 0
    assert result["detection"]["precision_defined"] is False
    assert result["detection"]["precision"] is None
    assert result["detection"]["f1"] is None
    assert result["detection"]["false_positives"] == 0
    assert result["unmatched_candidate_ids_out_of_coverage"] == ["obs-unscored"]


def test_synthetic_coverage_contains_its_reference_observations() -> None:
    reference = _reference()
    candidates = [
        _candidate_from_reference(reference, expected) for expected in reference.observations
    ]
    result = score_reference(reference, candidates)
    expected_primary = sum(
        item.claim_type == ClaimType.PRIMARY_RESULT for item in reference.observations
    )
    assert result["primary_candidates_in_coverage"] == expected_primary
    assert result["precision_basis"] == expected_primary
    assert result["detection"]["false_positives"] == 0


def test_reference_annotation_includes_context_layers() -> None:
    reference = _reference()
    purposes = {item.purpose for item in reference.evidence}
    assert EvidencePurpose.RESULT in purposes
    assert EvidencePurpose.TABLE_HEADER in purposes
