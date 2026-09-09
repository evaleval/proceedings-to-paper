from __future__ import annotations

from proceedings_to_eee.evaluation.corpus_score import aggregate_reference_scores


def _score(
    *,
    precision_basis: int,
    precision_true_positives: int,
    true_positives: int,
    false_negatives: int,
    field_value: float,
    false_primary: int = 0,
    controls_total: int = 2,
    matched_controls: int = 1,
    examined_controls: int | None = None,
    passed_by_abstention: int = 0,
    observable_references: int | None = None,
    paper_id: str | None = None,
) -> dict[str, object]:
    references = true_positives + false_negatives
    precision = precision_true_positives / precision_basis if precision_basis else None
    recall = true_positives / references if references else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and precision + recall
        else None
    )
    result = {
        "recall_basis": references,
        "field_matching_basis": references,
        "precision_basis": precision_basis,
        "detection": {
            "recall_basis": references,
            "true_positives": true_positives,
            "precision_true_positives": precision_true_positives,
            "false_positives": precision_basis - precision_true_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "field_accuracy": {
            field: field_value
            for field in (
                "claim_type",
                "system",
                "dataset",
                "metric",
                "value",
                "unit",
                "slice",
                "page",
                "evidence_kind",
                "evidence_label",
                "evidence_row",
                "evidence_column",
                "evidence_structure",
                "evidence_supported",
                "missingness",
                "joint_semantics",
            )
        },
        "matches": [
            {
                "value": True,
                "unit": True,
                "page": True,
                "evidence_supported": True,
                "expected_claim_type": "primary_result",
                "actual_claim_type": "primary_result",
            }
            for _ in range(references)
        ],
        "negative_control_safety": {
            "controls_total": controls_total,
            "control_ids": [f"control-{index}" for index in range(controls_total)],
            "matched_control_count": matched_controls,
            "matched_control_ids": [f"control-{index}" for index in range(matched_controls)],
            "matched_candidate_count": matched_controls,
            "false_primary_count": false_primary,
            "false_primary_export_count": false_primary,
            "examined_control_ids": [
                f"control-{index}"
                for index in range(
                    matched_controls if examined_controls is None else examined_controls
                )
            ],
            "passed_by_abstention_count": passed_by_abstention,
            "matches": [],
        },
        "input_observability": (
            {
                "status": "measured",
                "reference_observations": references,
                "observable_reference_observations": observable_references,
                "unobservable_reference_observations": references - observable_references,
                "observation_coverage": (
                    observable_references / references if references else None
                ),
                "model_conditional_detection": {
                    "true_positives": min(true_positives, observable_references),
                    "false_negatives": max(0, observable_references - true_positives),
                    "recall_basis": observable_references,
                    "recall": (
                        min(true_positives, observable_references) / observable_references
                        if observable_references
                        else None
                    ),
                },
            }
            if observable_references is not None
            else {"status": "not_assessed"}
        ),
    }
    if paper_id is not None:
        result["paper_id"] = paper_id
    return result


def test_aggregate_reference_scores_preserves_coverage_bases_and_gates() -> None:
    result = aggregate_reference_scores(
        [
            _score(
                precision_basis=2,
                precision_true_positives=2,
                true_positives=2,
                false_negatives=0,
                field_value=1.0,
            ),
            _score(
                precision_basis=0,
                precision_true_positives=0,
                true_positives=1,
                false_negatives=1,
                field_value=0.5,
            ),
        ]
    )

    assert result["bases"] == {
        "reference_observations": 4,
        "field_matching": 4,
        "precision_candidates_in_fully_annotated_regions": 2,
    }
    assert result["detection"]["precision"] == 1.0
    assert result["detection"]["recall"] == 0.75
    assert result["detection"]["precision_defined_papers"] == 1
    assert result["field_accuracy"]["joint_semantics"] == 0.75
    assert result["negative_control_safety"]["controls_total"] == 4
    assert result["negative_control_safety"]["controls_matched"] == 2
    assert result["negative_control_safety"]["control_match_coverage"] == 0.5
    assert result["quality_gates"]["candidate_detection_recall"]["status"] == "failed"
    assert result["quality_gates"]["false_primary_exports"]["status"] == "not_measured"
    assert result["quality_gates"]["claim_type_macro_f1"]["status"] == "not_measured"


def test_aggregate_reference_scores_exposes_false_primary_failure() -> None:
    result = aggregate_reference_scores(
        [
            _score(
                precision_basis=1,
                precision_true_positives=1,
                true_positives=1,
                false_negatives=0,
                field_value=1.0,
                false_primary=1,
            )
        ]
    )

    assert result["negative_control_safety"]["false_primary_count"] == 1
    assert not result["negative_control_safety"]["zero_false_primary_gate_passed"]
    assert result["quality_gates"]["false_primary_controls"]["status"] == "failed"


def test_negative_control_gate_is_not_measured_without_a_matched_control() -> None:
    result = aggregate_reference_scores(
        [
            _score(
                precision_basis=1,
                precision_true_positives=1,
                true_positives=1,
                false_negatives=0,
                field_value=1.0,
                controls_total=3,
                matched_controls=0,
            )
        ]
    )

    safety = result["negative_control_safety"]
    assert safety["controls_total"] == 3
    assert safety["controls_matched"] == 0
    assert safety["control_match_coverage"] == 0.0
    assert safety["measurement_status"] == "not_measured"
    assert safety["zero_false_primary_gate_passed"] is None
    assert result["quality_gates"]["false_primary_controls"]["status"] == "not_measured"
    assert result["quality_gates"]["false_primary_exports"]["status"] == "not_measured"


def test_official_control_examination_is_reported_separately_from_matching() -> None:
    """The repaired report must preserve the audited 17/21 examination result."""

    repetitions = [
        _score(
            precision_basis=0,
            precision_true_positives=0,
            true_positives=0,
            false_negatives=0,
            field_value=0.0,
            controls_total=21,
            matched_controls=0,
            examined_controls=17,
            passed_by_abstention=17,
            paper_id="official-control-frame",
        )
        for _ in range(2)
    ]
    result = aggregate_reference_scores(repetitions)

    safety = result["negative_control_safety"]
    assert safety["controls_matched"] == 0
    assert safety["control_match_coverage"] == 0.0
    assert safety["controls_examined"] == 17
    assert safety["controls_not_examined"] == 4
    assert safety["control_examination_coverage"] == 0.809524
    assert safety["passed_by_abstention_count"] == 17
    assert safety["control_trials"] == {
        "total": 42,
        "matched": 0,
        "examined": 34,
        "passed_by_abstention": 34,
    }
    assert safety["measurement_status"] == "partially_measured"
    assert safety["zero_false_primary_gate_passed"] is None
    assert result["quality_gates"]["false_primary_controls"]["status"] == "not_measured"


def test_zero_reference_paper_does_not_depress_macro_recall() -> None:
    zero_reference = _score(
        precision_basis=1,
        precision_true_positives=0,
        true_positives=0,
        false_negatives=0,
        field_value=0.0,
        controls_total=3,
        matched_controls=0,
    )
    zero_reference["detection"]["f1"] = 0.0
    result = aggregate_reference_scores(
        [
            _score(
                precision_basis=1,
                precision_true_positives=1,
                true_positives=1,
                false_negatives=0,
                field_value=1.0,
            ),
            zero_reference,
        ]
    )

    assert result["detection"]["macro_recall"] == 1.0
    assert result["detection"]["macro_f1"] == 1.0


def test_input_observability_keeps_pipeline_and_model_denominators_explicit() -> None:
    result = aggregate_reference_scores(
        [
            _score(
                precision_basis=1,
                precision_true_positives=1,
                true_positives=1,
                false_negatives=1,
                field_value=0.5,
                observable_references=1,
            )
        ]
    )

    assert result["detection"]["recall"] == 0.5
    assert result["input_observability"] == {
        "status": "measured",
        "papers_measured": 1,
        "papers_not_assessed": 0,
        "measured_reference_observations": 2,
        "reference_observations": 2,
        "observable_reference_observations": 1,
        "unobservable_reference_observations": 1,
        "observation_coverage": 0.5,
        "model_conditional_detection": {
            "true_positives": 1,
            "false_negatives": 0,
            "recall_basis": 1,
            "recall": 1.0,
        },
    }


def test_mixed_observability_assessment_never_publishes_subset_coverage() -> None:
    measured = _score(
        precision_basis=1,
        precision_true_positives=1,
        true_positives=1,
        false_negatives=0,
        field_value=1.0,
        observable_references=1,
    )
    unassessed = _score(
        precision_basis=1,
        precision_true_positives=1,
        true_positives=1,
        false_negatives=0,
        field_value=1.0,
    )

    result = aggregate_reference_scores([measured, unassessed])

    assert result["input_observability"] == {
        "status": "partially_assessed",
        "papers_measured": 1,
        "papers_not_assessed": 1,
        "measured_reference_observations": 1,
        "reference_observations": None,
        "observable_reference_observations": None,
        "unobservable_reference_observations": None,
        "observation_coverage": None,
        "model_conditional_detection": {
            "true_positives": None,
            "false_negatives": None,
            "recall_basis": None,
            "recall": None,
        },
    }
