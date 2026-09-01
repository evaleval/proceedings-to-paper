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
) -> dict[str, object]:
    references = true_positives + false_negatives
    precision = precision_true_positives / precision_basis if precision_basis else None
    recall = true_positives / references if references else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and precision + recall
        else None
    )
    return {
        "recall_basis": references,
        "field_matching_basis": references,
        "precision_basis": precision_basis,
        "detection": {
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
            "matched_control_count": matched_controls,
            "matched_candidate_count": matched_controls,
            "false_primary_count": false_primary,
            "false_primary_export_count": false_primary,
            "matches": [],
        },
    }


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
    assert result["quality_gates"]["false_primary_exports"]["status"] == "passed"
    assert result["quality_gates"]["claim_type_macro_f1"]["status"] == "passed"


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
