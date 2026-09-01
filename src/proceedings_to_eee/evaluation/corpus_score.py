"""Coverage-aware aggregation of paper reference scores and quality gates."""

from __future__ import annotations

from typing import Any

from proceedings_to_eee.evaluation.reference_score import score_claim_type_pairs

FIELD_NAMES = (
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


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _gate(value: float | None, threshold: float, *, direction: str = "at_least") -> dict[str, Any]:
    if value is None:
        return {
            "status": "not_measured",
            "value": None,
            "threshold": threshold,
            "direction": direction,
        }
    passed = value >= threshold if direction == "at_least" else value <= threshold
    return {
        "status": "passed" if passed else "failed",
        "value": value,
        "threshold": threshold,
        "direction": direction,
    }


def aggregate_reference_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate explicit paper-level denominators without claiming whole-paper gold."""

    reference_basis = sum(int(score["recall_basis"]) for score in scores)
    field_basis = sum(int(score["field_matching_basis"]) for score in scores)
    precision_basis = sum(int(score["precision_basis"]) for score in scores)
    true_positives = sum(int(score["detection"]["true_positives"]) for score in scores)
    false_negatives = sum(int(score["detection"]["false_negatives"]) for score in scores)
    false_positives = sum(int(score["detection"]["false_positives"]) for score in scores)
    precision_true_positives = sum(
        int(score["detection"]["precision_true_positives"]) for score in scores
    )
    precision = _ratio(precision_true_positives, precision_basis)
    recall = _ratio(true_positives, reference_basis)
    f1 = None
    if precision is not None and recall is not None:
        f1 = round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0

    field_accuracy = {
        field: _ratio(
            sum(
                float(score["field_accuracy"][field]) * int(score["field_matching_basis"])
                for score in scores
            ),
            field_basis,
        )
        for field in FIELD_NAMES
    }
    numeric_unit_correct = sum(
        bool(match["value"] and match["unit"]) for score in scores for match in score["matches"]
    )
    evidence_correct = sum(
        bool(match["page"] and match["evidence_supported"])
        for score in scores
        for match in score["matches"]
    )
    numeric_unit_accuracy = _ratio(numeric_unit_correct, reference_basis)
    evidence_accuracy = _ratio(evidence_correct, reference_basis)

    negative_safety = [score.get("negative_control_safety", {}) for score in scores]
    controls_total = sum(int(item.get("controls_total", 0)) for item in negative_safety)
    controls_matched = sum(int(item.get("matched_control_count", 0)) for item in negative_safety)
    matched_candidates = sum(
        int(item.get("matched_candidate_count", item.get("matched_control_count", 0)))
        for item in negative_safety
    )
    false_primary_count = sum(int(item.get("false_primary_count", 0)) for item in negative_safety)
    false_primary_export_count = sum(
        int(item.get("false_primary_export_count", 0)) for item in negative_safety
    )
    false_primary_rate = _ratio(false_primary_count, matched_candidates)
    control_match_coverage = _ratio(controls_matched, controls_total)
    negative_controls_measured = matched_candidates > 0
    claim_type_pairs: list[tuple[str, str]] = []
    for score in scores:
        claim_type_pairs.extend(
            (match["expected_claim_type"], match["actual_claim_type"])
            for match in score["matches"]
            if match.get("actual_claim_type") is not None
        )
        claim_type_pairs.extend(
            (match["expected_claim_type"], match["actual_claim_type"])
            for match in score.get("negative_control_safety", {}).get("matches", [])
        )
    claim_type_classification = score_claim_type_pairs(claim_type_pairs)

    macro_precision_values = [
        float(score["detection"]["precision"])
        for score in scores
        if score["detection"]["precision"] is not None
    ]
    macro_recall_values = [float(score["detection"]["recall"]) for score in scores]
    macro_f1_values = [
        float(score["detection"]["f1"]) for score in scores if score["detection"]["f1"] is not None
    ]

    result = {
        "schema_version": "corpus-reference-score/0.2",
        "papers_scored": len(scores),
        "coverage_statement": (
            "Recall and field accuracy cover annotated reference observations only. "
            "Precision covers only explicitly fully annotated labels. Sampled and "
            "excluded regions remain outside the precision basis and do not establish "
            "whole-paper gold."
        ),
        "bases": {
            "reference_observations": reference_basis,
            "field_matching": field_basis,
            "precision_candidates_in_fully_annotated_regions": precision_basis,
        },
        "detection": {
            "true_positives": true_positives,
            "precision_true_positives": precision_true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": _mean(macro_precision_values),
            "macro_recall": _mean(macro_recall_values),
            "macro_f1": _mean(macro_f1_values),
            "precision_defined_papers": len(macro_precision_values),
        },
        "field_accuracy": field_accuracy,
        "derived_accuracy": {
            "exact_numeric_value_and_unit": numeric_unit_accuracy,
            "evidence_page_and_text_support": evidence_accuracy,
        },
        "negative_control_safety": {
            "controls_total": controls_total,
            "controls_matched": controls_matched,
            "control_match_coverage": control_match_coverage,
            "control_match_coverage_defined": controls_total > 0,
            "measurement_status": ("measured" if negative_controls_measured else "not_measured"),
            "matched_candidates": matched_candidates,
            "false_primary_count": false_primary_count,
            "false_primary_export_count": false_primary_export_count,
            "false_primary_rate": false_primary_rate,
            "zero_false_primary_gate_passed": (
                false_primary_count == 0 if negative_controls_measured else None
            ),
            "zero_false_primary_export_gate_passed": (
                false_primary_export_count == 0 if negative_controls_measured else None
            ),
        },
        "claim_type_classification": claim_type_classification,
    }
    result["quality_gates"] = {
        "candidate_detection_recall": _gate(recall, 0.90),
        "exact_numeric_value_and_unit": _gate(numeric_unit_accuracy, 0.98),
        "joint_system_dataset_metric_value_slice": _gate(field_accuracy["joint_semantics"], 0.95),
        "evidence_page_and_text_support": _gate(evidence_accuracy, 0.95),
        "evidence_table_figure_row_column": _gate(field_accuracy["evidence_structure"], 0.95),
        "honest_missingness": _gate(field_accuracy["missingness"], 0.95),
        "claim_type_macro_f1": _gate(claim_type_classification["macro_f1"], 0.90),
        "false_primary_controls": _gate(
            float(false_primary_count) if negative_controls_measured else None,
            0.0,
            direction="at_most",
        ),
        "false_primary_exports": _gate(
            float(false_primary_export_count) if negative_controls_measured else None,
            0.0,
            direction="at_most",
        ),
    }
    return result
