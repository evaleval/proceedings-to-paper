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


def _identified_control_frame(
    scores: list[dict[str, Any]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]] | None:
    """Return unique known, matched, and examined controls when ids are available.

    A repeated bakeoff evaluates the same annotated controls more than once.  Counts of
    provider trials should retain that repetition, but annotation-frame coverage must
    not turn 17/21 unique controls into 34/42 merely because the run had two repetitions.
    Older score payloads without ids keep their historical count-based fallback.
    """

    known: set[tuple[str, str]] = set()
    matched: set[tuple[str, str]] = set()
    examined: set[tuple[str, str]] = set()
    for score in scores:
        paper_id = score.get("paper_id")
        safety = score.get("negative_control_safety", {})
        control_ids = safety.get("control_ids")
        matched_ids = safety.get("matched_control_ids")
        examined_ids = safety.get("examined_control_ids")
        if (
            not isinstance(paper_id, str)
            or not isinstance(control_ids, list)
            or not isinstance(matched_ids, list)
            or not isinstance(examined_ids, list)
        ):
            return None
        known.update((paper_id, str(control_id)) for control_id in control_ids)
        matched.update((paper_id, str(control_id)) for control_id in matched_ids)
        examined.update((paper_id, str(control_id)) for control_id in examined_ids)
    if not matched <= known or not examined <= known:
        raise ValueError("control score ids must be subsets of the annotated control frame")
    return known, matched, examined | matched


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
    control_trials_total = sum(int(item.get("controls_total", 0)) for item in negative_safety)
    control_trials_matched = sum(
        int(item.get("matched_control_count", 0)) for item in negative_safety
    )
    control_trials_examined = sum(
        max(
            int(item.get("matched_control_count", 0)),
            len(item.get("examined_control_ids", [])),
        )
        for item in negative_safety
    )
    control_trials_passed_by_abstention = sum(
        int(item.get("passed_by_abstention_count", 0)) for item in negative_safety
    )
    identified_frame = _identified_control_frame(scores)
    if identified_frame is None:
        controls_total = control_trials_total
        controls_matched = control_trials_matched
        controls_examined = control_trials_examined
        passed_by_abstention = control_trials_passed_by_abstention
    else:
        known_controls, matched_controls, examined_controls = identified_frame
        controls_total = len(known_controls)
        controls_matched = len(matched_controls)
        controls_examined = len(examined_controls)
        passed_by_abstention = len(examined_controls - matched_controls)
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
    controls_not_examined = max(0, controls_total - controls_examined)
    control_examination_coverage = _ratio(controls_examined, controls_total)
    negative_controls_partially_measured = controls_examined > 0
    negative_controls_fully_measured = controls_total > 0 and controls_examined == controls_total
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
    macro_recall_values = [
        float(score["detection"]["recall"])
        for score in scores
        if int(score["detection"].get("recall_basis", 0)) > 0
    ]
    macro_f1_values = [
        float(score["detection"]["f1"])
        for score in scores
        if score["detection"]["f1"] is not None
        and int(score["detection"].get("recall_basis", 0)) > 0
    ]

    observability = [
        score.get("input_observability", {})
        for score in scores
        if score.get("input_observability", {}).get("status") == "measured"
    ]
    observability_complete = bool(scores) and len(observability) == len(scores)
    observability_partial = bool(observability) and not observability_complete
    observable_references = sum(
        int(item.get("observable_reference_observations") or 0) for item in observability
    )
    unobservable_references = sum(
        int(item.get("unobservable_reference_observations") or 0) for item in observability
    )
    observable_true_positives = sum(
        int(item.get("model_conditional_detection", {}).get("true_positives") or 0)
        for item in observability
    )
    observable_false_negatives = sum(
        int(item.get("model_conditional_detection", {}).get("false_negatives") or 0)
        for item in observability
    )
    observable_total = observable_references + unobservable_references

    result = {
        "schema_version": "corpus-reference-score/0.4",
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
            "controls_examined": controls_examined,
            "controls_not_examined": controls_not_examined,
            "control_examination_coverage": control_examination_coverage,
            "passed_by_abstention_count": passed_by_abstention,
            "control_trials": {
                "total": control_trials_total,
                "matched": control_trials_matched,
                "examined": control_trials_examined,
                "passed_by_abstention": control_trials_passed_by_abstention,
            },
            "measurement_status": (
                "measured"
                if controls_total > 0 and controls_examined == controls_total
                else "partially_measured"
                if negative_controls_partially_measured
                else "not_measured"
            ),
            "matched_candidates": matched_candidates,
            "false_primary_count": false_primary_count,
            "false_primary_export_count": false_primary_export_count,
            "false_primary_rate": false_primary_rate,
            "zero_false_primary_gate_passed": (
                False if false_primary_count else True if negative_controls_fully_measured else None
            ),
            "zero_false_primary_export_gate_passed": (
                False
                if false_primary_export_count
                else True
                if negative_controls_fully_measured
                else None
            ),
        },
        "input_observability": {
            "status": (
                "measured"
                if observability_complete
                else "partially_assessed"
                if observability_partial
                else "not_assessed"
            ),
            "papers_measured": len(observability),
            "papers_not_assessed": len(scores) - len(observability),
            "measured_reference_observations": observable_total,
            "reference_observations": observable_total if observability_complete else None,
            "observable_reference_observations": (
                observable_references if observability_complete else None
            ),
            "unobservable_reference_observations": (
                unobservable_references if observability_complete else None
            ),
            "observation_coverage": (
                _ratio(observable_references, observable_total) if observability_complete else None
            ),
            "model_conditional_detection": {
                "true_positives": (observable_true_positives if observability_complete else None),
                "false_negatives": (observable_false_negatives if observability_complete else None),
                "recall_basis": observable_references if observability_complete else None,
                "recall": (
                    _ratio(observable_true_positives, observable_references)
                    if observability_complete
                    else None
                ),
            },
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
        "claim_type_macro_f1": _gate(
            claim_type_classification["macro_f1"]
            if int(claim_type_classification["supported_classes"]) >= 2
            else None,
            0.90,
        ),
        "false_primary_controls": _gate(
            (
                float(false_primary_count)
                if false_primary_count or negative_controls_fully_measured
                else None
            ),
            0.0,
            direction="at_most",
        ),
        "false_primary_exports": _gate(
            (
                float(false_primary_export_count)
                if false_primary_export_count or negative_controls_fully_measured
                else None
            ),
            0.0,
            direction="at_most",
        ),
    }
    return result
