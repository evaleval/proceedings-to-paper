"""Field-level scoring against prompt-isolated reference annotations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path
from typing import Any

from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    ExportStatus,
    TextSupportStatus,
)
from proceedings_to_eee.domain.units import canonicalize_unit
from proceedings_to_eee.io import write_json
from proceedings_to_eee.reference import (
    PaperReference,
    ReferenceEvidence,
    ReferenceObservation,
    load_reference,
)

_COVERAGE_QUALIFIER_STOPWORDS = {
    "against",
    "cell",
    "cells",
    "column",
    "columns",
    "model",
    "result",
    "results",
    "row",
    "rows",
    "score",
    "scores",
    "versus",
    "vs",
}

# Annotation files use two separators for coverage-region components. Six holdout papers
# use a middle dot, three use a spaced hyphen. Accepting only the dot skipped the qualifier
# check entirely on those three, so every candidate sharing the table label entered the
# precision denominator. A spaced hyphen is required so that names carrying an internal
# hyphen, such as Gemini-2.0-Flash-Lite or a year range like 2016-2017, stay intact.
_COVERAGE_SEPARATOR = re.compile(r"[·|]|\s-\s")

_NUMERIC_MENTION = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+))"
    r"\s*(?P<magnitude>[kKmMbB])?\s*(?P<percent>%)?"
    r"(?![A-Za-z0-9_.])"
)


def _norm(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9.%+-]+", " ", value.casefold()).strip()


def _candidate_system(candidate: CandidateObservation) -> str:
    return next(
        (actor.raw_name for actor in candidate.roles if actor.role == ActorRole.EVALUATED_SYSTEM),
        "",
    )


def _reference_system(reference: ReferenceObservation) -> str:
    return next(
        actor.raw_name for actor in reference.actors if actor.role == ActorRole.EVALUATED_SYSTEM
    )


def _similar(left: str | None, right: str | None) -> bool:
    a, b = _norm(left), _norm(right)
    return bool(a and b and (a == b or a in b or b in a))


def _coverage_tokens(value: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9.%]+", " ", value.casefold())
    return {
        token
        for token in normalized.split()
        if token and token not in _COVERAGE_QUALIFIER_STOPWORDS
    }


def _coverage_label_matches(anchor_label: str | None, coverage_label: str) -> bool:
    anchor = _norm(anchor_label)
    covered = _norm(coverage_label)
    return bool(
        anchor
        and covered
        and (
            anchor == covered
            or anchor.startswith(covered + " ")
            or covered.startswith(anchor + " ")
        )
    )


def _candidate_coverage_text(candidate: CandidateObservation, anchor_indexes: set[int]) -> str:
    parts: list[str] = []
    for actor in candidate.roles:
        parts.extend(filter(None, (actor.raw_name, actor.version, actor.provider)))
    if candidate.scope:
        parts.extend(
            str(value)
            for value in candidate.scope.model_dump(mode="python").values()
            if value is not None
        )
    if candidate.metric:
        metric = candidate.metric.model_dump(mode="python")
        parts.extend(str(value) for value in metric.values() if value is not None)
    if candidate.value:
        parts.extend(filter(None, (candidate.value.raw, candidate.value.unit)))
    parts.extend(
        filter(
            None,
            (
                candidate.evaluation_construct,
                candidate.operationalization,
                candidate.decision_rule,
            ),
        )
    )
    for index in sorted(anchor_indexes):
        anchor = candidate.evidence[index]
        parts.extend(filter(None, (anchor.label, anchor.row, anchor.column, anchor.quote)))
    return " ".join(parts)


def _candidate_in_coverage(
    candidate: CandidateObservation, fully_annotated_labels: list[str]
) -> bool:
    """Conservatively place a candidate inside an explicitly complete label region."""

    for covered_region in fully_annotated_labels:
        components = [
            item.strip() for item in _COVERAGE_SEPARATOR.split(covered_region) if item.strip()
        ]
        if not components:
            continue
        matching_anchors = {
            index
            for index, anchor in enumerate(candidate.evidence)
            if _coverage_label_matches(anchor.label, components[0])
        }
        if not matching_anchors:
            continue
        qualifiers = _coverage_tokens(" ".join(components[1:]))
        if not qualifiers:
            return True
        candidate_tokens = _coverage_tokens(_candidate_coverage_text(candidate, matching_anchors))
        if qualifiers <= candidate_tokens:
            return True
    return False


def _anchor_pair_matches(
    expected_anchor: ReferenceEvidence,
    actual_anchor: EvidenceAnchor,
    through: str,
) -> bool:
    if expected_anchor.page != actual_anchor.page:
        return False
    if through == "page":
        return True
    if expected_anchor.kind != actual_anchor.kind:
        return False
    if through == "kind":
        return True
    if expected_anchor.label is not None and not _similar(
        expected_anchor.label, actual_anchor.label
    ):
        return False
    if through == "label":
        return True
    if expected_anchor.row is not None and not _similar(expected_anchor.row, actual_anchor.row):
        return False
    if through == "row":
        return True
    return expected_anchor.column is None or _similar(expected_anchor.column, actual_anchor.column)


def _any_anchor_pair(
    expected_anchors: list[ReferenceEvidence],
    actual_anchors: list[EvidenceAnchor],
    through: str,
) -> bool:
    return any(
        _anchor_pair_matches(expected_anchor, actual_anchor, through)
        for expected_anchor in expected_anchors
        for actual_anchor in actual_anchors
    )


def _compatibility(reference: ReferenceObservation, candidate: CandidateObservation) -> int:
    score = 0
    score += 3 * _similar(_reference_system(reference), _candidate_system(candidate))
    score += 3 * _similar(
        reference.scope.dataset_raw, candidate.scope.dataset_raw if candidate.scope else None
    )
    score += 2 * _similar(
        reference.metric.raw_name, candidate.metric.raw_name if candidate.metric else None
    )
    score += 4 * _similar(reference.value.raw, candidate.value.raw if candidate.value else None)
    if reference.scope.group or (candidate.scope and candidate.scope.group):
        score += 2 * _similar(
            reference.scope.group, candidate.scope.group if candidate.scope else None
        )
    return score


@dataclass(frozen=True)
class MatchedReference:
    reference_id: str
    observation_id: str | None
    expected_claim_type: str
    actual_claim_type: str | None
    claim_type: bool
    system: bool
    dataset: bool
    metric: bool
    value: bool
    unit: bool
    slice: bool
    page: bool
    evidence_kind: bool
    evidence_label: bool
    evidence_row: bool
    evidence_column: bool
    evidence_structure: bool
    evidence_supported: bool
    missingness: bool

    @property
    def joint_semantics(self) -> bool:
        return (
            self.system and self.dataset and self.metric and self.value and self.unit and self.slice
        )


@dataclass(frozen=True)
class MatchedNegativeControl:
    """One candidate linked to a negative control by exact source structure."""

    control_id: str
    observation_id: str
    expected_claim_type: str
    actual_claim_type: str
    claim_type_matches: bool
    false_primary: bool
    export_status: str
    false_primary_export: bool
    matched_evidence_ids: tuple[str, ...]


def score_claim_type_pairs(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Compute macro-F1 on matched reference/control candidates only."""

    classes = [claim_type.value for claim_type in ClaimType]
    per_class: dict[str, dict[str, int | float | None]] = {}
    supported_f1: list[float] = []
    correct = 0
    for label in classes:
        true_positives = sum(expected == label and actual == label for expected, actual in pairs)
        false_positives = sum(expected != label and actual == label for expected, actual in pairs)
        false_negatives = sum(expected == label and actual != label for expected, actual in pairs)
        support = true_positives + false_negatives
        predicted = true_positives + false_positives
        precision = true_positives / predicted if predicted else None
        recall = true_positives / support if support else None
        f1 = None
        if support:
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else 0.0
            )
            supported_f1.append(f1)
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        correct += true_positives
    return {
        "basis": len(pairs),
        "basis_scope": "matched reference observations and matched negative controls",
        "supported_classes": len(supported_f1),
        "accuracy": correct / len(pairs) if pairs else None,
        "macro_f1": sum(supported_f1) / len(supported_f1) if supported_f1 else None,
        "per_class": per_class,
    }


def _anchor_matches_negative_evidence(
    candidate: CandidateObservation, evidence: ReferenceEvidence
) -> bool:
    for anchor in candidate.evidence:
        if anchor.page != evidence.page or anchor.kind != evidence.kind:
            continue
        if evidence.label and not _coverage_label_matches(anchor.label, evidence.label):
            continue
        if evidence.row and anchor.row and not _similar(evidence.row, anchor.row):
            continue
        if evidence.column and anchor.column and not _similar(evidence.column, anchor.column):
            continue
        if not _similar(evidence.exact_quote, anchor.quote):
            continue
        if not _candidate_value_matches_negative_evidence(candidate, evidence):
            continue
        return True
    return False


def _numeric_mentions(value: str) -> list[tuple[float, str | None]]:
    """Extract literal numeric targets and explicit units from control evidence."""

    mentions: list[tuple[float, str | None]] = []
    magnitude_scale = {None: 1.0, "k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}
    for match in _NUMERIC_MENTION.finditer(value):
        number = float(match.group("number").replace(",", ""))
        magnitude = match.group("magnitude")
        number *= magnitude_scale[magnitude.casefold() if magnitude else None]
        unit = "percent" if match.group("percent") else None
        mentions.append((number, unit))
    return mentions


def _candidate_value_matches_negative_evidence(
    candidate: CandidateObservation, evidence: ReferenceEvidence
) -> bool:
    """Require numeric controls to match the candidate's asserted result value.

    A candidate may quote an entire table row and carry a row-level sample count in
    its scope. Neither fact means that the candidate represents that sample count.
    When the negative-control quote names a numeric target, matching therefore uses
    ``candidate.value`` rather than any number merely co-located in the quote or
    scope. Nonnumeric controls retain their exact structural/quote matching path.
    """

    expected_values = _numeric_mentions(evidence.exact_quote)
    if not expected_values:
        return True
    if candidate.value is None:
        return False
    candidate_unit = canonicalize_unit(candidate.value.unit)
    if candidate_unit is None and "%" in candidate.value.raw:
        candidate_unit = "percent"
    return any(
        isclose(candidate.value.numeric, expected, rel_tol=1e-12, abs_tol=1e-12)
        and (expected_unit is None or candidate_unit == expected_unit)
        for expected, expected_unit in expected_values
    )


def _negative_control_matches(
    reference: PaperReference, candidates: list[CandidateObservation]
) -> list[MatchedNegativeControl]:
    evidence_by_id = {item.evidence_id: item for item in reference.evidence}
    matches: list[MatchedNegativeControl] = []
    exportable = {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}
    for control in reference.negative_controls:
        control_evidence = [evidence_by_id[evidence_id] for evidence_id in control.evidence_ids]
        for candidate in candidates:
            matched_evidence_ids = tuple(
                evidence.evidence_id
                for evidence in control_evidence
                if _anchor_matches_negative_evidence(candidate, evidence)
            )
            if not matched_evidence_ids:
                continue
            observation_id = candidate.observation_id or candidate.stable_id()
            false_primary = candidate.claim_type == ClaimType.PRIMARY_RESULT
            matches.append(
                MatchedNegativeControl(
                    control_id=control.control_id,
                    observation_id=observation_id,
                    expected_claim_type=str(control.expected_claim_type),
                    actual_claim_type=str(candidate.claim_type),
                    claim_type_matches=candidate.claim_type == control.expected_claim_type,
                    false_primary=false_primary,
                    export_status=str(candidate.export_status),
                    false_primary_export=false_primary and candidate.export_status in exportable,
                    matched_evidence_ids=matched_evidence_ids,
                )
            )
    return matches


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


CONTROL_MATCHED = "matched"
CONTROL_PASSED_BY_ABSTENTION = "passed_by_abstention"
CONTROL_NOT_EXAMINED = "not_examined"
CONTROL_EXAMINATION_UNKNOWN = "examination_unknown"


def _negative_control_report(
    reference: PaperReference,
    candidates: list[CandidateObservation],
    examination: dict[str, bool] | None = None,
) -> dict[str, Any]:
    matches = _negative_control_matches(reference, candidates)
    matched_control_ids = _ordered_unique([match.control_id for match in matches])
    matched_candidate_ids = _ordered_unique([match.observation_id for match in matches])
    false_primary_ids = _ordered_unique(
        [match.observation_id for match in matches if match.false_primary]
    )
    false_primary_export_ids = _ordered_unique(
        [match.observation_id for match in matches if match.false_primary_export]
    )
    known_control_ids = [control.control_id for control in reference.negative_controls]
    unmatched_control_ids = [
        control_id for control_id in known_control_ids if control_id not in set(matched_control_ids)
    ]
    rate_basis = len(matched_candidate_ids)
    controls_total = len(known_control_ids)
    control_match_coverage = len(matched_control_ids) / controls_total if controls_total else None

    # A control that attracted no candidate is only informative once it is known whether
    # extraction ever looked at its region. Examined-and-declined is the safety property
    # passing; never-examined says nothing.
    matched_set = set(matched_control_ids)
    control_status = {
        control_id: (
            CONTROL_MATCHED
            if control_id in matched_set
            else CONTROL_EXAMINATION_UNKNOWN
            if examination is None
            else CONTROL_PASSED_BY_ABSTENTION
            if examination.get(control_id)
            else CONTROL_NOT_EXAMINED
        )
        for control_id in known_control_ids
    }
    examined_control_ids = [
        control_id
        for control_id, status in control_status.items()
        if status in {CONTROL_MATCHED, CONTROL_PASSED_BY_ABSTENTION}
    ]
    not_examined_control_ids = [
        control_id
        for control_id, status in control_status.items()
        if status == CONTROL_NOT_EXAMINED
    ]
    control_examination_coverage = (
        len(examined_control_ids) / controls_total if controls_total else None
    )
    measurement_status = "measured" if examined_control_ids else "not_measured"
    return {
        "controls_total": controls_total,
        "control_ids": known_control_ids,
        "matched_control_count": len(matched_control_ids),
        "matched_control_ids": matched_control_ids,
        "unmatched_control_ids": unmatched_control_ids,
        "control_match_coverage": control_match_coverage,
        "control_match_coverage_defined": controls_total > 0,
        "control_status": control_status,
        "examined_control_ids": examined_control_ids,
        "not_examined_control_ids": not_examined_control_ids,
        "control_examination_coverage": control_examination_coverage,
        "passed_by_abstention_count": sum(
            status == CONTROL_PASSED_BY_ABSTENTION for status in control_status.values()
        ),
        "measurement_status": measurement_status,
        "matched_candidate_count": rate_basis,
        "matched_candidate_ids": matched_candidate_ids,
        "matches": [asdict(match) for match in matches],
        "false_primary_candidate_ids": false_primary_ids,
        "false_primary_count": len(false_primary_ids),
        "false_primary_rate": len(false_primary_ids) / rate_basis if rate_basis else 0.0,
        "false_primary_rate_basis": rate_basis,
        "false_primary_rate_defined": rate_basis > 0,
        "zero_false_primary_gate_passed": (
            not false_primary_ids if measurement_status == "measured" else None
        ),
        "false_primary_export_candidate_ids": false_primary_export_ids,
        "false_primary_export_count": len(false_primary_export_ids),
        "zero_false_primary_export_gate_passed": (
            not false_primary_export_ids if measurement_status == "measured" else None
        ),
    }


def score_reference(
    reference: PaperReference,
    candidates: list[CandidateObservation],
    control_examination: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Score annotated-reference recall and coverage-bounded candidate precision.

    Recall and field matching use every explicit reference observation. They do not imply
    whole-paper recall when the reference is a sampled slice. False positives and precision use
    only primary candidates anchored in ``coverage.fully_annotated_labels``.

    ``control_examination`` maps a control id to whether extraction ever saw its region.
    Supplying it separates a control that was examined and declined, which is the safety
    property passing, from one that was never looked at, which is uninformative. Omitting
    it leaves every unmatched control's examination unknown and preserves the previous,
    stricter reading in which only a matched control counts as measured.
    """

    remaining = set(range(len(candidates)))
    matched_candidate_indexes: set[int] = set()
    matches: list[MatchedReference] = []
    evidence_by_id = {item.evidence_id: item for item in reference.evidence}
    for expected in reference.observations:
        ranked = sorted(
            ((index, _compatibility(expected, candidates[index])) for index in remaining),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
        winner_index = ranked[0][0] if ranked and ranked[0][1] >= 6 else None
        actual = candidates[winner_index] if winner_index is not None else None
        if winner_index is not None:
            remaining.remove(winner_index)
            matched_candidate_indexes.add(winner_index)
        expected_anchors = [evidence_by_id[item] for item in expected.result_evidence_ids]
        actual_anchors = actual.evidence if actual else []

        expected_missing = set(expected.expected_missing_fields)
        actual_missing: set[str] = set()
        if actual:
            system = next(
                (actor for actor in actual.roles if actor.role == ActorRole.EVALUATED_SYSTEM),
                None,
            )
            if system and not system.version:
                actual_missing.add("evaluated_system.version")
            if actual.evaluation_date is None:
                actual_missing.add("evaluation_date")
        matches.append(
            MatchedReference(
                reference_id=expected.reference_id,
                observation_id=actual.observation_id if actual else None,
                expected_claim_type=str(expected.claim_type),
                actual_claim_type=str(actual.claim_type) if actual else None,
                claim_type=bool(actual and actual.claim_type == expected.claim_type),
                system=bool(
                    actual and _similar(_reference_system(expected), _candidate_system(actual))
                ),
                dataset=bool(
                    actual
                    and actual.scope
                    and _similar(expected.scope.dataset_raw, actual.scope.dataset_raw)
                ),
                metric=bool(
                    actual
                    and actual.metric
                    and _similar(expected.metric.raw_name, actual.metric.raw_name)
                ),
                value=bool(
                    actual and actual.value and _similar(expected.value.raw, actual.value.raw)
                ),
                unit=bool(
                    actual
                    and actual.value
                    and canonicalize_unit(expected.value.unit)
                    == canonicalize_unit(actual.value.unit)
                ),
                slice=bool(
                    actual
                    and actual.scope
                    and _norm(expected.scope.group) == _norm(actual.scope.group)
                    and _norm(expected.scope.split) == _norm(actual.scope.split)
                    and _norm(expected.scope.language) == _norm(actual.scope.language)
                ),
                page=bool(actual and _any_anchor_pair(expected_anchors, actual_anchors, "page")),
                evidence_kind=bool(
                    actual and _any_anchor_pair(expected_anchors, actual_anchors, "kind")
                ),
                evidence_label=bool(
                    actual and _any_anchor_pair(expected_anchors, actual_anchors, "label")
                ),
                evidence_row=bool(
                    actual and _any_anchor_pair(expected_anchors, actual_anchors, "row")
                ),
                evidence_column=bool(
                    actual and _any_anchor_pair(expected_anchors, actual_anchors, "column")
                ),
                evidence_structure=bool(
                    actual and _any_anchor_pair(expected_anchors, actual_anchors, "structure")
                ),
                evidence_supported=bool(
                    actual and actual.text_support == TextSupportStatus.SUPPORTED
                ),
                missingness=bool(actual and expected_missing <= actual_missing),
            )
        )
    true_positives = sum(match.observation_id is not None for match in matches)
    false_negatives = len(matches) - true_positives
    primary_candidate_indexes = {
        index
        for index, candidate in enumerate(candidates)
        if candidate.claim_type == ClaimType.PRIMARY_RESULT
    }
    primary_candidates_in_coverage = {
        index
        for index in primary_candidate_indexes
        if _candidate_in_coverage(candidates[index], reference.coverage.fully_annotated_labels)
    }
    precision_true_positive_indexes = matched_candidate_indexes & primary_candidates_in_coverage
    unmatched_primary_in_coverage = remaining & primary_candidates_in_coverage
    unmatched_primary_out_of_coverage = remaining & (
        primary_candidate_indexes - primary_candidates_in_coverage
    )
    false_positives = len(unmatched_primary_in_coverage)
    precision_basis = len(primary_candidates_in_coverage)
    precision_true_positives = len(precision_true_positive_indexes)
    precision = precision_true_positives / precision_basis if precision_basis else None
    recall = true_positives / len(matches) if matches else 0.0
    f1 = None
    if precision is not None:
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    fields = (
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
    unmatched_in_coverage = {
        index
        for index in remaining
        if _candidate_in_coverage(candidates[index], reference.coverage.fully_annotated_labels)
    }
    unmatched_out_of_coverage = remaining - unmatched_in_coverage
    negative_control_safety = _negative_control_report(reference, candidates, control_examination)
    claim_type_pairs = [
        (match.expected_claim_type, match.actual_claim_type)
        for match in matches
        if match.actual_claim_type is not None
    ]
    claim_type_pairs.extend(
        (item["expected_claim_type"], item["actual_claim_type"])
        for item in negative_control_safety["matches"]
    )
    return {
        "schema_version": "reference-score/0.6",
        "paper_id": reference.paper_id,
        "reference_observations": len(matches),
        "recall_basis": len(matches),
        "field_matching_basis": len(matches),
        "candidate_primary_results": len(primary_candidate_indexes),
        "primary_candidates_total": len(primary_candidate_indexes),
        "primary_candidates_in_coverage": len(primary_candidates_in_coverage),
        "primary_candidates_out_of_coverage": len(
            primary_candidate_indexes - primary_candidates_in_coverage
        ),
        "precision_basis": precision_basis,
        "coverage": {
            "recall_scope": "annotated_reference_observations",
            "field_matching_scope": "annotated_reference_observations",
            "precision_scope": "fully_annotated_labels",
            "fully_annotated_labels": list(reference.coverage.fully_annotated_labels),
            "sampled_labels": list(reference.coverage.sampled_labels),
        },
        "detection": {
            "true_positives": true_positives,
            "precision_true_positives": precision_true_positives,
            "precision_basis": precision_basis,
            "recall_basis": len(matches),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "precision_defined": precision_basis > 0,
            "recall": recall,
            "f1": f1,
        },
        "field_accuracy": {
            field: sum(bool(getattr(match, field)) for match in matches) / len(matches)
            if matches
            else 0.0
            for field in fields
        },
        "matches": [
            asdict(match) | {"joint_semantics": match.joint_semantics} for match in matches
        ],
        "unmatched_candidate_ids": [
            candidates[index].observation_id for index in sorted(remaining)
        ],
        "unmatched_candidate_ids_in_coverage": [
            candidates[index].observation_id for index in sorted(unmatched_in_coverage)
        ],
        "unmatched_candidate_ids_out_of_coverage": [
            candidates[index].observation_id for index in sorted(unmatched_out_of_coverage)
        ],
        "unmatched_primary_candidate_ids_in_coverage": [
            candidates[index].observation_id for index in sorted(unmatched_primary_in_coverage)
        ],
        "unmatched_primary_candidate_ids_out_of_coverage": [
            candidates[index].observation_id for index in sorted(unmatched_primary_out_of_coverage)
        ],
        "negative_controls": len(reference.negative_controls),
        "negative_control_safety": negative_control_safety,
        "claim_type_classification": score_claim_type_pairs(claim_type_pairs),
    }


def score_reference_files(
    reference_path: Path, observations_path: Path, output_path: Path
) -> dict[str, Any]:
    reference = load_reference(reference_path)
    candidates = [
        CandidateObservation.model_validate_json(line)
        for line in observations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = score_reference(reference, candidates)
    write_json(output_path, result)
    return result
