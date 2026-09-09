from __future__ import annotations

from proceedings_to_eee.domain.attribution import AttributionState
from proceedings_to_eee.domain.observation import CandidateObservation, Uncertainty
from proceedings_to_eee.domain.status import (
    ClaimType,
    ExportStatus,
    TextSupportStatus,
    ValueComparator,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.validation.candidates import (
    bounded_claim_present,
    deduplicate_candidates,
    resolve_references,
    route_candidate_attribution,
    validate_candidates,
    validate_non_origin_candidates,
)


def test_mean_auc_replay_repairs_provider_kind_and_resolves_reference(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate = eligible_candidate.model_copy(deep=True)
    assert candidate.metric is not None
    assert candidate.value is not None
    candidate.metric = candidate.metric.model_copy(
        update={
            "raw_name": "Mean AUC",
            "canonical_id": "auroc",
            "kind": "performance",
            "unit": None,
            "lower_is_better": False,
            "min_score": 0.0,
            "max_score": 1.0,
        }
    )
    candidate.value = candidate.value.model_copy(
        update={"raw": "0.960", "numeric": 0.96, "unit": None}
    )

    resolve_references(candidate)

    assert candidate.metric.canonical_id == "auroc"
    assert candidate.metric.kind == "auroc"
    assert candidate.metric.lower_is_better is False
    assert candidate.metric.unit == "proportion"
    assert candidate.metric.min_score == 0.0
    assert candidate.metric.max_score == 1.0
    assert candidate.value.unit == "proportion"
    assert candidate.referential_status.value == "resolved"


def test_numeric_claim_boundaries_reject_scientific_notation_substrings() -> None:
    for text, claim in (
        ("5e3", "5"),
        ("5e3", "3"),
        ("0.5e-3", "0.5"),
        ("1×10^3", "1"),
        ("1 x 10^3", "1"),
        ("1×10^3", "10"),
        ("1×10^3", "3"),
    ):
        assert not bounded_claim_present(text, claim)

    assert bounded_claim_present("The value is 5%.", "5%")
    assert bounded_claim_present("The score is 5%.", "5%")
    assert bounded_claim_present("The value is 0.5e-3.", "0.5e-3")


def _layout() -> PdfLayout:
    pages = [
        PageFragment(
            fragment_id=f"page-{number}",
            source_id="src_paper",
            page=number,
            text=(
                "empty\n" if number != 7 else "Table 2\nAtlas Moderation API  61.3  74.6%  58.2\n"
            ),
            text_sha256=str(number).zfill(64),
            character_count=5,
            numeric_token_count=3 if number == 7 else 0,
            result_signal_score=5 if number == 7 else 0,
        )
        for number in range(1, 8)
    ]
    return PdfLayout(
        source_id="src_paper",
        parser="fixture",
        parser_version="fixture-1",
        page_count=7,
        pages=pages,
    )


def test_supported_primary_proposal_without_positive_origin_stays_in_review(
    eligible_candidate: CandidateObservation,
) -> None:
    eligible_candidate.text_support = TextSupportStatus.UNVERIFIED
    validated = validate_candidates([eligible_candidate], {"src_paper": _layout()})
    assert validated[0].text_support == TextSupportStatus.SUPPORTED
    assert validated[0].attribution is not None
    assert validated[0].attribution.state is AttributionState.NO_SIGNAL
    assert validated[0].export_status == ExportStatus.NEEDS_REVIEW
    assert "attribution=no_signal" in (validated[0].export_reason or "")


def test_non_origin_validation_exposes_eligibility_without_mutating_frozen_candidate(
    eligible_candidate: CandidateObservation,
) -> None:
    eligible_candidate.attribution = None
    eligible_candidate.text_support = TextSupportStatus.UNVERIFIED
    eligible_candidate.export_status = ExportStatus.NEEDS_REVIEW
    before = eligible_candidate.model_dump(mode="json")

    [validated] = validate_non_origin_candidates(
        [eligible_candidate],
        {"src_paper": _layout()},
    )

    assert validated is not eligible_candidate
    assert validated.text_support is TextSupportStatus.SUPPORTED
    assert validated.export_status is ExportStatus.ELIGIBLE
    assert validated.attribution is None
    assert eligible_candidate.model_dump(mode="json") == before


def test_attribution_can_be_routed_after_independent_checks(
    eligible_candidate: CandidateObservation,
) -> None:
    pre_origin = validate_non_origin_candidates(
        [eligible_candidate],
        {"src_paper": _layout()},
    )

    assert pre_origin[0].export_status is ExportStatus.ELIGIBLE

    routed = route_candidate_attribution(pre_origin, {"src_paper": _layout()})

    assert routed is pre_origin
    assert routed[0].attribution is not None
    assert routed[0].export_status is ExportStatus.NEEDS_REVIEW


def test_wrong_quote_is_not_exported(eligible_candidate: CandidateObservation) -> None:
    changed_anchor = eligible_candidate.evidence[0].model_copy(
        update={"quote": "Atlas Moderation API 61.3 99.1%", "quote_sha256": None}
    )
    eligible_candidate.evidence = [changed_anchor]
    validated = validate_candidates([eligible_candidate], {"src_paper": _layout()})
    assert validated[0].text_support == TextSupportStatus.UNSUPPORTED
    assert validated[0].export_status == ExportStatus.NEEDS_REVIEW


def test_exact_value_and_context_lines_jointly_support_one_candidate(
    eligible_candidate: CandidateObservation,
) -> None:
    value_anchor = eligible_candidate.evidence[0]
    context_anchor = value_anchor.model_copy(
        update={
            "quote": "Table 2",
            "row": None,
            "column": None,
            "quote_sha256": None,
        }
    )
    eligible_candidate.evidence = [value_anchor, context_anchor]

    [validated] = validate_candidates([eligible_candidate], {"src_paper": _layout()})

    assert validated.text_support is TextSupportStatus.SUPPORTED


def test_stitched_cross_line_quote_is_not_text_supported(
    eligible_candidate: CandidateObservation,
) -> None:
    eligible_candidate.evidence[0] = eligible_candidate.evidence[0].model_copy(
        update={
            "quote": "Table 2 Atlas Moderation API  61.3  74.6%  58.2",
            "quote_sha256": None,
        }
    )

    [validated] = validate_candidates([eligible_candidate], {"src_paper": _layout()})

    assert validated.text_support is TextSupportStatus.UNSUPPORTED


def test_whitespace_normalized_but_non_verbatim_quote_is_not_text_supported(
    eligible_candidate: CandidateObservation,
) -> None:
    eligible_candidate.evidence[0] = eligible_candidate.evidence[0].model_copy(
        update={
            "quote": "Atlas Moderation API 61.3 74.6% 58.2",
            "quote_sha256": None,
        }
    )

    [validated] = validate_candidates([eligible_candidate], {"src_paper": _layout()})

    assert validated.text_support is TextSupportStatus.UNSUPPORTED


def test_raw_value_substring_inside_a_different_number_is_not_supported(
    eligible_candidate: CandidateObservation,
) -> None:
    assert eligible_candidate.value is not None
    eligible_candidate.value.raw = "4.6%"
    eligible_candidate.value.numeric = 4.6

    [validated] = validate_candidates([eligible_candidate], {"src_paper": _layout()})

    assert validated.text_support is TextSupportStatus.PARTIALLY_SUPPORTED


def test_exact_context_without_a_value_is_only_partially_supported(
    eligible_candidate: CandidateObservation,
) -> None:
    eligible_candidate.evidence[0] = eligible_candidate.evidence[0].model_copy(
        update={"quote": "Table 2", "quote_sha256": None}
    )

    [validated] = validate_candidates([eligible_candidate], {"src_paper": _layout()})

    assert validated.text_support is TextSupportStatus.PARTIALLY_SUPPORTED


def test_secondary_claim_is_retained_but_not_exported(
    eligible_candidate: CandidateObservation,
) -> None:
    eligible_candidate.claim_type = ClaimType.SECONDARY_CLAIM
    validated = validate_candidates([eligible_candidate], {"src_paper": _layout()})
    assert validated[0].export_status == ExportStatus.NOT_ELIGIBLE


def test_duplicate_proposals_merge_evidence(eligible_candidate: CandidateObservation) -> None:
    duplicate = eligible_candidate.model_copy(deep=True)
    merged = deduplicate_candidates([eligible_candidate, duplicate])
    assert len(merged) == 1
    assert "merged 2 duplicate proposals" in merged[0].notes


def test_deterministic_resolution_refreshes_the_semantic_observation_id(
    eligible_candidate: CandidateObservation,
) -> None:
    eligible_candidate.metric.unit = None
    eligible_candidate.metric.min_score = None
    eligible_candidate.metric.max_score = None
    eligible_candidate.value.unit = None
    eligible_candidate.value.raw = "0.746"
    eligible_candidate.value.numeric = 0.746
    eligible_candidate.evidence[0] = eligible_candidate.evidence[0].model_copy(
        update={
            "quote": "Atlas Moderation API  61.3  0.746  58.2",
            "quote_sha256": None,
        }
    )
    old_id = eligible_candidate.observation_id

    validated = validate_candidates([eligible_candidate], {"src_paper": _layout()})[0]

    assert validated.value.unit == "proportion"
    assert validated.observation_id == validated.stable_id()
    assert validated.observation_id != old_id


def test_same_quoted_value_with_incompatible_scope_is_routed_to_review(
    eligible_candidate: CandidateObservation,
) -> None:
    conflicting = eligible_candidate.model_copy(deep=True)
    assert conflicting.scope is not None
    conflicting.scope.dataset_raw = "Alternate Speech Set"

    validated = validate_candidates(
        [eligible_candidate, conflicting],
        {"src_paper": _layout()},
    )

    assert len(validated) == 2
    assert {candidate.export_status for candidate in validated} == {ExportStatus.NEEDS_REVIEW}
    assert {candidate.referential_status.value for candidate in validated} == {"wrong_scope"}
    assert all("ambiguous evidence" in (candidate.export_reason or "") for candidate in validated)


def test_untrusted_anchor_metadata_cannot_split_identical_quote_conflicts(
    eligible_candidate: CandidateObservation,
) -> None:
    conflicting = eligible_candidate.model_copy(deep=True)
    assert conflicting.scope is not None
    conflicting.scope.dataset_raw = "Alternate Speech Set"
    conflicting.evidence[0] = conflicting.evidence[0].model_copy(
        update={
            "label": "Fabricated table",
            "row": "Fabricated row",
            "column": "Fabricated column",
        }
    )

    validated = validate_candidates(
        [eligible_candidate, conflicting],
        {"src_paper": _layout()},
    )

    assert {candidate.referential_status.value for candidate in validated} == {"wrong_scope"}
    assert all("ambiguous evidence" in (candidate.export_reason or "") for candidate in validated)


def test_same_value_occurrence_with_different_uncertainty_is_a_conflict(
    eligible_candidate: CandidateObservation,
) -> None:
    conflicting = eligible_candidate.model_copy(deep=True)
    assert conflicting.value is not None
    conflicting.value.uncertainty = Uncertainty(standard_error=0.1)
    conflicting.evidence.append(
        conflicting.evidence[0].model_copy(update={"quote": "Table 2", "quote_sha256": None})
    )

    validated = validate_candidates(
        [eligible_candidate, conflicting],
        {"src_paper": _layout()},
    )

    assert {candidate.referential_status.value for candidate in validated} == {"wrong_scope"}


def test_same_value_occurrence_with_different_comparator_is_a_conflict(
    eligible_candidate: CandidateObservation,
) -> None:
    conflicting = eligible_candidate.model_copy(deep=True)
    assert conflicting.value is not None
    conflicting.value.comparator = ValueComparator.APPROXIMATELY

    validated = validate_candidates(
        [eligible_candidate, conflicting],
        {"src_paper": _layout()},
    )

    assert {candidate.referential_status.value for candidate in validated} == {"wrong_scope"}


def test_distinct_values_in_one_table_row_are_not_treated_as_ambiguous(
    eligible_candidate: CandidateObservation,
) -> None:
    second = eligible_candidate.model_copy(deep=True)
    assert second.metric is not None
    assert second.value is not None
    second.metric.raw_name = "F1"
    second.metric.canonical_id = "f1"
    second.value.raw = "58.2"
    second.value.numeric = 58.2
    second.evidence[0].column = "F1"

    validated = validate_candidates([eligible_candidate, second], {"src_paper": _layout()})

    assert {candidate.referential_status.value for candidate in validated} == {"resolved"}
    assert all(
        "ambiguous evidence" not in (candidate.export_reason or "") for candidate in validated
    )
    assert {candidate.export_status for candidate in validated} == {ExportStatus.NEEDS_REVIEW}


def test_untrusted_column_labels_cannot_split_the_same_exact_value_quote(
    eligible_candidate: CandidateObservation,
) -> None:
    second = eligible_candidate.model_copy(deep=True)
    assert second.metric is not None
    second.metric.raw_name = "Accuracy"
    second.metric.canonical_id = "accuracy"
    second.evidence[0].column = "Accuracy"

    validated = validate_candidates([eligible_candidate, second], {"src_paper": _layout()})

    assert {candidate.referential_status.value for candidate in validated} == {"wrong_scope"}
    assert all("ambiguous evidence" in (candidate.export_reason or "") for candidate in validated)
    assert {candidate.export_status for candidate in validated} == {ExportStatus.NEEDS_REVIEW}


def test_contained_row_quote_with_conflicting_scope_is_routed_to_review(
    eligible_candidate: CandidateObservation,
) -> None:
    conflicting = eligible_candidate.model_copy(deep=True)
    assert conflicting.scope is not None
    conflicting.scope.dataset_raw = "Alternate Speech Set"
    conflicting.evidence[0] = conflicting.evidence[0].model_copy(
        update={
            "label": "Table 2. Evaluation results",
            "quote": "Atlas Moderation API  61.3  74.6%  58.2  52.7%",
            "quote_sha256": None,
        }
    )

    validated = validate_candidates(
        [eligible_candidate, conflicting],
        {"src_paper": _layout()},
    )

    assert {candidate.export_status for candidate in validated} == {ExportStatus.NEEDS_REVIEW}


def test_overlapping_exact_spans_share_frozen_raw_value_occurrence_despite_label_drift(
    eligible_candidate: CandidateObservation,
) -> None:
    left = eligible_candidate.model_copy(deep=True)
    left.evidence[0] = left.evidence[0].model_copy(
        update={
            "quote": "Atlas Moderation API  61.3  74.6%",
            "quote_sha256": None,
            "label": "Claimed left label",
        }
    )
    right = eligible_candidate.model_copy(deep=True)
    assert right.scope is not None
    right.scope.dataset_raw = "Alternate Speech Set"
    right.evidence[0] = right.evidence[0].model_copy(
        update={
            "quote": "74.6%  58.2",
            "quote_sha256": None,
            "label": "Claimed right label",
            "row": "Claimed different row",
            "column": "Claimed different column",
        }
    )

    validated = validate_candidates([left, right], {"src_paper": _layout()})

    assert {candidate.referential_status.value for candidate in validated} == {"wrong_scope"}
    assert all("ambiguous evidence" in (candidate.export_reason or "") for candidate in validated)


def test_explicit_longer_system_alias_does_not_create_false_ambiguity(
    eligible_candidate: CandidateObservation,
) -> None:
    alias = eligible_candidate.model_copy(deep=True)
    alias.roles[0].raw_name = "Atlas Moderation API classifier"

    validated = validate_candidates([eligible_candidate, alias], {"src_paper": _layout()})

    assert {candidate.referential_status.value for candidate in validated} == {"resolved"}
    assert all(
        "ambiguous evidence" not in (candidate.export_reason or "") for candidate in validated
    )
    assert {candidate.export_status for candidate in validated} == {ExportStatus.NEEDS_REVIEW}


def test_structurally_contradictory_roles_and_fields_abstain(
    eligible_candidate: CandidateObservation,
) -> None:
    assert eligible_candidate.scope is not None
    assert eligible_candidate.metric is not None
    eligible_candidate.scope.dataset_raw = "English"
    eligible_candidate.scope.language = "English"
    eligible_candidate.metric.raw_name = "Tool-Named Corpus"

    [validated] = validate_candidates([eligible_candidate], {"src_paper": _layout()})

    assert validated.export_status == ExportStatus.NEEDS_REVIEW
    assert validated.referential_status.value == "wrong_scope"
    assert "semantic safety check" in (validated.export_reason or "")
