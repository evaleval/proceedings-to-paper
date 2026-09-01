from __future__ import annotations

from proceedings_to_eee.domain.attribution import AttributionState
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus, TextSupportStatus
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.validation.candidates import (
    deduplicate_candidates,
    validate_candidates,
)


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


def test_wrong_quote_is_not_exported(eligible_candidate: CandidateObservation) -> None:
    changed_anchor = eligible_candidate.evidence[0].model_copy(
        update={"quote": "Atlas Moderation API 61.3 99.1%", "quote_sha256": None}
    )
    eligible_candidate.evidence = [changed_anchor]
    validated = validate_candidates([eligible_candidate], {"src_paper": _layout()})
    assert validated[0].text_support == TextSupportStatus.UNSUPPORTED
    assert validated[0].export_status == ExportStatus.NEEDS_REVIEW


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


def test_same_value_in_distinct_table_columns_is_not_treated_as_ambiguous(
    eligible_candidate: CandidateObservation,
) -> None:
    second = eligible_candidate.model_copy(deep=True)
    assert second.metric is not None
    second.metric.raw_name = "Accuracy"
    second.metric.canonical_id = "accuracy"
    second.evidence[0].column = "Accuracy"

    validated = validate_candidates([eligible_candidate, second], {"src_paper": _layout()})

    assert {candidate.referential_status.value for candidate in validated} == {"resolved"}
    assert all(
        "ambiguous evidence" not in (candidate.export_reason or "") for candidate in validated
    )
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
