from __future__ import annotations

import hashlib
import json

from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.extraction.llm_schema import provider_json_schema
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.region_index import build_page_region_index
from proceedings_to_eee.resolution.attribution import (
    attribute_candidate,
    load_lexicon,
)
from proceedings_to_eee.validation.candidates import _route_attribution, validate_candidates

# Invented table with one external comparator, one paper-owned result, and a caption
# that contains both external-reporting and first-party language.
MIXED_ORIGIN_PAGE = """                              Model                                  ROC-AUC
                              Linear Baseline                             0.611
                              Paper System                                0.742
                              Example Ensemble (Leaderboard Entry)         0.781
Table 2. Reported benchmark performance alongside the system developed in this study.
"""

# Every row carries the marker, so there is no contrast to decide on.
ALL_FOREIGN_PAGE = """Table 4. Published leaderboard entries.

     System                          Score
     Alpha Leaderboard Entry          0.91
     Beta Leaderboard Entry           0.88
     Gamma Leaderboard Entry          0.87
"""

# Citations here name invented datasets, not the origin of the numbers.
DATASET_CITATION_PAGE = """Table 1. Detector performance per dataset.

     Dataset                  AUC
     Dataset Alpha [11]      0.68
     Dataset Beta [90]       0.59
     Dataset Gamma [40]      0.63
"""


def _page(text: str, page: int = 13) -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_src_paper_{page:04d}",
        source_id="src_paper",
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=0,
        result_signal_score=1.0,
    )


def _candidate(
    quote: str,
    row: str,
    value: str,
    *,
    page: int = 13,
    export_status: ExportStatus = ExportStatus.ELIGIBLE,
) -> CandidateObservation:
    return CandidateObservation.model_validate(
        {
            "paper_id": "fixture",
            "claim_type": ClaimType.PRIMARY_RESULT,
            "roles": [{"role": "evaluated_system", "raw_name": row, "confidence": 0.9}],
            "scope": {"dataset_raw": "Synthetic Benchmark"},
            "metric": {"raw_name": "ROC-AUC", "canonical_id": "auroc", "unit": "proportion"},
            "value": {"raw": value, "numeric": float(value)},
            "evidence": [
                {
                    "source_id": "src_paper",
                    "page": page,
                    "kind": "table",
                    "label": "Table 2",
                    "row": row,
                    "column": "ROC-AUC",
                    "quote": quote,
                }
            ],
            "export_status": export_status,
            "extraction_method": "fixture",
            "extraction_confidence": 0.95,
        }
    )


def _attribute(page_text: str, quote: str, row: str, value: str, page: int = 13):
    fragment = _page(page_text, page)
    return attribute_candidate(
        _candidate(quote, row, value, page=page),
        build_page_region_index(fragment),
        fragment,
        load_lexicon(),
    )


def test_external_comparator_is_attributed_to_its_own_source() -> None:
    verdict = _attribute(
        MIXED_ORIGIN_PAGE,
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    assert verdict.state is AttributionState.EXTERNALLY_SOURCED
    assert verdict.rule_id == "row_scoped_foreign_cue"
    assert [cue.cue_id for cue in verdict.cues] == ["leaderboard"]
    assert verdict.demotes


def test_the_papers_own_row_in_an_ambiguous_caption_stays_unresolved() -> None:
    """Caption-scoped evidence may neither decide external nor positive origin."""

    verdict = _attribute(MIXED_ORIGIN_PAGE, "Paper System 0.742", "Paper System", "0.742")
    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.demotes
    # The caption cue is recorded, so the evidence stays visible in review.
    assert [cue.cue_id for cue in verdict.cues] == ["caption_reported"]
    assert verdict.rule_id == "weak_cue_recorded"


def test_a_dataset_citation_abstains_without_claiming_external_origin() -> None:
    verdict = _attribute(
        DATASET_CITATION_PAGE,
        "Dataset Beta [90] 0.59",
        "Dataset Beta [90]",
        "0.59",
        page=4,
    )
    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.demotes
    assert "citation_bracket" in {cue.cue_id for cue in verdict.cues}


def test_a_marker_on_every_row_abstains_rather_than_demoting_the_table() -> None:
    verdict = _attribute(
        ALL_FOREIGN_PAGE, "Beta Leaderboard Entry 0.88", "Beta Leaderboard Entry", "0.88", page=9
    )
    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "decisive_cue_without_contrast"
    assert verdict.contrast_rows_total == verdict.contrast_rows_matched


def test_missing_structure_is_unresolved_not_no_signal() -> None:
    candidate = _candidate("System A 0.80", "System A", "0.80", page=1)
    verdict = attribute_candidate(candidate, None, None, load_lexicon())

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "no_page_index"


def test_unlocatable_quote_is_unresolved_not_no_signal() -> None:
    fragment = _page("Table 1. Results.\nSystem    AUC\nSystem A  0.81\n", page=1)
    candidate = _candidate("System A  0.80", "System A", "0.80", page=1)
    verdict = attribute_candidate(
        candidate,
        build_page_region_index(fragment),
        fragment,
        load_lexicon(),
    )

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "unlocatable"


def test_non_table_quote_is_unresolved_not_no_signal() -> None:
    text = "Results\nSystem A achieved an AUC of 0.80 on Dataset A.\n"
    verdict = _attribute(text, "System A achieved an AUC of 0.80", "System A", "0.80", page=1)

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "not_a_table_row"


def test_paper_produced_exists_as_a_trusted_state_but_is_not_inferred() -> None:
    trusted = AttributionVerdict(
        state=AttributionState.PAPER_PRODUCED,
        rule_id="explicit_test_fixture",
    )
    inferred = _attribute(
        """Table 1. Results.\nSystem    AUC\nSystem A  0.80\n""",
        "System A  0.80",
        "System A",
        "0.80",
        page=1,
    )

    assert trusted.allows_canonical_export
    assert not trusted.demotes
    assert inferred.state is AttributionState.NO_SIGNAL
    assert inferred.rule_id == "no_cue"
    assert inferred.state is not AttributionState.PAPER_PRODUCED
    assert inferred.demotes


def test_verdict_records_the_lexicon_it_used() -> None:
    verdict = _attribute(
        MIXED_ORIGIN_PAGE,
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    lexicon = load_lexicon()
    assert verdict.lexicon_id == lexicon.lexicon_id
    assert verdict.lexicon_sha256 == lexicon.sha256


def _layout(text: str, page: int = 13) -> dict[str, PdfLayout]:
    """Pad to the requested page so page numbers line up with the real pipeline."""

    pages = [_page("filler\n", number) for number in range(1, page)]
    pages.append(_page(text, page))
    return {
        "src_paper": PdfLayout(
            source_id="src_paper",
            parser="poppler-pdftotext-layout",
            parser_version="test",
            page_count=page,
            pages=pages,
        )
    }


def test_gate_demotes_the_comparator_to_review_not_to_a_silent_drop() -> None:
    comparator = _candidate(
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    own = _candidate("Paper System 0.742", "Paper System", "0.742")
    validate_candidates([comparator, own], _layout(MIXED_ORIGIN_PAGE), min_confidence=0.0)

    assert comparator.export_status is ExportStatus.NEEDS_REVIEW
    assert comparator.export_reason is not None
    assert "attribution=externally_sourced" in comparator.export_reason
    # NOT_ELIGIBLE carries no risk reason in the human review lane, so it would drop the
    # candidate out of sight. Review keeps it inspectable and still clears the export gate.
    assert comparator.export_status is not ExportStatus.NOT_ELIGIBLE
    assert comparator.attribution is not None
    assert comparator.attribution.state is AttributionState.EXTERNALLY_SOURCED
    assert own.attribution is not None
    assert own.attribution.state is AttributionState.UNRESOLVED
    assert own.export_status is ExportStatus.NEEDS_REVIEW
    assert "attribution=unresolved" in (own.export_reason or "")


def test_attribution_is_demote_only() -> None:
    """The attribution step alone may only worsen an export status, never improve it."""

    layouts = _layout(MIXED_ORIGIN_PAGE)
    held = _candidate(
        "Paper System 0.742",
        "Paper System",
        "0.742",
        export_status=ExportStatus.NEEDS_REVIEW,
    )
    held.export_reason = "held for another reason"
    comparator = _candidate(
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
        export_status=ExportStatus.NEEDS_REVIEW,
    )
    _route_attribution([held, comparator], layouts)

    # A no_signal verdict changes nothing at all.
    assert held.export_status is ExportStatus.NEEDS_REVIEW
    assert held.export_reason == "held for another reason"
    # A demoting verdict on an already-held candidate does not resurrect it either.
    assert comparator.export_status is ExportStatus.NEEDS_REVIEW
    assert comparator.attribution is not None
    assert comparator.attribution.state is AttributionState.EXTERNALLY_SOURCED


def test_many_systems_with_no_cue_all_stay_in_review() -> None:
    """No rule may treat one entity as the target and the rest as distractors."""

    text = """Table 5. Aggregate performance.

     System                 AUC
     System Cedar          0.68
     System Juniper        0.59
     System Maple          0.63
     System Willow         0.71
"""
    layouts = _layout(text, page=5)
    rows = [
        ("System Cedar 0.68", "System Cedar", "0.68"),
        ("System Juniper 0.59", "System Juniper", "0.59"),
        ("System Maple 0.63", "System Maple", "0.63"),
        ("System Willow 0.71", "System Willow", "0.71"),
    ]
    candidates = [_candidate(quote, row, value, page=5) for quote, row, value in rows]
    validate_candidates(candidates, layouts, min_confidence=0.0)
    assert all(item.export_status is ExportStatus.NEEDS_REVIEW for item in candidates)
    assert all(
        item.attribution is not None and item.attribution.state is AttributionState.NO_SIGNAL
        for item in candidates
    )


def test_absent_layout_demotes_an_otherwise_exportable_candidate() -> None:
    candidate = _candidate("System A 0.80", "System A", "0.80", page=1)

    _route_attribution([candidate], {})

    assert candidate.attribution is not None
    assert candidate.attribution.state is AttributionState.UNRESOLVED
    assert candidate.export_status is ExportStatus.NEEDS_REVIEW


def test_attribution_never_enters_the_provider_wire_schema() -> None:
    """Standing guard: the extraction schema is hash-pinned in the holdout seal."""

    assert "attribution" not in json.dumps(provider_json_schema())


def test_attribution_never_becomes_a_claim_type_member() -> None:
    """Standing guard: ClaimType types the wire model, so adding a member voids the seal."""

    assert {member.value for member in ClaimType} == {
        "primary_result",
        "secondary_claim",
        "illustration",
        "method_metadata",
        "uncertain",
    }


def test_attribution_does_not_move_observation_ids() -> None:
    """stable_id() builds from an explicit whitelist, so the new field must be inert."""

    candidate = _candidate(
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    before = candidate.stable_id()
    _route_attribution([candidate], _layout(MIXED_ORIGIN_PAGE))
    assert candidate.attribution is not None
    assert candidate.attribution.state is AttributionState.EXTERNALLY_SOURCED
    assert candidate.stable_id() == before
