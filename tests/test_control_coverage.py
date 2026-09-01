from __future__ import annotations

import hashlib

import pytest

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation.control_coverage import control_examination
from proceedings_to_eee.evaluation.control_proposals import propose_controls, worklist
from proceedings_to_eee.evaluation.reference_score import (
    CONTROL_EXAMINATION_UNKNOWN,
    CONTROL_MATCHED,
    CONTROL_NOT_EXAMINED,
    CONTROL_PASSED_BY_ABSTENTION,
    score_reference,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.reference import PaperReference

# The invented table draws a block; the sample-count sentence sits far enough below it
# to fall between blocks.
PAGE = """Table 2. Reported benchmark performance.

     Model                            ROC-AUC
     Linear Baseline                       0.611
     Paper System                          0.742
     Example Ensemble (Leaderboard Entry)  0.781

We describe the architecture and the training procedure in the following
paragraphs, and we discuss the deployment considerations afterwards. The
frontend was written to be model agnostic so the backend can be swapped
without any change to the explanation pipeline that we have described.

4.7.1 Dataset. The synthetic collection contains 123456 generated records and
was filtered by length before the illustrative systems were configured.
"""


def _page(text: str = PAGE, page: int = 13) -> PageFragment:
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


def _layout(page: PageFragment) -> PdfLayout:
    return PdfLayout(
        source_id="src_paper",
        parser="fixture",
        parser_version="fixture/1",
        page_count=page.page,
        pages=[_page("filler\n", number) for number in range(1, page.page)] + [page],
    )


def _reference(controls: list[dict]) -> PaperReference:
    return PaperReference.model_validate(
        {
            "paper_id": "fixture",
            "source_sha256": "a" * 64,
            "annotation_protocol": "fixture/0.1",
            "annotation_status": "fixture",
            "coverage": {
                "fully_annotated_labels": ["Table 2"],
                "inclusion_rule": "fixture",
                "exclusion_rule": "fixture",
            },
            "evidence": [
                {
                    "evidence_id": "ev-target",
                    "purpose": "result",
                    "page": 13,
                    "kind": "table",
                    "label": "Table 2",
                    "row": "Paper System",
                    "column": "ROC-AUC",
                    "exact_quote": "Paper System                          0.742",
                },
                *[control["evidence"] for control in controls],
            ],
            "observations": [
                {
                    "reference_id": "ref-target",
                    "claim_type": "primary_result",
                    "actors": [{"role": "evaluated_system", "raw_name": "Paper System"}],
                    "scope": {"dataset_raw": "Synthetic Benchmark"},
                    "metric": {"raw_name": "ROC-AUC", "unit": "proportion"},
                    "value": {"raw": "0.742", "numeric": 0.742, "unit": "proportion"},
                    "result_evidence_ids": ["ev-target"],
                }
            ],
            "negative_controls": [control["control"] for control in controls],
        }
    )


IN_BLOCK_CONTROL = {
    "evidence": {
        "evidence_id": "ev-leaderboard",
        "purpose": "negative_control",
        "page": 13,
        "kind": "table",
        "label": "Table 2",
        "row": "Example Ensemble (Leaderboard Entry)",
        "exact_quote": "Example Ensemble (Leaderboard Entry)  0.781",
    },
    "control": {
        "control_id": "nc-leaderboard",
        "expected_claim_type": "secondary_claim",
        "evidence_ids": ["ev-leaderboard"],
        "reason_not_primary": "External leaderboard comparator.",
    },
}

OUT_OF_BLOCK_CONTROL = {
    "evidence": {
        "evidence_id": "ev-dataset-size",
        "purpose": "negative_control",
        "page": 13,
        "kind": "prose",
        "exact_quote": "The synthetic collection contains 123456 generated records",
    },
    "control": {
        "control_id": "nc-dataset-size",
        "expected_claim_type": "method_metadata",
        "evidence_ids": ["ev-dataset-size"],
        "reason_not_primary": "A sample count is method metadata.",
    },
}


def test_a_control_inside_an_extracted_block_counts_as_examined() -> None:
    page = _page()
    blocks = segment_page_result_blocks(page)
    examined = control_examination(_reference([IN_BLOCK_CONTROL]), _layout(page), blocks)
    assert examined == {"nc-leaderboard": True}


def test_a_control_outside_every_block_counts_as_not_examined() -> None:
    """Prose the segmenter never selected is a region extraction never saw."""

    page = _page()
    blocks = segment_page_result_blocks(page)
    examined = control_examination(_reference([OUT_OF_BLOCK_CONTROL]), _layout(page), blocks)
    assert examined == {"nc-dataset-size": False}


def test_a_control_with_no_blocks_at_all_counts_as_not_examined() -> None:
    page = _page()
    examined = control_examination(_reference([IN_BLOCK_CONTROL]), _layout(page), [])
    assert examined == {"nc-leaderboard": False}


def test_examined_and_declined_is_a_measured_pass_not_an_unmeasured_gap() -> None:
    """The whole point: 'no candidate' means something once examination is known."""

    reference = _reference([IN_BLOCK_CONTROL])
    score = score_reference(reference, [], {"nc-leaderboard": True})
    safety = score["negative_control_safety"]
    assert safety["control_status"]["nc-leaderboard"] == CONTROL_PASSED_BY_ABSTENTION
    assert safety["measurement_status"] == "measured"
    assert safety["control_examination_coverage"] == pytest.approx(1.0)
    assert safety["control_match_coverage"] == pytest.approx(0.0)
    assert safety["zero_false_primary_gate_passed"] is True
    assert safety["zero_false_primary_export_gate_passed"] is True


def test_never_examined_stays_unmeasured() -> None:
    score = score_reference(_reference([IN_BLOCK_CONTROL]), [], {"nc-leaderboard": False})
    safety = score["negative_control_safety"]
    assert safety["control_status"]["nc-leaderboard"] == CONTROL_NOT_EXAMINED
    assert safety["measurement_status"] == "not_measured"
    assert safety["zero_false_primary_gate_passed"] is None


def test_omitting_examination_preserves_the_previous_stricter_reading() -> None:
    score = score_reference(_reference([IN_BLOCK_CONTROL]), [])
    safety = score["negative_control_safety"]
    assert safety["control_status"]["nc-leaderboard"] == CONTROL_EXAMINATION_UNKNOWN
    assert safety["measurement_status"] == "not_measured"
    assert safety["zero_false_primary_gate_passed"] is None


def _candidate(quote: str, row: str, value: str) -> CandidateObservation:
    return CandidateObservation.model_validate(
        {
            "paper_id": "fixture",
            "claim_type": "primary_result",
            "roles": [{"role": "evaluated_system", "raw_name": row, "confidence": 0.9}],
            "scope": {"dataset_raw": "Synthetic Benchmark"},
            "metric": {"raw_name": "ROC-AUC", "unit": "proportion"},
            "value": {"raw": value, "numeric": float(value), "unit": "proportion"},
            "evidence": [
                {
                    "source_id": "src_paper",
                    "page": 13,
                    "kind": "table",
                    "label": "Table 2",
                    "row": row,
                    "column": "ROC-AUC",
                    "quote": quote,
                }
            ],
            "export_status": "exported",
            "extraction_method": "fixture",
            "extraction_confidence": 0.95,
        }
    )


def test_a_matched_control_still_reports_a_false_primary() -> None:
    candidate = _candidate(
        "Example Ensemble (Leaderboard Entry)  0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    score = score_reference(_reference([IN_BLOCK_CONTROL]), [candidate], {"nc-leaderboard": True})
    safety = score["negative_control_safety"]
    assert safety["control_status"]["nc-leaderboard"] == CONTROL_MATCHED
    assert safety["false_primary_count"] == 1
    assert safety["false_primary_export_count"] == 1
    assert safety["zero_false_primary_gate_passed"] is False


# ----------------------------------------------------------------------------------
# Proposals
# ----------------------------------------------------------------------------------


def test_proposals_only_come_from_inside_examined_blocks() -> None:
    page = _page()
    proposals = propose_controls(paper_id="fixture", layout=_layout(page), blocks=[])
    assert proposals == []


def test_a_foreign_row_cue_is_proposed_with_a_stated_reason() -> None:
    page = _page()
    proposals = propose_controls(
        paper_id="fixture", layout=_layout(page), blocks=segment_page_result_blocks(page)
    )
    foreign = [item for item in proposals if item.rule_id == "foreign_row_cue"]
    assert len(foreign) == 1
    assert foreign[0].row == "Example Ensemble (Leaderboard Entry)"
    assert foreign[0].expected_claim_type == "secondary_claim"
    assert "leaderboard" in foreign[0].reason
    assert not foreign[0].confirmed


def test_an_already_annotated_quote_is_not_proposed_again() -> None:
    page = _page()
    proposals = propose_controls(
        paper_id="fixture",
        layout=_layout(page),
        blocks=segment_page_result_blocks(page),
        reference=_reference([IN_BLOCK_CONTROL]),
    )
    assert not any(item.row == "Example Ensemble (Leaderboard Entry)" for item in proposals)


def test_a_sibling_row_never_gets_a_claim_type_asserted_for_it() -> None:
    """A paper may report primary results for many systems; only a human can decide."""

    page = _page()
    proposals = propose_controls(
        paper_id="fixture",
        layout=_layout(page),
        blocks=segment_page_result_blocks(page),
        reference=_reference([IN_BLOCK_CONTROL]),
    )
    siblings = [item for item in proposals if item.rule_id == "sibling_row_needs_label"]
    assert siblings
    assert all(item.expected_claim_type is None for item in siblings)


def test_the_worklist_marks_itself_unconfirmed_and_separates_the_two_kinds() -> None:
    page = _page()
    document = worklist(
        propose_controls(
            paper_id="fixture", layout=_layout(page), blocks=segment_page_result_blocks(page)
        )
    )
    assert document["status"] == "proposed-unconfirmed"
    assert "human confirmation" in document["warning"]
    assert document["counts"]["with_asserted_claim_type"] == len(document["proposals"])
    assert document["counts"]["needing_a_human_label"] == len(
        document["rows_needing_a_human_label"]
    )
    assert all(
        item["expected_claim_type"] is None for item in document["rows_needing_a_human_label"]
    )
